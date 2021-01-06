import argparse
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import yaml
import os
import shutil
from lib.nets import ImageSampler, ImageCritic, NetEnsemble, Density
from lib.distributions import Gaussian, TruncatedGaussian, Uniform, ProductDistribution, ImageDataset
import numpy as np
import scipy.stats as stats
import torch
import torchvision
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

class MNISTNetEnsemble(NetEnsemble):
    def __init__(self, device, reg_strength, transport_cost):
        self.device = device
        self.sampler = ImageSampler(input_im_size=32, output_im_size=32,
                                   downsampler_layers=5, upsampler_layers=5,
                                   downsampler_channels=[3, 64, 128, 256, 512, 128],
                                   upsampler_channels=[256, 512, 256, 128, 64, 1]).to(device)
        self.inp_density_parameter = ImageCritic(input_im_size=32, layers=4, channels=[3, 64, 128, 256, 512])
        self.outp_density_parameter = ImageCritic(input_im_size=32, layers=4, channels=[1, 64, 128, 256, 512])
        self.density = Density(self.inp_density_parameter, self.outp_density_parameter, regularization="entropy",
                               transport_cost=transport_cost, reg_strength=reg_strength).to(device)
        sampler_opt_critic = ImageCritic(input_im_size=32, layers=4, channels=[1, 64, 128, 256, 512]).to(device)
        self.sampler_opt = WassSamplerOptimizer(self.sampler, self.density, sampler_opt_critic,
                                                critic_steps=15, reg_strength=reg_strength, transport_cost=transport_cost)
    def save(self, path):
        self.save_net(self.sampler, os.path.join(path, f"sampler.pt"))
        self.save_net(self.inp_density_parameter, os.path.join(path, f"inp_density_parameter.pt"))
        self.save_net(self.outp_density_parameter, os.path.join(path, f"outp_density_parameter.pt"))
        self.save_net(self.sampler_opt.critic, os.path.join(path, f"critic.pt"))

    def load(self, path):
        self.load_net(self.sampler, os.path.join(path, f"sampler.pt"))
        self.load_net(self.sampler_opt.critic, os.path.join(path, f"critic.pt"))
        self.load_net(self.inp_density_parameter, os.path.join(path, f"inp_density_parameter.pt"))
        self.load_net(self.outp_density_parameter, os.path.join(path, f"outp_density_parameter.pt"))


class WassSamplerOptimizer():
    def __init__(self, sampler, density, critic, critic_steps, reg_strength, transport_cost):
        '''
        The SamplerOptimizer matches the sampler output density to the density induced by the density parameter.
        This implementation optimizes sampler in Wasserstein-1 distance.

        NOTE: transport_cost corresponds to the cost associated to the learned OT plan conditional density. It is NOT
        related to the transport cost which is used implicitly in the WGAN formulation, used here to fit the sampler
        density to the conditional density.
        '''
        self.sampler = sampler
        self.sampler_opt = torch.optim.RMSprop(sampler.parameters(), lr=0.00005)
        self.density = density
        self.critic = critic
        self.critic_opt = torch.optim.RMSprop(critic.parameters(), lr=0.00005)
        self.critic_steps = critic_steps
        self.reg_strength = reg_strength
        self.transport_cost = transport_cost
    def step(self, inp_batch, outp_batch, z_batch):
        for _ in range(self.critic_steps):
            self.critic_opt.step(lambda: self._critic_closure(inp_batch, outp_batch, z_batch))
            self.critic.clip_weights(0.01)
        return self.sampler_opt.step(lambda: self._sampler_closure(inp_batch, outp_batch, z_batch))

    def _sampler_closure(self, inp_batch, outp_batch, z_batch):
        self.sampler_opt.zero_grad()
        crit_fake = self.critic(self.sampler(inp_batch, z_batch))
        crit_real = self.critic(outp_batch)
        with torch.no_grad():
            density = self.density(inp_batch, outp_batch)
        #density = torch.ones_like(crit_real)
        obj = torch.mean(density * crit_real - crit_fake)
        obj.backward()
        return obj
    def _critic_closure(self, inp_batch, outp_batch, z_batch):
        self.critic_opt.zero_grad()
        crit_fake = self.critic(self.sampler(inp_batch, z_batch))
        crit_real = self.critic(outp_batch)
        with torch.no_grad():
            density = self.density(inp_batch, outp_batch)
        #density = torch.ones_like(crit_real)
        obj = torch.mean(density * crit_real - crit_fake)
        (-obj).backward() # for gradient ascent
        return obj


def run(inp_batch_iter, outp_batch_iter, z_batch_iter, net_ensemble, transport_cost,
    reg_strength, opt_iter_schedule, artifacts_path, device):
    '''
    Train a conditional OT sampler which provides samples from P(Y | X = x). The point x is an
    input point sampled from the distribution P and the point Y is an output point corresponding to X.

    We call P the 'input distribution' and Q the 'true output distribution.'
    Arguments:
        - inp_batch_iter: an iterator which produces batches of data from the input distribution P
        - outp_batch_iter: an iterator which produces batches of data from the output distribution Q
        - z_batch_iter: an iterator which produces batches of data from the latent code distribution
        - net_ensemble: a NetEnsemble class which provides the sampler, critic, and discriminator networks.
        - transport_cost: an function which takes two data tensors and returns a vector of transport costs.
            The vector has one dimension per input rows of data. The value of each dimension is
            the transport cost between corresp. rows of input data.
        - opt_iter_schedule: a tuple of four integers (N, G, D). Run N optimization steps, each including
            G steps of training the generator and D steps of training the OT plan conditional density.
        - reg_strength: the regularization parameter. As lambda increases the regularization strength decreases.
        - artifacts_path: path to a folder where experimental data is saved
    '''

    assert (len(opt_iter_schedule) == 3) and all([type(o) == int for o in opt_iter_schedule]),\
        "opt_iter_schedule must contain 3 integers"

    opt_loops, sampler_loops, density_loops = opt_iter_schedule

    sampler, density = net_ensemble.sampler, net_ensemble.density
    sampler_opt_manager = net_ensemble.sampler_opt

    density_p_opt = torch.optim.Adam(params = density.parameters(), lr=0.00001)

    def density_p_closure(inp_sample, outp_sample):
        density_p_opt.zero_grad()
        density_real_inp = density.inp_density_param(inp_sample)
        density_real_outp = density.outp_density_param(outp_sample)
        density_reg = density.penalty(inp_sample, outp_sample)
        obj = torch.mean(density_real_inp + density_real_outp - density_reg)
        (-obj).backward() # for gradient ascent rather than descent
        return obj

    writer = SummaryWriter(log_dir=artifacts_path)

    k = 5
    t_ex_inp_sample = torch.cat([torch.FloatTensor(next(inp_batch_iter)) for _ in range(k)], dim=0).to(device)
    t_ex_z_sample = torch.cat([torch.FloatTensor(next(z_batch_iter)) for _ in range(k)], dim=0).to(device)
    t_ex_outp_sample = torch.cat([torch.FloatTensor(next(outp_batch_iter)) for _ in range(k)], dim=0).to(device)

    ex_inp_grid = torchvision.utils.make_grid( (t_ex_inp_sample+1)/2)
    writer.add_image('Example Inputs', ex_inp_grid)
    ex__outp_grid = torchvision.utils.make_grid( (t_ex_outp_sample+1)/2)
    writer.add_image('Example Outputs', ex__outp_grid)


    def new_batch(inp_batch_iter, outp_batch_iter, z_batch_iter, device):
        def _safe_sample(itr):
            try:
                return itr, torch.FloatTensor(next(itr)).to(device)
            except StopIteration:
                fresh_itr = iter(itr)
                return fresh_itr, torch.FloatTensor(next(fresh_itr)).to(device)
        inp_batch_iter, inp_sample = _safe_sample(inp_batch_iter)
        outp_batch_iter, outp_sample = _safe_sample(outp_batch_iter)
        z_batch_iter, z_sample = _safe_sample(z_batch_iter)
        return inp_sample, outp_sample, z_sample


    d_val, s_val = 0, 0
    global_itr = 0
    for itr in range(opt_loops):
        for d_step in range(density_loops):
            global_itr = global_itr + 1
            inp_sample, outp_sample, _ = new_batch(inp_batch_iter, outp_batch_iter, z_batch_iter, device)
            d = density_p_opt.step(lambda: density_p_closure(inp_sample, outp_sample))
            d_val = round(d.item(), 5)
            print(f"\rO{global_itr} - Density Loss: {d_val}, Sampler: {s_val}", end="")
            writer.add_scalars('Optimization', {
                'Sampler': s_val,
                'Density': d_val
            }, global_itr)
        for s_step in range(sampler_loops):
            global_itr = global_itr + 1
            inp_sample, outp_sample, z_sample = new_batch(inp_batch_iter, outp_batch_iter, z_batch_iter, device)
            s = sampler_opt_manager.step(inp_sample, outp_sample, z_sample)
            s_val = round(s.item(), 5)
            print(f"\rO{global_itr} - Density Loss: {d_val}, Sampler: {s_val}", end="")
            writer.add_scalars('Optimization', {
                'Sampler': s_val,
                'Density': d_val
            }, global_itr)

        if(itr % 1 == 0):
            samples = sampler(t_ex_inp_sample, t_ex_z_sample)
            img_grid = torchvision.utils.make_grid((samples + 1 )/2)
            writer.add_image('Samples', img_grid, itr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Help text.")
    parser.add_argument("-n", "--name", type=str, help="Name of this experiment.")
    parser.add_argument("-y", "--yaml", type=str, help="Path of a yaml configuration file to use. If provided, this config will overwrite any arguments.")
    parser.add_argument("--overwrite", action="store_true", help="If true, overwrite previous experimental results with the same name")
    parser.add_argument("-d", "--dataset", type=str, help="Choose a dataset: digits")
    parser.add_argument("--use_cpu", action="store_true", help="If true, train on CPU.")
    parser.add_argument("-b", "--batch_size", type=int, default=50, help="Batch size for training the density and generator.")
    parser.add_argument("-r", "--reg_strength", type=float, default=0.1, help="Regularization strength.")
    args = parser.parse_args()

    args.device = 'cpu' if args.use_cpu else 'cuda'

    if args.yaml is not None:
        try:
            with open(args.yaml) as yaml:
                args = yaml.safe_load(yaml)
        except FileNotFoundError:
            raise FileNotFoundError("Could not load the provided yaml file.")

    path = os.path.join("artifacts", args.name)
    if(os.path.exists(path)):
        if(args.overwrite):
            shutil.rmtree(path)
        else:
            path += "_1"
            i = 1
            while os.path.exists(path):
                i = i + 1
                path = path.split("_")[0] + "_" + str(i)

    os.makedirs(path)

    args.yaml = os.path.join(path, "config.yml")

    with open(args.yaml, "w+") as f_out:
        yaml.dump(vars(args), f_out, default_flow_style=False, allow_unicode=True)

    bs = args.batch_size
    reg_strength = args.reg_strength
    if(args.dataset == "digits"):
        im_size = 32
        im_dim = 32*32*3
        P = ImageDataset(path="data/svhn", batch_size=bs, im_size=32)
        Q = ImageDataset(path="data/mnist", batch_size=bs, im_size=32, channels=1)
        R = ProductDistribution(P, Uniform(mu=0, batch_size=bs, data_dim=(1, 32, 32), bound=1))
        Z = Uniform(mu=0, batch_size=bs, data_dim=128, bound=1)
        c = lambda x, y: torch.mean((x-y)**2, dim=(1, 2, 3))[:, None]
        net_ensemble = MNISTNetEnsemble(args.device, reg_strength=reg_strength, transport_cost=c)

    net_ensemble.save(path)
    run(inp_batch_iter=P,
        outp_batch_iter=Q,
        z_batch_iter=Z,
        net_ensemble=net_ensemble,
        transport_cost=c,
        reg_strength=reg_strength,
        opt_iter_schedule=(40, 100, 100),
        artifacts_path=path,
        device=args.device)
    net_ensemble.save(path)
