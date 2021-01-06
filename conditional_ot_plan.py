import argparse
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import yaml
import os
import shutil
from lib.nets import ImageSampler, ImageCritic, NetEnsemble
from lib.distributions import Gaussian, TruncatedGaussian, Uniform, ProductDistribution, ImageDataset
import numpy as np
import scipy.stats as stats
import torch
import torchvision
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

class MNISTNetEnsemble(NetEnsemble):
    def __init__(self, device):
        self.device = device
        self.sampler = ImageSampler(input_im_size=32, output_im_size=32,
                                   downsampler_layers=5, upsampler_layers=5,
                                   downsampler_channels=[3, 64, 128, 256, 512, 128],
                                   upsampler_channels=[256, 512, 256, 128, 64, 1]).to(device)
        self.critic = ImageCritic(input_im_size=32, layers=4, channels=[1, 64, 128, 256, 512]).to(device)
        self.discriminator = ImageCritic(input_im_size=32, layers=4, channels = [4, 64, 128, 256, 512]).to(device)

    def save(self, path):
        self.save_net(self.sampler, os.path.join(path, f"sampler.pt"))
        self.save_net(self.critic, os.path.join(path, f"critic.pt"))
        self.save_net(self.discriminator, os.path.join(path, f"discriminator.pt"))

    def load(self, path):
        self.load_net(self.sampler, os.path.join(path, f"sampler.pt"))
        self.load_net(self.critic, os.path.join(path, f"critic.pt"))
        self.load_net(self.discriminator, os.path.join(path, f"discriminator.pt"))

def run(P_batch_iter, Q_batch_iter, ref_batch_iter, z_batch_iter, net_ensemble, transport_cost, opt_iter_schedule, artifacts_path, device):
    '''
    Train a conditional OT sampler which provides samples from P(Y | X = x). The point x is an
    input point sampled from the distribution P and the point Y is an output point corresponding to X.

    We call P the 'input distribution' and Q the 'true output distribution.'
    Arguments:
        - P_batch_iter: an iterator which produces batches of data from the input distribution P
        - Q_batch_iter: an iterator which produces batches of data from the output distribution Q
        - net_ensemble: a NetEnsemble class which provides the sampler, critic, and discriminator networks.
        - transport_cost: an function which takes two data tensors and returns a vector of transport costs.
            The vector has one dimension per input rows of data. The value of each dimension is
            the transport cost between corresp. rows of input data.
        - opt_iter_schedule: a tuple of four integers (N, S, C, D). Run N optimization steps, each including
            S sampler steps, C critic steps, D discriminator steps, in that order.
        - artifacts_path: path to a folder where experimental data is saved
    '''

    assert (len(opt_iter_schedule) == 4) and all([type(o) == int for o in opt_iter_schedule]),\
        "opt_iter_schedule must contain 4 integers"

    opt_loops, sampler_steps, critic_steps, discriminator_steps = opt_iter_schedule

    # TODO: add support for either choosing or checking the batch size of output from batch iterators
    inp_data_dim = P_batch_iter.data_dim
    outp_data_dim = Q_batch_iter.data_dim
    ref_data_dim = ref_batch_iter.data_dim
    z_data_dim = z_batch_iter.data_dim

    sampler, critic, discriminator = net_ensemble.sampler, net_ensemble.critic, net_ensemble.discriminator

    sampler_opt = torch.optim.RMSprop(params = sampler.parameters(), lr=0.00005)
    critic_opt = torch.optim.RMSprop(params = critic.parameters(), lr=0.00005)
    discriminator_opt = torch.optim.RMSprop(params = discriminator.parameters(), lr=0.00005)


    # cf. f-GAN paper -- here we hardcode a choice of f divergence by choosing an activation function.
    # specifically, we use KL divergence from reference distribution
    gf = lambda x: x
    f = lambda x: torch.exp(x - 1)

    # inf me
    def sampler_closure(inp_sample, z_sample, true_outp_sample, ref_inp_sample, ref_outp_sample):
        sampler_opt.zero_grad()
        s = 1/500 * transport_cost(inp_sample, sampler(inp_sample, z_sample))
        c = torch.mean(critic(true_outp_sample) - critic(sampler(inp_sample, z_sample)), dim=0)
        d = torch.mean( gf(discriminator(ref_inp_sample, ref_outp_sample)) - \
                      f(discriminator(inp_sample, sampler(inp_sample, z_sample))), dim=0)

        total_loss = s + c + d
        total_loss.backward()
        return total_loss

    # sup me
    def critic_closure(inp_sample, z_sample, true_outp_sample):
        critic_opt.zero_grad()
        c = torch.mean(critic(true_outp_sample) - critic(sampler(inp_sample, z_sample)), dim=0)
        (-c).backward() # for gradient ascent rather than descent
        return c

    # sup me
    def discriminator_closure(ref_inp_sample, ref_outp_sample, inp_sample, z_sample):
        discriminator_opt.zero_grad()

        d = torch.mean( gf(discriminator(ref_inp_sample, ref_outp_sample)) - \
                       f(discriminator(inp_sample, sampler(inp_sample, z_sample))), dim=0)

        (-d).backward() # for gradient ascent rather than descent
        return d

    writer = SummaryWriter(log_dir=artifacts_path)

    k = 5
    t_ex_inp_sample = torch.cat([torch.FloatTensor(next(P_batch_iter)) for _ in range(k)], dim=0).to(device)
    t_ex_z_sample = torch.cat([torch.FloatTensor(next(z_batch_iter)) for _ in range(k)], dim=0).to(device)
    t_ex_outp_sample = torch.cat([torch.FloatTensor(next(Q_batch_iter)) for _ in range(k)], dim=0).to(device)

    ex_inp_grid = torchvision.utils.make_grid(t_ex_inp_sample)
    writer.add_image('Example Inputs', ex_inp_grid)
    ex__outp_grid = torchvision.utils.make_grid(t_ex_outp_sample)
    writer.add_image('Example Outputs', ex__outp_grid)

    for itr in range(opt_loops):
        try:
            inp_sample = torch.FloatTensor(next(P_batch_iter)).to(device)
        except StopIteration:
            P_batch_iter = iter(P_batch_iter)
            inp_sample = torch.FloatTensor(next(P_batch_iter)).to(device)
        try:
            outp_sample = torch.FloatTensor(next(Q_batch_iter)).to(device)
        except StopIteration:
            Q_batch_iter = iter(Q_batch_iter)
            outp_sample = torch.FloatTensor(next(Q_batch_iter)).to(device)
        try:
            z_sample = torch.FloatTensor(next(z_batch_iter)).to(device)
        except StopIteration:
            z_batch_iter = iter(z_batch_iter)
            z_sample = torch.FloatTensor(next(z_batch_iter)).to(device)

        ref_inp_sample, ref_outp_sample = ref_batch_iter.next_tuple()
        ref_inp_sample = torch.FloatTensor(ref_inp_sample).to(device)
        ref_outp_sample = torch.FloatTensor(ref_outp_sample).to(device)

        s_val, c_val, d_val = 0, 0, 0
        for s_step in range(sampler_steps):
            s = sampler_opt.step(lambda: sampler_closure(inp_sample, z_sample, outp_sample, ref_inp_sample, ref_outp_sample))
            s_val = round(s.item(), 5)
            print(f"\rO{itr} - S{s_step} - Sampler: {s_val}, Critic: {c_val}, Disc: {d_val}", end="")
        for c_step in range(critic_steps):
            c = critic_opt.step(lambda: critic_closure(inp_sample, z_sample, outp_sample))
            c_val = round(c.item(), 5)
            print(f"\rO{itr} - C{s_step} - Sampler: {s_val}, Critic: {c_val}, Disc: {d_val}", end="")
        for d_step in range(discriminator_steps):
            d = discriminator_opt.step(lambda: discriminator_closure(ref_inp_sample, ref_outp_sample, inp_sample, z_sample))
            d_val = round(d.item(), 5)
            print(f"\rO{itr} - D{s_step} - Sampler: {s_val}, Critic: {c_val}, Disc: {d_val}", end="")

        writer.add_scalars('Optimization', {
            'sampler': s.item(),
            'critic': c.item(),
            'discriminator': d.item()
        }, itr)

        if(itr % 100 == 0):
            '''
            plt.scatter(*ex_inp_sample.T, color="b", label="Input")
            samples = sampler(t_ex_inp_sample, t_ex_z_sample).detach().cpu().numpy()
            plt.scatter(*samples.T, color="r", label="Samples")
            plt.scatter(*ex_outp_sample.T, color="g", label="Output")
            plt.legend()
            plt.title(f"OT Sampler - {iter} iterations")
            plt.savefig(os.path.join(artifacts_path, f"Samples_{iter}.png"))
            plt.clf()
            '''
            samples = sampler(t_ex_inp_sample, t_ex_z_sample)
            img_grid = torchvision.utils.make_grid(samples)
            writer.add_image('Samples', img_grid, itr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Help text.")
    parser.add_argument("-n", "--name", type=str, help="Name of this experiment.")
    parser.add_argument("-y", "--yaml", type=str, help="Path of a yaml configuration file to use. If provided, this config will overwrite any arguments.")
    parser.add_argument("--overwrite", action="store_true", help="If true, overwrite previous experimental results with the same name")
    parser.add_argument("-d", "--dataset", type=str, help="Choose a dataset: digits")
    parser.add_argument("--use_cpu", action="store_true", help="If true, train on CPU.")
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

    bound = 2
    bs = 50
    latent_dim = 10

    if(args.dataset == "digits"):
        im_size = 32
        im_dim = 32*32*3
        P = ImageDataset(path="data/svhn", batch_size=bs, im_size=32)
        Q = ImageDataset(path="data/mnist", batch_size=bs, im_size=32, channels=1)
        R = ProductDistribution(P, Uniform(mu=0, batch_size=bs, data_dim=(1, 32, 32), bound=1))
        Z = Uniform(mu=0, batch_size=bs, data_dim=128, bound=1)
        c = lambda x, y: torch.mean((x-y)**2)
        net_ensemble = MNISTNetEnsemble(args.device)

    run(P_batch_iter=P,
        Q_batch_iter=Q,
        ref_batch_iter=R,
        z_batch_iter=Z,
        net_ensemble=net_ensemble,
        transport_cost=c,
        opt_iter_schedule=(20000, 10, 1, 1),
        artifacts_path=path,
        device=args.device)