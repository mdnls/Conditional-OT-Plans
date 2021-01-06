import argparse
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import yaml
import os
import shutil
from lib.nets import ReLU_CNN, ReLU_MLP, Joint_ReLU_MLP, ImageSampler, ImageCritic
from lib.distributions import Gaussian, TruncatedGaussian, Uniform, ProductDistribution, ImageDataset
import numpy as np
import scipy.stats as stats
import torch
import torchvision
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter


class MNISTNetEnsemble():
    def __init__(self, device):
        self.device = device
        self.sampler = ImageSampler(input_im_size=32, output_im_size=32,
                                   downsampler_layers=5, upsampler_layers=5,
                                   downsampler_channels=[3, 64, 128, 256, 512, 128],
                                   upsampler_channels=[256, 512, 256, 128, 64, 1]).to(device)
        self.critic = ImageCritic(input_im_size=32, layers=4, channels=[1, 64, 128, 256, 512]).to(device)


def run(Q_batch_iter, P_batch_iter, z_batch_iter, net_ensemble, opt_iter_schedule, artifacts_path, device):

    assert (len(opt_iter_schedule) == 3) and all([type(o) == int for o in opt_iter_schedule]),\
        "opt_iter_schedule must contain 3 integers"

    opt_loops, sampler_steps, critic_steps = opt_iter_schedule
    sampler, critic = net_ensemble.sampler, net_ensemble.critic

    sampler_opt = torch.optim.RMSprop(params = sampler.parameters(), lr=0.00005)
    critic_opt = torch.optim.RMSprop(params = critic.parameters(), lr=0.00005)

    # inf me
    def sampler_closure(inp_sample, z_sample, true_outp_sample):
        sampler_opt.zero_grad()
        total_loss = torch.mean(critic(true_outp_sample) - critic(sampler(inp_sample, z_sample)), dim=0)
        total_loss.backward()
        return total_loss

    # sup me
    def critic_closure(inp_sample, z_sample, true_outp_sample):
        critic_opt.zero_grad()
        total_loss = torch.mean(critic(true_outp_sample) - critic(sampler(inp_sample, z_sample)), dim=0)
        (-total_loss).backward() # for gradient ascent rather than descent
        return total_loss

    writer = SummaryWriter(log_dir=artifacts_path)

    inp_batch_size = next(z_batch_iter).shape[0]
    outp_batch_size = next(Q_batch_iter).size()[0]
    dummy_sampler_inp = torch.zeros((outp_batch_size, 3, 32, 32)).to(device)

    k = 5
    t_ex_z_sample = torch.cat([torch.FloatTensor(next(z_batch_iter)) for _ in range(k)], dim=0).to(device)
    t_ex_outp_sample = torch.cat([torch.FloatTensor(next(Q_batch_iter)) for _ in range(k)], dim=0).to(device)
    t_ex_inp_sample = torch.zeros((k * inp_batch_size, 3, 32, 32)).to(device)

    ex_outp_grid = torchvision.utils.make_grid( (t_ex_outp_sample + 1)/2 )
    writer.add_image('Example Outputs', ex_outp_grid)

    for itr in range(opt_loops):
        try:
            outp_sample = torch.FloatTensor(next(Q_batch_iter)).to(device)
        except StopIteration:
            Q_batch_iter = iter(Q_batch_iter)
            outp_sample = torch.FloatTensor(next(Q_batch_iter)).to(device)
        try:
            inp_sample = torch.FloatTensor(next(P_batch_iter)).to(device)
        except StopIteration:
            P_batch_iter = iter(P_batch_iter)
            inp_sample = torch.FloatTensor(next(P_batch_iter)).to(device)
        try:
            z_sample = torch.FloatTensor(next(z_batch_iter)).to(device)
        except StopIteration:
            z_batch_iter = iter(z_batch_iter)
            z_sample = torch.FloatTensor(next(z_batch_iter)).to(device)

        s_val, c_val, d_val = 0, 0, 0
        for s_step in range(sampler_steps):
            s = sampler_opt.step(lambda: sampler_closure(inp_sample, z_sample, outp_sample))
            s_val = round(s.item(), 5)
            print(f"\rO{itr} - S{s_step} - Sampler: {s_val}, Critic: {c_val}, Disc: {d_val}", end="")
        for c_step in range(critic_steps):
            c = critic_opt.step(lambda: critic_closure(inp_sample, z_sample, outp_sample))
            critic.clip_weights(0.01)
            c_val = round(c.item(), 5)
            print(f"\rO{itr} - C{s_step} - Sampler: {s_val}, Critic: {c_val}, Disc: {d_val}", end="")

        writer.add_scalars('Optimization', {
            'sampler': s.item(),
            'critic': c.item(),
            'Min Pixel Value': sampler(t_ex_inp_sample, t_ex_z_sample).min()
        }, itr)

        if(itr % 100 == 0):
            samples = sampler(t_ex_inp_sample, t_ex_z_sample)
            img_grid = torchvision.utils.make_grid( (samples+1)/2 )
            writer.add_image('Samples', img_grid, itr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Help text.")
    parser.add_argument("-n", "--name", type=str, help="Name of this experiment.")
    parser.add_argument("-y", "--yaml", type=str, help="Path of a yaml configuration file to use. If provided, this config will overwrite any arguments.")
    parser.add_argument("--overwrite", action="store_true", help="If true, overwrite previous experimental results with the same name")
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
    latent_dim = 128

    im_size = 32
    im_dim = 32*32*3
    P = ImageDataset(path='data/svhn', batch_size=bs, im_size=32, channels=3)
    Q = ImageDataset(path="data/mnist", batch_size=bs, im_size=32, channels=1)
    Z = Uniform(mu=0, bound=1, batch_size=bs, data_dim=128)

    net_ensemble = MNISTNetEnsemble(device='cuda')

    run(Q_batch_iter=Q,
        P_batch_iter=P,
        z_batch_iter=Z,
        net_ensemble=net_ensemble,
        opt_iter_schedule=(20000, 1, 5),
        artifacts_path=path,
        device=args.device)