import torch
import os

class SinkSamplerOptimizer():
    def __init__(self, sampler, target_cpat, gan_training_cpats, critic_steps):
        '''
        Learn the sampler using Sinkhorn Divergence:

        Compute 2OT(a, b) - OT(a, a) - OT(b, b) and minimize with respect to a. This requires computing gradients of
        the first two terms, each of which gets a dedicated compatibility function.

        Arguments:
            - sampler (ImageSampler): optimize this sampler to be a GAN
            - target_cpat (Compatibility): the compatibility function of the true OT plan which the generator should learn.
            - gan_training_cpats (tuple of two Compatibility): two compatibility instances to be trained during the GAN
                optimization process.
            - critic_steps (int): number of optimization steps applied to each dual variable of each compatibility function
                per single GAN step.
        '''
        assert len(gan_training_cpats) == 2, "GAN_training_cpats must be a tuple of two compatibility functions."
        self.sampler = sampler
        self.target_cpat = target_cpat
        self.cross_cpat = gan_training_cpats[0]
        self.negentropy_cpat = gan_training_cpats[1]
        self.critic_steps = critic_steps

        # todo: give density parameters a shorter name
        params = sum([self.cross_cpat.input_density_param_net.parameters(),
                      self.cross_cpat.output_density_param_net.parameters(),
                      self.negentropy_cpat.input_density_param_net.parameters(),
                      self.negentropy_cpat.output_density_param_net.parameters()], [])
        self.critic_opt = torch.optim.RMSprop(params = params, lr=0.0005)
        self.sampler_opt = torch.optim.RMSprop(params = self.sampler.params(), lr=0.0005)

    def step(self, inp_sample, outp_sample, z_sample):
        for _ in range(self.critic_steps):
            self.critic_opt.step(lambda: self._critic_closure(inp_sample, outp_sample, z_sample))
        return self.sampler_opt.step(lambda: self._sampler_closure(inp_sample, outp_sample, z_sample))

    def _critic_closure(self, inp_sample, outp_sample, z_sample):
        # 1. optimize cross OT cost:
        c_density_real_inp = self.cross_cpat.inp_density_param(inp_sample)
        c_density_real_outp = self.cross_cpat.outp_density_param(outp_sample)
        c_density_reg = self.cross_cpat.penalty(inp_sample, outp_sample)
        c_obj = torch.mean(c_density_real_inp + c_density_real_outp - c_density_reg)

        # 2. optimize negentropy OT cost:
        ne_density_real_inp = self.negentropy_cpat.inp_density_param(inp_sample)
        ne_density_real_outp = self.negentropy_cpat.outp_density_param(outp_sample)
        ne_density_reg = self.negentropy_cpat.penalty(inp_sample, outp_sample)
        ne_obj = torch.mean(ne_density_real_inp + ne_density_real_outp - ne_density_reg)

        (ne_obj - c_obj).backward() # for gradient ascent rather than descent

class BPSamplerOptimizer():
    def __init__(self, sampler, compatibility):
        '''
        Compute the barycentric projection of the learned OT plan.
        '''
        self.sampler = sampler
        self.sampler_opt = torch.optim.RMSprop(sampler.parameters(), lr=0.0005)
        self.compatibility = compatibility
        self.transport_cost = lambda x, y: torch.mean((x-y)**2, dim=(1, 2, 3))[:, None]

    def step(self, inp_batch, outp_batch, z_batch):
        self.sampler_opt.zero_grad()
        dummy_latents = torch.zeros_like(z_batch)
        loss = torch.mean(self.transport_cost(self.sampler(inp_batch, dummy_latents), outp_batch) * self.compatibility(inp_batch, outp_batch))
        loss.backward()
        self.sampler_opt.step()
        return loss

    def save(self, path):
        torch.save(self.sampler, os.path.join(path, 'bary_map.pt'))

    def load(self, path):
        try:
            self.sampler.load_state_dict(torch.load(os.path.join(path, 'bary_map.pt')))
        except FileNotFoundError:
            print("Starting from randomly initialized map.")

class WSamplerOptimizer():
    def __init__(self, sampler, compatibility, critic, critic_steps):
        '''
        The SamplerOptimizer matches the sampler output density to the learned OT plan.
        This implementation optimizes sampler in Wasserstein-1 distance.

        NOTE: transport_cost corresponds to the cost associated to the learned OT plan conditional density. It is NOT
        related to the transport cost which is used implicitly in the WGAN formulation, used here to fit the sampler
        density to the conditional density.
        '''
        self.sampler = sampler
        self.sampler_opt = torch.optim.RMSprop(sampler.parameters(), lr=0.00005)
        self.compatibility = compatibility
        self.critic = critic
        self.critic_opt = torch.optim.RMSprop(critic.parameters(), lr=0.00005)
        self.critic_steps = critic_steps

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
            cpat = self.compatibility(inp_batch, outp_batch)
        # self reweighted importance sampling
        obj = torch.sum(cpat * crit_real)/torch.sum(cpat) - torch.mean(crit_fake)
        obj.backward()
        return obj
    def _critic_closure(self, inp_batch, outp_batch, z_batch):
        self.critic_opt.zero_grad()
        crit_fake = self.critic(self.sampler(inp_batch, z_batch))
        crit_real = self.critic(outp_batch)
        with torch.no_grad():
            cpat = self.compatibility(inp_batch, outp_batch)
        # self reweighted importance sampling
        obj = torch.sum(cpat * crit_real)/torch.sum(cpat) - torch.mean(crit_fake)
        (-obj).backward() # for gradient ascent
        return obj

    def save(self, path):
        torch.save(self.critic.state_dict(), os.path.join(path, 'critic.pt'))

    def load(self, path):
        try:
            self.critic.load_state_dict(torch.load(os.path.join(path, f"critic.pt")))
        except FileNotFoundError:
            print("Starting from randomly initialized critic.")

class WGPSamplerOptimizer(WSamplerOptimizer):
    def __init__(self, sampler, compatibility, critic, critic_steps):
        super().__init__(sampler, compatibility, critic, critic_steps)
        self.gp_weight = 10

    def step(self, inp_batch, outp_batch, z_batch):
        for _ in range(self.critic_steps):
            self.critic_opt.step(lambda: self._critic_closure(inp_batch, outp_batch, z_batch))
        return self.sampler_opt.step(lambda: self._sampler_closure(inp_batch, outp_batch, z_batch))

    def _critic_closure(self, inp_batch, outp_batch, z_batch):
        self.critic_opt.zero_grad()
        bs = outp_batch.shape[0]
        samples = self.sampler(inp_batch, z_batch)
        gp_mix_weights = torch.rand(size=(bs, 1, 1, 1), device=outp_batch.device)
        gp_mix = gp_mix_weights * outp_batch + (1-gp_mix_weights) * samples
        grad = torch.autograd.grad(outputs=list(self.critic(gp_mix)), inputs=gp_mix,
                                   create_graph=True, retain_graph=True)[0]

        grad_norms = torch.sqrt(torch.sum(grad**2, dim=(1, 2, 3)) + 1e-12)
        gp = self.gp_weight * torch.sum((grad_norms - torch.ones((bs,), device=outp_batch.device) )**2)
        crit_fake = self.critic(samples)
        crit_real = self.critic(outp_batch)
        with torch.no_grad():
            cpat = self.compatibility(inp_batch, outp_batch)
        # self reweighted importance sampling
        obj = torch.sum(cpat * crit_real)/torch.sum(cpat) - torch.mean(crit_fake)
        (gp - obj).backward() # for gradient ascent
        return obj

