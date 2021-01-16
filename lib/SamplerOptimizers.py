import torch
import os

class SinkSamplerOptimizer():
    def __init__(self, sampler, compatibiilty, cross_density, negentropy_density, critic_steps):
        '''
        Learn the sampler using Sinkhorn Divergence:

        Compute 2OT(a, b) - OT(a, a) - OT(b, b) and minimize with respect to a.
        '''
        pass

class BPSamplerOptimizer():
    def __init__(self, sampler, compatibility, reg_strength):
        '''
        Compute the barycentric projection of the learned OT plan.
        '''
        self.sampler = sampler
        self.sampler_opt = torch.optim.RMSprop(sampler.parameters(), lr=0.0005)
        self.compatibility = compatibility
        self.reg_strength = reg_strength
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

