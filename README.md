Hypothetical paper on Conditional Optimal Transport

**Method**: introduce the sampler and the optimization scheme. 
- Architecture
- Motivation
- *Key benefit*: we learn a sampler directly rather than learning a density

**Experiments**: image generation, domain adaptation 
- Would need to show good performance in domain adaptation

**Theory**:
- Show that solving discriminator and critic problems gives accurate gradients of the (strongly convex) primal
- Analyze the extent to which soft penalty enforcement via Lagrangians yields an error margin conditional output density.