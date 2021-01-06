import torch
from torch import nn



class ReLU_MLP(nn.Module):
    # MLP net with set input dimensionality, given intermediate layer dims, and fixed outputs
    def __init__(self, layer_dims, output="linear", layernorm=False):
        '''
        A generic ReLU MLP network.

        Arguments:
            - layer_dims: a list [d1, d2, ..., dn] where d1 is the dimension of input vectors and d1, ..., dn
                        is the dimension of outputs of each of the intermediate layers.
            - output: output activation function, either "sigmoid" or "linear".
            - layernorm: if True, apply layer normalization to the input of each layer.
        '''
        super(ReLU_MLP, self).__init__()
        layers = []
        for i in range(1, len(layer_dims) - 1):
            if (layernorm):
                layers.append(nn.LayerNorm(layer_dims[i - 1]))
            layers.append(nn.Linear(layer_dims[i - 1], layer_dims[i]))
            layers.append(nn.ReLU(layer_dims[i - 1]))
        if (output == "sigmoid"):
            layers.append(nn.Linear(layer_dims[-2], layer_dims[-1]))
            layers.append(nn.Sigmoid())
        if (output == "linear"):
            layers.append(nn.Linear(layer_dims[-2], layer_dims[-1]))

        self.out = nn.Sequential(*layers)

    def forward(self, input):
        return self.out(input)

class Joint_ReLU_MLP(ReLU_MLP):
    def __init__(self, layer_dims, output="linear", layernorm=False):
        super().__init__(layer_dims, output, layernorm)
    def forward(self, input, z):
        return super().forward(torch.cat((input, z), dim=1))

class ReLU_CNN(nn.Module):
    def __init__(self, imdims, channels, filter_size, output="sigmoid", layernorm=False):
        '''
        A generic ReLU CNN network.

        Arguments:
            - imdims: a length-2 tuple of integers, or a list of these tuples. Each is image dimensions in HW format.
                If input is a list, the list must have one fewer item than the length of channels. The output of each
                layer is resized to the given dimensions.
            - channels: a list [c1, c2, ..., cn] where c1 is the number of input channels and c2, ..., cn
                    is the number of output channels of each intermediate layer.

                    The final layer does not resize the image so len(channels) = len(imdims) + 1 is required.
            - filter_size: size of convolutional filters in each layer.
            - output: output activation function, either "sigmoid" or "tanh".
            - layernorm: if True, apply layer normalization to the input of each layer.
        '''

        super(ReLU_CNN, self).__init__()
        layers = []

        assert all([type(x) == int for x in channels]), "Channels must be a list of integers"
        def istwotuple(x):
            return (type(x) == tuple) and (len(x) == 2) and (type(x[0]) == int) and (type(x[1]) == int)

        if(istwotuple(imdims)):
            imdims = [imdims for _ in range(len(channels) + 1)]
        elif(all([istwotuple(x) for x in imdims])):
            assert len(imdims)+1 == len(channels), "The length of channels must be one greater than the length of imdims."
        else:
            raise ValueError("Input image dimensions are not correctly formatted.")

        self.imdims = imdims

        padding = int((filter_size - 1) / 2)
        for i in range(1, len(channels) - 1):
            if (layernorm and not i == 1):
                input_shape = (channels[i - 1], self.imdims[i - 1][0], self.imdims[i - 1][1])
                #layers.append(nn.LayerNorm(input_shape))
                layers.append(nn.BatchNorm2d(num_features=channels[i-1]))
            layers.append(nn.Conv2d(channels[i - 1], channels[i], kernel_size=filter_size, padding=padding))
            if(imdims[i-1] != imdims[i]):
                layers.append(torch.nn.Upsample(imdims[i], mode='bilinear', align_corners=True))
            layers.append(nn.ReLU(channels[i - 1]))
        if (output == "sigmoid"):
            layers.append(nn.Conv2d(channels[-2], channels[-1], kernel_size=filter_size, padding=padding))
            layers.append(nn.Sigmoid())
        elif (output == "tanh"):
            layers.append(nn.Conv2d(channels[-2], channels[-1], kernel_size=filter_size, padding=padding))
            layers.append(nn.Tanh())
        elif (output == "none"):
            layers.append(nn.Conv2d(channels[-2], channels[-1], kernel_size=filter_size, padding=padding))
        else:
            raise ValueError("Unrecognized output function.")
        self.layers = layers
        self.out = nn.Sequential(*layers)

    def forward(self, inp):
        return self.out(inp)

    def clip_weights(self, c):
        for layer in self.layers:
            if(isinstance(layer, nn.Conv2d) or isinstance(layer, nn.LayerNorm)):
                layer.weight.data = torch.clamp(layer.weight.data, -c, c)
                layer.bias.data = torch.clamp(layer.bias.data, -c, c)


class ImageCritic(nn.Module):
    def __init__(self, input_im_size, layers, channels):
        '''
        The critic accepts an image or batch of images and outputs a scalar or batch of scalars.
        Given the input image size and a desired number of layers, the input image is downsampled at each layer until
            is a set of 4x4 feature maps. The scalar output is a regression over the channel dimension.

        Arguments:
            - input_im_size (int): size of input images, which are assumed to be square
            - layers (int): number of layers
            - channels (int):
        '''
        super(ImageCritic, self).__init__()


        scale = (input_im_size/4)**(-1/ (layers-1) )

        imdims = [input_im_size] + \
                 [int(input_im_size * (scale**k)) for k in range(1, layers-1)]  + \
                 [4]

        imdims = [(x, x) for x in imdims]

        self.net = ReLU_CNN(imdims, channels, filter_size=3, output="none", layernorm=True)
        self.linear = torch.nn.Linear(4 * 4 * channels[-1], 1)
    def forward(self, image, *args):
        # the *args is a hack to allow extra arguments, for ex, if one wants to pass two input images.
        # TODO: nicely implement functionality for networks which take in two or more arguments.
        if(len(args) > 0):
            image = torch.cat([image] + list(args), dim=1)
        return self.linear(torch.flatten(self.net(image), start_dim=1))
    def clip_weights(self, c):
        for p in self.parameters():
            p.data.clamp_(-c, c)

class ImageSampler(nn.Module):
    def __init__(self, input_im_size, output_im_size, downsampler_layers, upsampler_layers, downsampler_channels, upsampler_channels):
        '''
        The sampler U-Net accepts an image and a latent code. The image is input to a convolutional network
            which outputs a second latent code. Both latent codes are fed to an upsampling network to produce
            an output image.

        The final number of channels in the last downsampling layer is the dimensionality of the second latent code.
        The difference between this number and the number of channels of the first upsampling layer is the dimensionality
            of the first latent code.

        Arguments:
            - input_im_size (int): side length of input images, which are assumed square
            - output_im_size (int): side length of output images, which are assumed square
            - downsampler_layers (int): total number of downsampling layers. Each downsamples by a constant factor,
                reducing side length from input_im_size to 1.
            - upsampler_layers (int): total number of upsampling layers. Each upsamples by a constant factor,
                increasing side length from 1 to output_im_size.
            - downsampler_channels (int): number of channels in each downsampling layer
            - upsampler_channels (int): number of channels in each upsampling layer
        '''
        super(ImageSampler, self).__init__()
        self.latent_dim = upsampler_channels[0] - downsampler_channels[-1]
        assert self.latent_dim > 0, "The input latent code must have dimensionality greater than zero."

        # the offset -1 for layer counts is because the final layer does not include an upsampling operation
        # which is to increase output fidelity by avoiding upsampling blur in the final output
        downsampler_scale = (input_im_size)**(-1/ (downsampler_layers-1) )
        upsampler_scale = (output_im_size)**(1/(upsampler_layers-1) )

        downsampler_imdims = [input_im_size] + \
                             [int(input_im_size * (downsampler_scale**k)) for k in range(1, downsampler_layers-1)] + \
                             [1]
        downsampler_imdims = [(x, x) for x in downsampler_imdims]
        upsampler_imdims = [1] + \
                             [int(upsampler_scale**k) for k in range(1, upsampler_layers-1)] + \
                             [output_im_size]
        upsampler_imdims = [(x, x) for x in upsampler_imdims]

        self.downsampler = ReLU_CNN(imdims=downsampler_imdims, channels=downsampler_channels, filter_size=3, output="none", layernorm=True)
        self.upsampler = ReLU_CNN(imdims=upsampler_imdims, channels=upsampler_channels, filter_size=3, output="tanh", layernorm=True)

    def forward(self, input_image, latent):
        if(len(latent.shape) == 2):
            latent = latent[:, :, None, None]
        double_latent = torch.cat((self.downsampler(input_image), latent), dim=1)
        return self.upsampler(double_latent)

class Density(nn.Module):
    def __init__(self, inp_density_param, outp_density_param, regularization, reg_strength, transport_cost):
        '''
        Learn a density over the data by training an ImageCritic to solve a regularized OT problem.
        The ImageCritic induces a data density via a regularization specific function.

        Arguments:
            - inp_density_parameter (ImageCritic): an instantiated image critic which represents input density paramater
            - outp_density_parameter (ImageCritic):  an instantiated image critic which represents output density parameter
            - regularization (str): 'entropy' or 'L2'
            - reg_strength (float): weight of the regularization term
            - transport_cost (func: Image_Batch x Image_Batch -> vector): pairwise transport cost of images in two batches
        '''
        super(Density, self).__init__()
        self.transport_cost = transport_cost

        # based on regularization: need a method to output a density
        # and a method to output a regularization value, to be used in a dual objective
        r = reg_strength
        if(regularization == "entropy"):
            self.penalty_fn = lambda x, y: r * torch.exp((1/r)*self._violation(x, y) - 1)
            self.density_fn = lambda x, y: torch.exp((1/r)*self._violation(x, y) - 1)
        elif(regularization == "l2"):
            self.penalty_fn = lambda x, y: (1/(4*r)) * torch.relu(self._violation(x, y))**2
            self.density_fn = lambda x, y: (1/(2*r)) * torch.relu(self._violation(x, y))
        else:
            raise ValueError("Invalid choice of regularization")

        self.inp_density_param_net = inp_density_param
        self.outp_density_param_net = outp_density_param

    def _violation(self, x, y):
        return self.inp_density_param_net(x) + self.outp_density_param_net(y) - self.transport_cost(x, y)

    def penalty(self, x, y):
        return self.penalty_fn(x, y)

    def forward(self, x, y):
        return self.density_fn(x, y)

    def inp_density_param(self, x):
        return self.inp_density_param_net(x)

    def outp_density_param(self, y):
        return self.outp_density_param_net(y)

class NetEnsemble():
    '''
    Base class for net ensemble classes which group together multiple neural networks.
    '''
    def save_net(self, net, path):
        torch.save(net.state_dict(), path)
    def load_net(self, net, path):
        net.load_state_dict(torch.load(path))
if __name__ == "__main__":
    import numpy as np

    M = ReLU_MLP([3, 4, 4], output="linear", layernorm=True)
    C = ReLU_CNN((32, 32), [3, 4, 4], 5, output="linear", layernorm=True)
    U = ImageSampler(input_im_size=32, output_im_size=32,
                    downsampler_layers=3, upsampler_layers=3,
                    downsampler_channels=[3, 10, 10, 5],
                    upsampler_channels=[10, 10, 10, 3])

    m = torch.FloatTensor(np.random.normal(size=(20, 3)))
    c = torch.FloatTensor(np.random.normal(size=(20, 3, 32, 32)))
    z = torch.FloatTensor(np.random.normal(size=(20, 5)))

    m_out = M(m)
    c_out = C(c)
    _ = U(c, z)
