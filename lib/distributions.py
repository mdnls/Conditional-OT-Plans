import numpy as np
from scipy import stats
import torch
import torchvision

class Gaussian():
    def __init__(self, mu, sigma, batch_size, data_dim):
        '''
        Gaussian distribution.

        Arguments:
            - mu (np.ndarray): the mean of the distribution, same shape as data_dim
            - sigma (number): the standard deviation of each coordinate of the distribution
            - batch_size (int): number of samples per batch
            - data_dim (int) or list of ints: the dimensionality of output data
        '''
        if(type(data_dim) == int):
            data_dim = (data_dim,)
        self.mu = mu
        self.sigma = sigma
        self.batch_size = batch_size
        self.data_dim = data_dim

    def __iter__(self):
        return self

    def __next__(self):
        return np.random.normal(loc=self.mu, scale=self.sigma, size= (self.batch_size,) + self.data_dim)

class TruncatedGaussian(Gaussian):
    def __init__(self, mu, sigma, batch_size, data_dim, bound):
        '''
        Truncated Gaussian distribution.

        Arguments:
            - mu, sigma, batch_size, data_dim: see Gaussian.
            - bound: a scalar. Samples which deviate from the mean by a magnitude larger than this number are
                thrown out.
        '''
        super().__init__(mu, sigma, batch_size, data_dim)
        self.bound = bound
    def __iter__(self):
        return self
    def __next__(self):
        return stats.truncnorm.rvs(loc=self.mu, scale=self.sigma,
                                   size=(self.batch_size,) + self.data_dim,
                                   a=-self.bound, b=self.bound)


class Uniform():
    def __init__(self, mu, batch_size, data_dim, bound):
        '''
        Uniform distribution.

        Arguments:
            - mu (number): the mean of the uniform distribution, same shape as data_dim
            - batch_size (int): number of samples per batch
            - data_dim (int) or list of ints: the dimensionality of output data
            - bound (number): each coordinate of each point is sampled from (mu - bound, mu + bound)
        '''
        if(type(data_dim) == int):
            data_dim = (data_dim,)
        self.mu = mu
        self.batch_size = batch_size
        self.data_dim = data_dim
        self.bound = bound
    def __iter__(self):
        return self
    def __next__(self):
        return stats.uniform.rvs(loc=self.mu - self.bound, scale=2 * self.bound,
                                   size=(self.batch_size,) + self.data_dim)

class ProductDistribution():
    def __init__(self, *args):
        '''
        Construct a product distribution whose output samples are tuples where each component
            is a sample from each input distribution.
        '''
        data_batch_sizes = [dist.batch_size for dist in args]
        assert all([dist.batch_size == args[0].batch_size for dist in args]),\
            "Product distribution requires input distributions whose batch size parameters are equal."
        self.dists = args
        self.batch_size = data_batch_sizes[0]

        if(all([type(dist.data_dim) == int for dist in self.dists])):
            # if all distributions in the product are over vectors, we can concatenate them to make a higher dim vector
            self.data_dim = sum([dist.data_dim for dist in self.dists])
        else:
            # if the distributions are not all vectors, they cannot necessarily be concatenated.
            self.data_dim = None
    def __iter__(self):
        return self
    def __next__(self):
        if(self.data_dim is None):
            raise ValueError("Cannot concatenate non-vector distributions. Use .next_tuple() instead.")
        return np.concatenate(self.next_tuple(), axis=-1)
    def next_tuple(self):
        return (next(dist) for dist in self.dists)


class ImageDataset():
    def __init__(self, path, batch_size, im_size, channels=3):
        '''
        A distribution given by samples from a dataset of images. Each image is assumed to be square
        and normalized to [0, 1].

        Arguments:
            - path (string): path to a dataset of images
            - batch_size (int): number of images per batch
            - im_size (int): images will be square cropped and resized so their side length is im_size.
            - channels (int): number of channels in output images
        '''
        assert type(im_size) == int, "Image size must be an integer"
        assert type(batch_size) == int, "Batch size must be an integer"
        assert type(path) == str, "Path must be a string"
        assert type(channels) == int and channels in [1, 3], "Input must be a greyscale or RGB image having 1 or 3 channels"

        self.path = path
        self.batch_size = batch_size
        self.im_size = im_size
        self.data_dim = (3, im_size, im_size)
        self.channels = channels

        transforms = [
            torchvision.transforms.Resize(im_size),
            torchvision.transforms.CenterCrop(im_size)
        ]

        if(self.channels == 1):
            transforms.append(torchvision.transforms.Grayscale(num_output_channels=1))

        # greyscale transformation must preceed the ToTensor() or pytorch will give an error.
        transforms.append(torchvision.transforms.ToTensor())

        transform = torchvision.transforms.Compose(transforms)
        dataset = torchvision.datasets.ImageFolder(root = self.path, transform = transform)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size = self.batch_size, shuffle=True, num_workers=0)

        self.dataset = dataset
        self.dataloader = dataloader
        self.dataloader_iter = iter(dataloader)

    def __iter__(self):
        self.dataloader_iter = iter(self.dataloader)
        return self
    def __next__(self):
        return 2 * next(self.dataloader_iter)[0] - 1


