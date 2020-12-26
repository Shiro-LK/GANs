# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 18:04:44 2020

@author: Shiro
"""
import torch
from torch import nn
from torch.nn import  functional as F
import numpy as np

## Architecture guidelines for stable Deep Convolutional GANs
# •Replace any pooling layers with strided convolutions (discriminator) and fractional-stridedconvolutions (generator)
# •Use batchnorm in both the generator and the discriminator.
# •Remove fully connected hidden layers for deeper architectures.
# •Use ReLU activation in generator for all layers except for the output, which uses Tanh.
# •Use LeakyReLU activation in the discriminator for all layers.


# conv2dTranspose output : Hout​=(Hin​−1)×stride[0]−2×padding[0]+dilation[0]×(kernel_size[0]−1)+output_padding[0]+1
# MNIST
# Hout1 = 0 - 0 + 2 + 0 + 1  = 3
# Hout2 = 2 + 0 + 3 + 1 = 6
# Hout3 = 10 + 0 + 2 + 1 = 13
# Hout4 = 24 + 0 + 3 + 0 + 1  = 28
class DCGAN_Generator(nn.Module):
    '''
    DCGAN Generator Class
    Values:
        z_dim: the dimension of the noise vector, a scalar
        im_chan: the number of channels in the images, fitted for the dataset used, a scalar
        hidden_dim: the inner dimension, a scalar
        kernel_sizes : the different kernel size of each layer
        strides : the different strides used for each layer
    '''
    def __init__(self, z_dim=100, im_chan=1, hidden_dim=32, kernel_sizes=[3, 4, 3, 4 ], strides=[2, 1, 2, 2]):
        super(DCGAN_Generator, self).__init__()
        self.z_dim = z_dim
        # Build the neural network
        self.conv1 = self.make_gen_block(z_dim, hidden_dim * 4, kernel_size=kernel_sizes[0], stride=strides[0])
        self.conv2 = self.make_gen_block(hidden_dim * 4, hidden_dim * 2, kernel_size=kernel_sizes[1], stride=strides[1])
        self.conv3 = self.make_gen_block(hidden_dim * 2, hidden_dim, kernel_size=kernel_sizes[2], stride=strides[2])
        self.conv4 = self.make_gen_block(hidden_dim, im_chan, kernel_size=kernel_sizes[3], stride=strides[3], final_layer=True)
        

    def make_gen_block(self, input_channels, output_channels, kernel_size=3, stride=2, final_layer=False):
        '''
        Function to return a sequence of operations corresponding to a generator block of DCGAN, 
        corresponding to a transposed convolution, a batchnorm (except for in the last layer), and an activation.
        Parameters:
            input_channels: how many channels the input feature representation has
            output_channels: how many channels the output feature representation should have
            kernel_size: the size of each convolutional filter, equivalent to (kernel_size, kernel_size)
            stride: the stride of the convolution
            final_layer: a boolean, true if it is the final layer and false otherwise 
                      (affects activation and batchnorm)
        '''

        # Build the neural block
        if not final_layer:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(),
            )
        else: # Final Layer, use tanh instead of ReLU and not Batch Normlization
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride),
                nn.Tanh(),
            )

    def unsqueeze_noise(self, noise):
        '''
        Function for completing a forward pass of the generator: Given a noise tensor, 
        returns a copy of that noise with width and height = 1 and channels = z_dim.
        Parameters:
            noise: a noise tensor with dimensions (n_samples, z_dim)
        '''
        return noise.view(len(noise), self.z_dim, 1, 1)

    def forward(self, noise):
        '''
        Forward pass of the generator: Given a noise tensor, returns generated images.
        Parameters:
            noise: a noise tensor with dimensions (n_samples, z_dim)
        '''
        x = self.unsqueeze_noise(noise)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return (x + 1)/2.0
    
    
class DCGAN_Discriminator(nn.Module):
    '''
    DCGAN Discriminator Class
    Values:
        im_chan: the number of channels in the images, fitted for the dataset used, a scalar
    hidden_dim: the inner dimension, a scalar
    '''
    def __init__(self, input_chan=1, hidden_dim=16, kernel_sizes=[4,4,4], strides=[2,2,2]):
        super(DCGAN_Discriminator, self).__init__()
        self.conv1 = self.make_disc_block(input_chan, hidden_dim, kernel_size=kernel_sizes[0], stride=strides[0])
        self.conv2 = self.make_disc_block(hidden_dim, hidden_dim * 2, kernel_size=kernel_sizes[1], stride=strides[1])
        self.conv3 = self.make_disc_block(hidden_dim * 2, 1,  kernel_size=kernel_sizes[2], stride=strides[2], final_layer=True)

    def make_disc_block(self, input_channels, output_channels, kernel_size=4, stride=2, final_layer=False):
        '''
        Function to return a sequence of operations corresponding to a discriminator block of DCGAN, 
        corresponding to a convolution, a batchnorm (except for in the last layer), and an activation.
        Parameters:
            input_channels: how many channels the input feature representation has
            output_channels: how many channels the output feature representation should have
            kernel_size: the size of each convolutional filter, equivalent to (kernel_size, kernel_size)
            stride: the stride of the convolution
            final_layer: a boolean, true if it is the final layer and false otherwise 
                      (affects activation and batchnorm)
        '''
        
        # Build the neural block
        if not final_layer:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride),
                nn.BatchNorm2d(output_channels),
                nn.LeakyReLU(0.2),
            )
        else: # Final Layer
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride),
            )

    def forward(self, image):
        '''
        Function for completing a forward pass of the discriminator: Given an image tensor, 
        returns a 1-dimension tensor representing fake/real.
        Parameters:
            image: a flattened image tensor with dimension (im_dim)
        '''
        x = self.conv1(image)
        #print(x.shape)
        x = self.conv2(x)
        #print(x.shape)
        x = self.conv3(x)
        #print(x.shape)
        return x.view(len(x), -1) 
    
    
if __name__ == "__main__":
    
    # MNIST
    gen = DCGAN_Generator(z_dim=100, im_chan=1, hidden_dim=32, kernel_sizes=[3, 4, 3, 4 ], strides=[2, 1, 2, 2])
    disc = DCGAN_Discriminator(input_chan=1, hidden_dim=16, kernel_sizes=[4,4,4], strides=[2,2,2])
    image = gen(torch.randn(10, 100, device="cpu"))
    
    disc(image)