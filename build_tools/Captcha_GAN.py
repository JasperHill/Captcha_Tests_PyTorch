from __future__ import absolute_import, division

import re
import os
import time
import pathlib
import string
import torch
import torch.nn                as nn
import torch.nn.functional     as F
import torch.optim             as optim
import numpy                   as np
import matplotlib.pyplot       as plt

from Custom_Layers                    import BasisRotation, Projection, Vectorizer
from optparse                         import OptionParser
from torch.autograd                   import Function
from skimage                          import io
from torch.utils.data                 import Dataset, DataLoader


parser = OptionParser()
parser.add_option("--save_model", action='store_true', dest="save")
options, args = parser.parse_args()
save = options.save

"""
Create the GAN
"""

## input and output specs
## alphanumeric contains all allowed captcha characters
## D is the full length of each vectorized captcha string
## N is the number of characters in each captcha string

alphanumeric = string.digits + string.ascii_lowercase
D            = len(alphanumeric)
N            = 5

IMG_HEIGHT   =  50
IMG_WIDTH    = 200
IMG_CHANNELS =   3

######################################################################################
## create the discriminator and generator
######################################################################################

## the generator always produces 4-channel PIL images
## no need to specify output shape
class Generator(nn.Module):
    def __init__(self, input_shape):
        super(Generator, self).__init__()
        self.basis_rotation = BasisRotation(input_shape=input_shape,
                                            output_channels=input_shape[0])
        
        self.projection = Projection(input_shape=input_shape,
                                     output_shape=[IMG_CHANNELS,IMG_HEIGHT,IMG_WIDTH])


    def forward(self, x):
        x = self.basis_rotation(x)
        x = self.projection(x)

        return x

class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        self.projection = Projection(input_shape=input_shape,
                                     output_shape=[N,D,D])
        
        self.basis_rotation = BasisRotation(input_shape=[N,D,D],
                                            output_channels=N)

        self.vectorizer = Vectorizer(input_shape=[N,D,D], output_channels=2)
        self.linear = nn.Linear(in_features=2*D, out_features=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):        
        x = self.projection(x)
        x = self.basis_rotation(x)
        x = self.vectorizer(x)
        
        x = x.view(-1, self.num_flat_features(x)).to(torch.float)
        x = self.linear(x)
        
        return self.sigmoid(x)

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1

        for s in size:
            num_features *= s

        return num_features


GEN_SAVE_PATH = "./gen_saves.pth"
DISC_SAVE_PATH = "./disc_saves.pth"

generator = Generator(input_shape=[N,D,D])
discriminator = Discriminator(input_shape=[IMG_CHANNELS,IMG_HEIGHT,IMG_WIDTH])

gen_optimizer = optim.Adam(generator.parameters())
disc_optimizer = optim.Adam(discriminator.parameters())

if save:
    torch.save(generator.state_dict(), GEN_SAVE_PATH)
    torch.save(discriminator.state_dict(), DISC_SAVE_PATH)        

