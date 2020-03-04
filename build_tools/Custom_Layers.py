from __future__ import absolute_import, division

import re
import os
import time
import math
import pathlib
import string
import torch
import torch.nn                as nn
import torch.nn.functional     as F
import torch.optim             as optim
import numpy                   as np
import matplotlib.pyplot       as plt
import custom_layers_cpp

from optparse                  import OptionParser
from torch.autograd            import Function
from skimage                   import io
from torch.utils.data          import Dataset, DataLoader

## note: all python-bound c++ functions require tensors of dtype double

######################################################################################
## basis rotation layer and function
##
## rotates a square mat_dim-dimensional matrix with input_channels input channels
## into output_channels output channels
######################################################################################

class BasisRotationOperation(Function):
    @staticmethod
    def forward(ctx, input, operator):

        output = custom_layers_cpp.BasisRotation_forward(input.to(torch.double),
                                                         operator.to(torch.double))
        ctx.save_for_backward(input, operator)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, operator = ctx.saved_tensors
        grad_input, grad_me = custom_layers_cpp.BasisRotation_backward(input.to(torch.double),
                                                                       grad_output.to(torch.double),
                                                                       operator.to(torch.double))
        
        return grad_input, grad_me


class BasisRotation(nn.Module):
    def __init__(self, input_shape, output_channels):
        super(BasisRotation, self).__init__()
        self.mat_dim = input_shape[-1]
        self.input_channels = input_shape[-3]
        self.output_channels = output_channels

        ## initialize each square operator of the rank-4 tensor as an identity operator
        R = torch.rand([self.output_channels,self.input_channels,self.mat_dim,self.mat_dim])
        c = 1/(self.output_channels * self.input_channels * self.mat_dim)
        #c = 1
        
        #for i in range(self.output_channels):
            #for j in range(self.input_channels):
                #for  k in range(self.mat_dim):
                    #R[i][j][k][k] = c

        self.R = nn.Parameter(c*R)
        
    def forward(self, input):
        return BasisRotationOperation.apply(input, self.R)

                    
######################################################################################
## projection operator layer and function
##
## projects an MxN matrix with input_channels input channels
## onto an MpxNp matrix with output_channels output channels
######################################################################################

class ProjectionOperation(Function):
    @staticmethod
    def forward(ctx, input, operator, operator_t):
        output = custom_layers_cpp.Projection_forward(input.to(torch.double),
                                                      operator.to(torch.double),
                                                      operator_t.to(torch.double))
        ctx.save_for_backward(input, operator, operator_t)
        
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, operator, operator_t = ctx.saved_tensors
        grad_input, grad_me, grad_me_t = custom_layers_cpp.Projection_backward(input.to(torch.double),
                                                                               grad_output.to(torch.double),
                                                                               operator.to(torch.double),
                                                                               operator_t.to(torch.double))
        return grad_input, grad_me, grad_me_t


class Projection(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(Projection, self).__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape

        self.input_channels = input_shape[-3]
        self.output_channels = output_shape[-3]

        ## initialize left- and right- hand operators as identity operators on the smallest dimension between input and output shapes
        P = torch.rand([self.output_channels, self.input_channels, self.output_shape[-2], self.input_shape[-2]])
        PT = torch.rand([self.output_channels, self.input_channels, self.input_shape[-1], self.output_shape[-1]])
        #c = 1
        c = 1/(self.input_channels * max(self.output_shape[-2],self.output_shape[-1]))
        
        #for i in range(self.output_channels):
            #for j in range(self.input_channels):
                #for k in range(min([self.input_shape[-2], self.input_shape[-1], self.output_shape[-2], self.output_shape[-1]])):
                    #P[i][j][k][k] = c
                    #PT[i][j][k][k] = c
         
        self.P = nn.Parameter(c*P)
        self.PT = nn.Parameter(c*PT)
        
    def forward(self, input):
        return ProjectionOperation.apply(input, self.P, self.PT)


                    
######################################################################################
## vectorizer layer and function
##
## treats incoming matrices as operators, which act on the layer's vector kernel
## the output is a tensor of rank 1 less than the input
######################################################################################

class VectorizerFunction(Function):
    @staticmethod
    def forward(ctx, input, X):
        output = custom_layers_cpp.Vectorizer_forward(input.to(torch.double),
                                                      X.to(torch.double))
        ctx.save_for_backward(input, X)
        
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, X = ctx.saved_tensors
        grad_input, grad_ve = custom_layers_cpp.Vectorizer_backward(input.to(torch.double),
                                                                    grad_output.to(torch.double),
                                                                    X.to(torch.double))

        return grad_input, grad_ve
        
class Vectorizer(nn.Module):
    def __init__(self, input_shape, output_channels):
        super(Vectorizer, self).__init__()
        self.output_channels = output_channels
        self.input_channels = input_shape[-3]
        self.dim = input_shape[-1]

        # normalization coefficient to ensure transformed matrix elements initially lie within (0,1)
        #c = 1/(self.output_channels * self.input_channels * self.dim)
        c = 0.5
        self.X = nn.Parameter(1 - c*torch.rand([self.output_channels, self.input_channels, self.dim],
                                               dtype=torch.double))

    def forward(self, input):
        return VectorizerFunction.apply(input, self.X)
