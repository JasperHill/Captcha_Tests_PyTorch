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
import torchvision.transforms  as transforms

from optparse                  import OptionParser
from torch.autograd            import Function
from skimage                   import io
from torch.utils.data          import Dataset, DataLoader

######################################################################################
## basis rotation layer and function
##
## rotates a square mat_dim-dimensional matrix with input_channels input channels
## into output_channels output channels
######################################################################################

class BasisRotationOperation(Function):
    @staticmethod
    def forward(ctx, input, operator):
        batch_size = input.shape[0]
        input, operator = input.detach(), operator.detach()
        input, operator = input.numpy(), operator.numpy()
        
        output_channels, input_channels = operator.shape[0], input.shape[1]
        mat_dim = input.shape[-1]
        output = np.zeros([batch_size, output_channels, mat_dim, mat_dim])
        c = 0
        
        for sample in input:
            for i in range(output_channels):
                for j in range(input_channels):
                    O = operator[i][j]
                    temp = np.matmul(O, sample[j])
                    temp = np.matmul(temp, O.T)                
                    output[c][i] += temp
            c += 1

        ctx.save_for_backward(torch.from_numpy(input).to(torch.float), torch.from_numpy(operator).to(torch.float))
        return torch.as_tensor(output, dtype=torch.float)

    @staticmethod
    def backward(ctx, grad_output):
        print('basis rotation backward')
        grad_output = grad_output.detach().numpy()
        input, operator = ctx.saved_tensors
        input, operator = input.numpy(), operator.numpy()
        
        batch_size = input.shape[0]
        output_channels, input_channels, mat_dim = operator.shape[0], input.shape[1], input.shape[-1]
        
        grad_me    = np.zeros([batch_size, output_channels, input_channels, mat_dim, mat_dim])
        grad_input = np.zeros_like(input)
        c = 0
        
        for sample in input:
            for i in range(input_channels):
                print('sample no. {} | i={}'.format(c,i))
                for j in range(mat_dim):
                    for k in range(mat_dim):
                        for l in range(output_channels):
                            for m in range(mat_dim):
                                for n in range(mat_dim):
                                    ## each element of output is like O_mj*input_jk*OT_kn => O_mj*input_jk*O_nk
                                    grad_input[c][i][j][k] += operator[l][i][m][j]*operator[l][i][n][k]
                                    grad_me[c][l][i][j][k] += sample[i][k][m]*operator[l][i][n][m]


            c += 1
        print('complete')
        return torch.from_numpy(grad_input).to(torch.float), torch.from_numpy(grad_me).to(torch.float)


class BasisRotation(nn.Module):
    def __init__(self, input_shape, output_channels):
        super(BasisRotation, self).__init__()
        self.mat_dim = input_shape[-1]
        self.input_channels = input_shape[0]
        self.output_channels = output_channels

        ## initialize each square operator of the rank-4 tensor as an identity operator
        R = np.zeros([self.output_channels,self.input_channels,self.mat_dim,self.mat_dim])
        for i in range(self.output_channels):
            for j in range(self.input_channels):
                for  k in range(self.mat_dim):
                    R[i][j][k][k] = 1

        self.R = nn.Parameter(torch.tensor(R))
        
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
        input, operator, operator_t = input.detach(), operator.detach(), operator_t.detach()
        input, operator, operator_t = input.numpy(), operator.numpy(), operator_t.numpy()

        batch_size = input.shape[0]
        output_channels, input_channels = operator.shape[0], input.shape[1]
        M, N, Mp, Np = input.shape[-2], input.shape[-1], operator.shape[-2], operator_t.shape[-1]
        output = np.zeros([batch_size, output_channels, Mp, Np])
        c = 0
        
        for sample in input:
            for i in range(output_channels):
                for j in range(input_channels):
                    O = operator[i][j]
                    OT = operator_t[i][j]
                    temp = np.matmul(O, sample[j])
                    temp = np.matmul(temp, OT)                
                    output[c][i] += temp
            c += 1

        ctx.save_for_backward(torch.from_numpy(input).to(torch.float), torch.from_numpy(operator).to(torch.float), torch.from_numpy(operator_t).to(torch.float))
        return torch.as_tensor(output, dtype=torch.float)

    @staticmethod
    def backward(ctx, grad_output):
        print('projection backward')        
        grad_output = grad_output.detach().numpy()
        input, operator, operator_t = ctx.saved_tensors
        input, operator, operator_t = input.numpy(), operator.numpy(), operator_t.numpy()
        
        batch_size = input.shape[0]
        output_channels, input_channels = operator.shape[0], input.shape[1]
        M, N, Mp, Np = input.shape[-2], input.shape[-1], operator.shape[-2], operator_t.shape[-1]

        grad_input = np.zeros_like(input)
        grad_me    = np.zeros([batch_size, output_channels, input_channels, Mp, M])
        grad_me_t  = np.zeros([batch_size, output_channels, input_channels, N, Np])
        c = 0
        
        for sample in input:
            for i in range(input_channels):
                for j in range(M):
                    for k in range(N):
                        for l in range(output_channels):
                            for m in range(Mp):
                                for n in range(Np):
                                    grad_input[c][i][j][k]   += operator[l][i][m][j]*operator_t[l][i][k][n]
                                    grad_me[c][l][i][m][j]   += sample[i][j][k]*operator_t[l][i][k][n]
                                    grad_me_t[c][l][i][k][n] += operator[l][i][m][j]*sample[i][j][k]
            c += 1
        print('complete')
        return torch.from_numpy(grad_input).to(torch.float), torch.from_numpy(grad_me).to(torch.float), torch.from_numpy(grad_me_t).to(torch.float)


class Projection(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(Projection, self).__init__()
        self.input_shape = input_shape[1:]
        self.input_channels = input_shape[0]

        self.output_shape = output_shape[1:]
        self.output_channels = output_shape[0]

        ## initialize left- and right- hand operators as identity operators on the smallest dimension between input and output shapes
        P = np.zeros([self.output_channels, self.input_channels, self.output_shape[0], self.input_shape[0]])
        PT = np.zeros([self.output_channels, self.input_channels, self.input_shape[1], self.output_shape[1]])

        for i in range(self.output_channels):
            for j in range(self.input_channels):
                for k in range(min([self.input_shape[0], self.input_shape[1], self.output_shape[0], self.output_shape[1]])):
                    P[i][j][k][k] = 1
                    PT[i][j][k][k] = 1

        self.P = nn.Parameter(torch.tensor(P))
        self.PT = nn.Parameter(torch.tensor(PT))
        
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
        input, X = input.detach(), X.detach()
        input, X = input.numpy(), X.numpy()

        batch_size = input.shape[0]
        output_channels, input_channels = X.shape[0], input.shape[1]
        M, N = input.shape[-2], input.shape[-1]
        output = np.zeros([batch_size,output_channels, M])
        c = 0
        
        for sample in input:
            for i in range(output_channels):
                for j in range(input_channels):
                    output[c][i] += np.matmul(sample[j],X[i][j])
                    
            c += 1
        ctx.save_for_backward(torch.from_numpy(input).to(torch.float), torch.from_numpy(X).to(torch.float))
        return torch.as_tensor(output, dtype=torch.float)

    @staticmethod
    def backward(ctx, grad_output):
        print('vectorization backward')        
        grad_output = grad_output.detach().numpy()
        input, X = ctx.saved_tensors
        input, X = input.numpy(), X.numpy()

        batch_size = input.shape[0]
        output_channels, input_channels = X.shape[0], input.shape[1]
        M, N = input.shape[-2], input.shape[-1]

        grad_input = np.zeros_like(input)
        grad_ve = np.zeros([batch_size, output_channels, input_channels, N])
        c = 0

        for sample in input:
            for i in range(output_channels):
                for j in range(input_channels):
                    for k in range(M):
                        for l in range(N):
                            grad_ve[c][i][j][l] += sample[j][k][l]
                            grad_input[c][j][k][l] += X[i][j][l]
            c += 1

        print('complete')
        return torch.from_numpy(grad_input).to(torch.float), torch.from_numpy(grad_ve).to(torch.float)
        
class Vectorizer(nn.Module):
    def __init__(self, input_shape, output_channels):
        super(Vectorizer, self).__init__()
        self.output_channels = output_channels
        self.input_channels = input_shape[0]
        self.dim = input_shape[-1]
        self.X = nn.Parameter(torch.tensor(np.ones([self.output_channels, self.input_channels, self.dim])))

    def forward(self, input):
        return VectorizerFunction.apply(input, self.X)
