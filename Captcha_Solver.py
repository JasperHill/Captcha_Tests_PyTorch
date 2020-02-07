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


parser = OptionParser()
parser.add_option("--load_model", action='store_true', dest="opt")
options, args = parser.parse_args()
opt = options.opt

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
IMG_CHANNELS =   4

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
                for j in range(mat_dim):
                    for k in range(mat_dim):
                        for l in range(output_channels):
                            for m in range(mat_dim):
                                for n in range(mat_dim):
                                    ## each element of output is like O_mj*input_jk*OT_kn => O_mj*input_jk*O_nk
                                    grad_input[c][i][j][k] += operator[l][i][m][j]*operator[l][i][n][k]
                                    grad_me[c][l][i][j][k] += sample[i][k][m]*operator[l][i][n][m]


            c += 1
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

    def backward(ctx, grad_output):
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

        self.projection = Projection(input_shape=input_shape, output_shape=[N,D,D])
        self.basis_rotation = BasisRotation(input_shape=[N,D,D], output_channels=N)

        self.vectorizer = Vectorizer(input_shape=[N,D,D], output_channels=2)
        self.linear = nn.Linear(in_features=2*D, out_features=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):        
        x = self.projection(x)
        x = self.basis_rotation(x)
        x = self.vectorizer(x)
        
        x = x.view(-1, self.num_flat_features(x))
        x = self.linear(x)
        
        return self.sigmoid(x)

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1

        for s in size:
            num_features *= s

        return num_features


#############################################################################################################################
##  preprocess data and train the GAn
##  
#############################################################################################################################

"""
CaptchaSolver.py is an adversarial generative neural network inspired by the work of
Ye et al., Yet Another Captcha Solver: An Adversarial Generative Neural Network Based Approach

The configuration herein is much simpler and less robust, but it follows the same method:
A relatively small sample of captcha images are presented to a network containing a generator,
attempting to reproduce such captcha images from their corresponding labels and a discriminator,
attempting to discern authentic and synthetic images. The two work toward opposing goals, and
training ceases when the discriminator is unable to correctly classify a certain fraction of the
inputs.

A solver is then trained with synthetic captcha images from the generator. Finally, the solver
is refined via training with the authentic captchas.
"""

## note: one of the png files from this source is improperly named; mv 3bnfnd.png 3bfnd.png
sampledir = os.path.join('.','samples')
p         = pathlib.Path(sampledir)

jpg_count = len(list(p.glob('*/*.jpg')))
png_count = len(list(p.glob('*/*.png')))
NUM_OF_IMAGES = jpg_count + png_count

train_dir = os.path.join(sampledir, 'training_samples')
test_dir = os.path.join(sampledir, 'testing_samples')

print('jpg_count: {}, png_count: {}'.format(jpg_count, png_count))


#############################################################
## processing and mapping functions for the files
#############################################################

## label generation and vectorization ##

alphanumeric = string.digits + string.ascii_lowercase
D = len(alphanumeric)
N = 5 ## number of characters in each captcha title

def char_to_vec(char):
    vec = np.zeros(D, dtype=np.float32)
    match = re.search(char,alphanumeric)
    vec[match.span()[0]] = 1
    return vec

def string_to_mat_and_sparse_mat(string):
    N = len(string)
    mat = np.zeros([N,D], dtype=np.float32)
    sparse_mat = np.zeros([N,D,D], dtype=np.float32)

    d = 0
    for char in string:
        mat[d] = char_to_vec(char)
        sparse_mat[d] = np.tensordot(mat[d],mat[d],axes=0)
        d += 1

    return mat,sparse_mat

def NN_mat_to_string(nnmat):
    string = ''

    for i in range(N):
        idx = tf.argmax(nnmat[i])
        string += alphanumeric[idx]

    return string

## transform matrices from the dataset back to strings for visualization
def mat_to_string(mat):
     string = ''
     npmat = mat.numpy()

     for i in range(N):
         for j in range(D):
             if (npmat[i][j] == 1):
                 string += alphanumeric[j]
                 break

     return string

def generate_labels(filename):
    parts = re.split('\.',filename)
    string_label = parts[0]

    mat_label, sparse_label = string_to_mat_and_sparse_mat(string_label)        

    return string_label, mat_label, sparse_label


## create a dataset subclass
class CaptchaDataset(Dataset):
    def __init__(self, imgdir, transform=None):
        super(Dataset, self).__init__()
        self.imgdir = imgdir
        #self.fnames = list(pathlib.Path(imgdir).glob('*'))
        self.fnames = os.listdir(imgdir)
        self.transform = transform

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        if torch.is_tensor(idx): idx = idx.tolist()

        fname                                 = self.fnames[idx]
        imgpath                               = os.path.join(self.imgdir, fname)
        img                                   = io.imread(imgpath)
        string_label, mat_label, sparse_label = generate_labels(fname)

        if self.transform: img = self.transform(img)
        
        return {'images': img, 'string labels': string_label, 'mat labels': mat_label, 'sparse labels': sparse_label}

## auxiliary function to visualize the data
def show_batch(sample_batch,guesses,filename):
    plt.figure(figsize=(10,10))
    batch_size = len(sample_batch)
    str_labels = sample_batch['string label']
    imgs       = sample_batch['image']

    
    for n in range(batch_size):
        str_label = str_labels[n]
        img = imgs[n]
        if guesses is not None: guess = NN_mat_to_string(guesses[n])

        ax = plt.subplot(np.ceil(batch_size/2),2,n+1)
        plt.imshow(img)

        if guesses is not None: plt.title('guess: {}'.format(guess))
        else:                   plt.title(str(str_label))

        plt.axis('off')
        plt.savefig(filename+'.pdf')

        
    
#############################################################
##  prepare dataset
#############################################################

EPOCHS      = range(5)
BATCH_SIZE  =       10
NUM_WORKERS =        1

train_ds = CaptchaDataset(train_dir, transform=transforms.Compose([transforms.ToTensor()]))
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

test_ds = CaptchaDataset(test_dir, transform=transforms.Compose([transforms.ToTensor()]))
test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

## reference vectors to be used in loss calculations
ref_synth_guesses = torch.from_numpy(np.ones([BATCH_SIZE,1])).to(torch.float)
ref_auth_guesses = torch.from_numpy(np.zeros([BATCH_SIZE,1])).to(torch.float)

GEN_SAVE_PATH = "./gen_saves.pth"
DISC_SAVE_PATH = "./disc_saves.pth"

generator = Generator(input_shape=[N,D,D])
discriminator = Discriminator(input_shape=[IMG_CHANNELS,IMG_HEIGHT,IMG_WIDTH])

gen_optimizer = optim.Adam(generator.parameters())
disc_optimizer = optim.Adam(discriminator.parameters())

if opt:
    generator.load_state_dict(torch.load(GEN_SAVE_PATH))
    discriminator.load_state_dict(torch.load(DISC_SAVE_PATH))

torch.save(generator.state_dict(), GEN_SAVE_PATH)
torch.save(discriminator.state_dict(), DISC_SAVE_PATH)        

gen_train_loss_hist  = []
disc_train_loss_hist = []

gen_test_loss_hist   = []
disc_test_loss_hist  = []

## train the GAN
print('epoch',' | ','generator training loss',' | ','generator testing loss',' | ','discriminator training loss',' | ','discriminator testing loss')
for epoch in EPOCHS:
    gen_loss = 0.0
    disc_loss = 0.0

    print('beginning training step')
    i = 0
    for data in train_dl:
        print('i: ',i)
        auth_imgs, string_labels, mat_labels, sparse_labels = data['images'], data['string labels'], data['mat labels'], data['sparse labels']

        gen_optimizer.zero_grad()
        disc_optimizer.zero_grad()

        synth_imgs = generator(sparse_labels)
        synth_guesses = discriminator(synth_imgs)
        auth_guesses = discriminator(auth_imgs)

        generator_loss = F.binary_cross_entropy(synth_guesses, ref_auth_guesses, reduction='sum')
        discriminator_loss  = F.binary_cross_entropy(synth_guesses, ref_synth_guesses, reduction='sum')
        discriminator_loss += F.binary_cross_entropy(auth_guesses, ref_auth_guesses, reduction='sum')

        generator_loss.backward()
        discriminator_loss.backward()

        gen_optimizer.step()
        disc_optimizer.step()

        gen_loss += generator_loss.item()
        disc_loss += discriminator_loss.item()
        i += 1

    gen_train_loss_hist.append(gen_loss/len(train_ds))
    disc_train_loss_hist.append(disc_loss/len(train_ds))

    ## test the GAN at the end of each training epoch
    print('beginning testing step')
    with torch.no_grad():
        for i, data in enumerate(test_dl, 0):
            print('i: ',i)
            auth_imgs, string_labels, mat_labels, sparse_labels = data['image'], data['string label'], data['mat label'], data['sparse label']

            synth_imgs = generator(sparse_labels)
            synth_guesses = discriminator(synth_imgs)
            auth_guesses = discriminator(auth_imgs)

            generator_loss = F.binary_cross_entropy(synth_guesses, ref_auth_guesses, reduction='sum')
            discriminator_loss  = F.binary_cross_entropy(synth_guesses, ref_synth_guesses, reduction='sum')
            discriminator_loss += F.binary_cross_entropy(auth_guesses, ref_auth_guesses, reduction='sum')

            gen_loss += generator_loss.item()
            disc_loss += discriminator_loss.item()

        gen_test_loss_hist.append(gen_loss/len(train_ds))
        disc_test_loss_hist.append(disc_loss/len(train_ds))

    print('%(epoch)i' % {'epoch': epoch},
          ' | ',
          '%(gtrl)5f' % {'gtrl': gen_train_loss_hist[-1]},
          ' | ',
          '%(gtsl)5f' % {'gtsl': gen_test_loss_hist[-1]},
          ' | ',
          '%(dtrl)5f' % {'dtrl': disc_train_loss_hist[-1]},
          ' | ',
          '%(dtsl)5f' % {'dtsl': disc_test_loss_hist[-1]})


