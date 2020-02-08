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

from Captcha_GAN               import Generator, Discriminator
from optparse                  import OptionParser
from torch.autograd            import Function
from skimage                   import io
from torch.utils.data          import Dataset, DataLoader



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

EPOCHS      = range(1)
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

generator.load_state_dict(torch.load(GEN_SAVE_PATH))
discriminator.load_state_dict(torch.load(DISC_SAVE_PATH))

gen_optimizer = optim.Adam(generator.parameters())
disc_optimizer = optim.Adam(discriminator.parameters())

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


torch.save(generator.state_dict(), GEN_SAVE_PATH)
torch.save(discriminator.state_dict(), DISC_SAVE_PATH)        

