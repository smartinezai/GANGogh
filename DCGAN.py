#Based on the Tutorial-Code
##https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

import numpy as np
import torchvision.transforms
from tqdm import tqdm, trange # To visualize training progress
import matplotlib.pyplot as plt
import sys
#import pathlib #to get current path
#---Pytorch---
import torch
import torch.nn as nn # Base class for all neural network modules
import torch.nn.parallel # To parallalize the training process
import torch.backends.cudnn as cudnn # cuDNN is library for NN based on CUDA (To enable GPU training on Nvidia GPUs)
import torch.optim as optim # optimization algorithms (such as vanilla gradient descent, AdaGrad or Adam)
import torch.utils.data # For loading the data
import torchvision.datasets as dset # represents the set with a torch-compatible datatype
import torchvision.transforms as transforms # Image transformations (convolution etc.)
import torchvision.utils as vutils # Some utile functions
import torch.nn.functional as F
#-----Visualization of computation graph-----
#from graphviz import Digraph
#from torch.autograd import Variable
#from torchviz import make_dot
#----Own files----
import models as m

#------program-parameter----
# Root directory for dataset
#dataroot = "data"
dataroot = "data"
# Number of workers for dataloader
workers = 4
# Spatial size of training images. All images will be resized to this size.
image_size = 256
# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1
#-------hyperparameter-------
# Size of the mini-batch
batch_size = 64
# Size of z latent vector
nz = 100
# Number of training epochs
num_epochs = 100
double_epochs_through_round = True
# Learning rate
lr = 0.00005
# Beta1 hyperparam for Adam optimizer
beta1 = 0.
beta2 = 0.99
#regularization lambda of discriminator L2 regularization
l2_lambda = 0.01
#--ONLY FOR WGAN---
# How is the critic trained in proportion to the generator
n_critic = 5
weight_cliping_limit = 0.01
#----------------------------

class GAN:
    def __init__(self,type='DCGAN',load = False,path=""):
        self.type = type
        # Loading training data (all artists)
        self.dataset = dset.ImageFolder(root=dataroot,
                                   transform=transforms.Compose([
                                       transforms.Resize(image_size),
                                       transforms.CenterCrop(image_size),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                   ]))
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)
        #Specify device for GPU training (if available)
        self.device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

        if(load): #load a existing model if a path is given
            self.load(path)
            return
        #Create Generator
        if(type == 'DCGAN' or type == 'WGAN'):
            self.generator = m.generator(self.device,nz,ProGAN=False).to(self.device)
        elif(type == 'ProGAN' or type == 'ProWGAN'):
            self.generator = m.generator(self.device,nz, ProGAN=True).to(self.device)

        if (self.device.type == 'cuda') and (ngpu > 1):
            self.generator = nn.DataParallel(self.generator, list(range(ngpu)))

        #Create discriminator
        if(type =='DCGAN'):
            self.discriminator = m.discriminator(self.device,nz,ProGAN=False).to(self.device)
        elif(type == 'ProGAN'):
            self.discriminator = m.discriminator(self.device,nz, ProGAN=True).to(self.device)
        elif(type=='WGAN'):
            self.discriminator = m.critic(3*256*256,self.device).to(self.device)
        elif(type=='ProWGAN'):
            self.discriminator = m.critic(3*8*8,self.device).to(self.device)


        if (self.device.type == 'cuda') and (ngpu > 1):
            self.discriminator = nn.DataParallel(self.discriminator, list(range(ngpu)))

        # Initialize Losses
        #One can transform the loss of the discriminator and the generator in BCELoss ()
        self.criterion = nn.BCELoss()

        # Establish convention for real and fake labels during training
        self.real_label = 1.
        self.fake_label = 0.


    #On an IDE (like pycharm) it is important to set "Emulate terminal in output console"
    #that the nested progress bar can work correctly
    def train(self,_num_epochs=num_epochs): #base train method from pytorch expanded according to project needs
        # Setup Adam optimizers for both G and D (Improved gradient descent algorithm)
        self.optimizerD = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(beta1, beta2))
        self.optimizerG = optim.Adam(self.generator.parameters(), lr=lr, betas=(beta1, beta2))
        #To safe the progress
        self.G_losses = []
        self.D_losses = []
        # For each epoch
        for epoch in tqdm(range(_num_epochs),total=_num_epochs,file=sys.stdout,position=0,unit="epochs"):
            # For each batch in the dataloader
            # Update alpha of FadeIn Layer if it exists
            last_name, last_model = list(self.generator.main.named_children())[-1]
            if last_name == 'fade_in':
                last_model.update_alpha(1 / _num_epochs)
            #with tqdm(total=len(self.dataloader),file=sys.stdout,position=1,leave=False,unit="batches") as pbar2:
            for i, data in tqdm(enumerate(self.dataloader, 0),total=len(self.dataloader),file=sys.stdout,position=1,leave=False,unit="batches"):
                ############################
                # (1) Update Discriminator: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                ## Train with all-real batch [first part of the discriminator loss function: -1/m sum_{i=1}^m log(D(x_i)).]
                self.discriminator.zero_grad()
                # Format batch
                image_size = int(2**(self.discriminator.num_blocks+1))
                real_cpu = self.resizeTensor(data[0].to(self.device),(image_size,image_size))
                #print(real_cpu.shape)
                b_size = real_cpu.size(0)
                #create all-real labels, so we can use CEL for the first part of the discriminator loss
                label = torch.full((b_size,), self.real_label, dtype=torch.float, device=self.device)
                # Forward pass real batch through D
                #print(self.discriminator(real_cpu).shape)
                output = self.discriminator(real_cpu).view(-1) #view(-1) is to flatten 128x1x1 -> 128
                #print("batch size", real_cpu.size())
                # Calculate loss on all-real batch
                errD_real = self.criterion(output, label)
                #make_dot(errD_real).view()
                # Calculate gradients for D in backward pass
                errD_real.backward()
                D_x = output.mean().item()

                ## Train with all-fake batch [second part of the discriminator loss function: -1/m sum_{i=1}^m log(1 - D(G(z_i))).]
                # Generate batch of latent vectors
                noise = torch.randn(b_size, nz, 1, 1, device=self.device)
                # Generate fake image batch with G
                fake = self.generator(noise)
                #print("TEST",fake.shape)
                label.fill_(self.fake_label)
                #print("fake size",fake.size())
                # Classify all fake batch with D
                output = self.discriminator(fake.detach()).view(-1)
                # Calculate D's loss on the all-fake batch
                errD_fake = self.criterion(output, label)
                # Calculate the gradients for this batch, accumulated (summed) with previous gradients
                errD_fake.backward()
                D_G_z1 = output.mean().item()

                """
                #Calculate L2 regularization
                l2_reg = torch.tensor(0.,device=self.device)
                for param in self.discriminator.parameters():
                    l2_reg += torch.norm(param)
                """

                # Compute error of D as sum over the fake and the real batches
                #errD = errD_real + errD_fake + l2_lambda*l2_reg
                errD = errD_real + errD_fake

                # Update D
                self.optimizerD.step()

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                self.generator.zero_grad()
                label.fill_(self.real_label)  # fake labels are real for generator cost
                # Since we just updated D, perform another forward pass of all-fake batch through D
                output = self.discriminator(fake).view(-1)
                # Calculate G's loss based on this output
                errG = self.criterion(output, label)
                # Calculate gradients for G
                errG.backward()
                D_G_z2 = output.mean().item()
                # Update G
                self.optimizerG.step()

                #Update inner progress bar for the batches
                #pbar2.update()

                #Add to average batch loss
                self.G_losses.append(errG.item())
                self.D_losses.append(errD.item())
            
    
    def WGAN_train(self,_num_epochs=num_epochs):
        assert self.type == 'WGAN' or self.type == 'ProWGAN'
        # Setup Adam optimizers for both G and D (Improved gradient descent algorithm)
        self.optimizerD = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(beta1, beta2))
        self.optimizerG = optim.Adam(self.generator.parameters(), lr=lr, betas=(beta1, beta2))
        #To safe the progress
        self.G_losses = []
        self.D_losses = []
        # For each epoch
        for epoch in tqdm(range(_num_epochs),total=_num_epochs,file=sys.stdout,position=0,unit="epochs"):
            # For each batch in the dataloader
            for i, data in tqdm(enumerate(self.dataloader, 0),total=len(self.dataloader),file=sys.stdout,position=1,leave=False,unit="batches"):
                ############################
                # (1) Update Critic
                ###########################
                self.discriminator.zero_grad()
                #weight clipping
                # Clamp parameters to a range [-c, c], c=weight_cliping_limit
                for p in self.discriminator.parameters():
                    p.data.clamp_(-weight_cliping_limit, weight_cliping_limit)

                image_size = int(2 ** (self.discriminator.num_blocks + 1))
                real_cpu = self.resizeTensor(data[0].to(self.device), (image_size, image_size))
                #print(real_cpu.shape)
                b_size = real_cpu.size(0)
                noise = torch.randn(b_size, nz, 1, 1, device=self.device)
                # Generate fake image batch with G
                fake = self.generator(noise)
                loss_D = -torch.mean(self.discriminator(real_cpu)) + torch.mean(self.discriminator(fake))
                loss_D.backward()
                self.optimizerD.step()
                self.D_losses.append(loss_D.item())
                #Train the generator only every n_critic step
                if i % n_critic == 0:
                    ############################
                    # (2) Update Generator
                    ###########################
                    self.generator.zero_grad()
                    fake = self.generator(noise)
                    loss_G = -torch.mean(self.discriminator(fake))
                    loss_G.backward()
                    self.optimizerG.step()
                    self.G_losses.append(loss_G.item())

#These two training methods can be more cleanly consolidated into one. Ideally a main train method that according to the GAN type calls the correspondent train method
    def ProGAN_train(self):
        assert self.type == 'ProGAN'
        for round in range(5):
            if(double_epochs_through_round):
                self.train(num_epochs*2**round)
                self.plot_training_loss("Round " + str(round+1),num_epochs*2**round)
            else:
                self.train()
                self.plot_training_loss("Round " + str(round+1))

            self.plot_generated_images()
            if round != 4: #Dont grow again in the last round
                print("Grow networks...")
                self.generator.grow()
                self.discriminator.grow()
            #print(list(self.discriminator.main.named_children()))

    def ProWGAN_train(self):
        assert self.type == 'ProWGAN'
        for round in range(5):
            if(double_epochs_through_round):
                self.WGAN_train(num_epochs*2**round)
                self.plot_training_loss("Round " + str(round+1),num_epochs*2**round)
            else:
                self.WGAN_train()
                self.plot_training_loss("Round " + str(round+1))

            self.plot_generated_images()
            if round != 4: #Dont grow again in the last round
                print("Grow networks...")
                self.generator.grow()
                self.discriminator.replace_first_layer((2**(round+1)*8)**2*3)
            #print(list(self.discriminator.main.named_children()))

    def generate(self,z):
        #self.generator.eval()
        return self.generator(z)

    def resizeTensor(self,tensor,size): #resizes the images of an tensor without a record in calc-graph!
        with torch.no_grad():
            result = torchvision.transforms.Resize(size)(tensor)
        return result

    def plot_training_images(self):
        # Plot some training images
        batch = next(iter(self.dataloader))
        plt.figure(figsize=(8,8))
        plt.axis("off")
        plt.title("Training Images")
        plt.imshow(np.transpose(vutils.make_grid(batch[0].to(self.device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
        plt.show()

    def plot_generated_images(self):
        noise = torch.randn(64, nz, 1, 1,device=self.device)
        images = self.generate(noise)
        #print(images.size())
        plt.figure(figsize=(8,8))
        plt.axis("off")
        plt.title("Generated Images")
        plt.imshow(np.transpose(vutils.make_grid(images.to(self.device), padding=2, normalize=True).cpu(),(1,2,0)))
        plt.show()

    def plot_generated_image(self):
        noise = torch.randn(1, nz, 1, 1, device=self.device)
        image = self.generate(noise)[0]
        plt.axis("off")
        plt.title("Generated image")
        plt.imshow(np.transpose(vutils.make_grid(image,normalize=True).cpu().detach().numpy(),(1,2,0)))
        plt.show()

    def save(self,path="",saveDiscriminator=False):
        torch.save(self.generator, path+"generator_save")
        if saveDiscriminator:
            torch.save(self.discriminator, path+"discriminator_save")

    def load(self,path="",loadDiscriminator=False):
        self.generator = torch.load(path+"generator_save")
        #self.generator.eval()
        #self.generator.eval()
        if loadDiscriminator:
            self.discriminator = torch.load(path+"discriminator_save")
            self.discriminator.eval()

    def plot_training_loss(self,round_str="",_num_epochs=num_epochs):
        #print(self.G_losses)
        if len(self.G_losses) > 0:
            plt.figure(figsize=(10,5))
            plt.title("Generator and Discriminator Loss During Training "+round_str)
            if self.type == 'WGAN' or 'ProWGAN':
                plt.plot(range(0,len(self.D_losses)),self.D_losses, label="D", color="blue")
                plt.plot([n_critic*i for i in range(len(self.G_losses))],self.G_losses, label="G", color="red")
            else:
                plt.plot(self.G_losses, label="G", color="red")
                plt.plot(self.D_losses, label="D", color="blue")
            #max_y = max([max(self.D_losses),max(self.G_losses)])
            #for i in range(1,_num_epochs):# for epoche
            #    plt.vlines(len(self.dataloader)*i,0,max_y,colors="gray",linestyles="dotted",alpha=0.3)
            plt.xlabel("iterations")
            plt.ylabel("Loss")
            plt.legend()
            #plt.xticks(range(0, num_epochs))
            plt.show()
        else: raise Exception("The model has to be trained before plotting the losses")

if __name__ == '__main__':
    gan1 = GAN(type='ProWGAN')
    #gan1.WGAN_train()
    gan1.ProWGAN_train()
    gan1.save(saveDiscriminator=True)
    #gan1 = GAN(load=True)
    #gan1.plot_generated_image()
    #gan1.plot_training_loss()
