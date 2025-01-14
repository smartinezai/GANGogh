import torch
import torch.nn as nn # Base class for all neural network modules
import copy
import torch.nn.functional as F

#---Model changes---
#Using minibatch standart deviation concat in the discriminator before last block
discriminator_batchstdconcat = False
#Using spectral norm after each convolutional layer
discriminator_spectral_norm = False
generator_spectral_norm = False
#normalization technique after each conv layer
# ['batch','pixel']
discriminator_norm = 'batch'
generator_norm = 'batch'
#initialization method ['kaiming','xavier','normal']
weight_init_method = 'normal'
#use equalized conv (also kown as equalized learning rate)
equalized_convolution = False
#---Normalize the latent variable with PixelNorm---
norm_latent = False
#---ProGAN smooth growing---
smooth_growing = True


def weights_init(m): #init the weights of the unmodified Conv and BatchNorm layers
    if isinstance(m,nn.Conv2d) or isinstance(m,nn.ConvTranspose2d):
        if(weight_init_method == 'normal'):
            nn.init.normal_(m.weight, 0.0,0.02)
        elif(weight_init_method == 'kaiming'):
            nn.init.kaiming_normal_(m.weight, a=nn.init.calculate_gain('conv2d'))
        elif(weight_init_method == 'xavier'):
            nn.init.xavier_normal_(m.weight)
    elif isinstance(m,nn.BatchNorm2d): #scale and bias are learnable
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


#-------- Custom layers-----------
# https://github.com/github-pengge/PyTorch-progressive_growing_of_gans/blob/master/models/base_model.py
#Calculate the batch-standart deviation of specific dimensions
class BatchStdConcat(nn.Module):
    def __init__(self, averaging='all'):
        super(BatchStdConcat, self).__init__()
        self.averaging = averaging.lower()
        if 'group' in self.averaging:
            self.n = int(self.averaging[5:])
        else:
            assert self.averaging in ['all', 'flat', 'spatial', 'none', 'gpool'], 'Invalid averaging mode'%self.averaging
        self.adjusted_std = lambda x, **kwargs: torch.sqrt(torch.mean((x - torch.mean(x, **kwargs)) ** 2, **kwargs) + 1e-8)

    def forward(self, x):
        #print("input shape",x.shape)
        shape = list(x.size())
        target_shape = copy.deepcopy(shape)
        vals = self.adjusted_std(x, dim=0, keepdim=True) #batch mean
        if self.averaging == 'all':
            target_shape[1] = 1
            vals = torch.mean(vals, dim=1, keepdim=True)
        elif self.averaging == 'spatial': #over sptial dimensions 2,3
            if len(shape) == 4:
                vals = torch.mean(torch.mean(vals, 2, keepdim=True), 3, keepdim=True)
        elif self.averaging == 'none':
            target_shape = [target_shape[0]] + [s for s in target_shape[1:]]
        elif self.averaging == 'gpool':
            if len(shape) == 4:
                vals = torch.mean(torch.mean(torch.mean(x, 2, keepdim=True), 3, keepdim=True), 0, keepdim=True)
        elif self.averaging == 'flat':
            target_shape[1] = 1
            vals = torch.FloatTensor([self.adjusted_std(x)])
        else:                                                           # self.averaging == 'group'
            target_shape[1] = self.n
            vals = vals.view(self.n, self.shape[1]/self.n, self.shape[2], self.shape[3])
            vals = torch.mean(vals, axis=0, keepdim=True).view(1, self.n, 1, 1)
        vals = vals.expand(*target_shape)
        return torch.cat([x, vals], 1) #concatenate the result feature map to the other feature maps

    def __repr__(self): #print object
        return self.__class__.__name__ + '(averaging = %s)' % (self.averaging)

#normalize each pixel in the activation maps to unit length
class PixelNorm(nn.Module):
    def __init__(self,epsilon = 1e-8):
        super(PixelNorm, self).__init__()
        self.epsilon = epsilon

    #normalize each pixel with the L2-norm of the corresponding channel
    def forward(self, input):
        return input / torch.sqrt((torch.mean(input ** 2, dim=1, keepdim=True) + self.epsilon))


class Equalized_ConvTranspose2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride,padding,bias):
        super(Equalized_ConvTranspose2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weight = nn.Parameter(torch.Tensor(self.in_channels, self.out_channels, self.kernel_size, self.kernel_size))
        self.scale = (2 / self.in_channels) ** 0.5
        nn.init.normal_(self.weight, 0., 1.)
        if bias:
            self.bias = nn.Parameter(torch.tensor(out_channels))
            nn.init.zeros_(self.bias)
        else: self.bias = None

    def forward(self, input):
        return F.conv_transpose2d(input, self.weight * self.scale, self.bias, self.stride, self.padding)

class Equalized_Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias):
        super(Equalized_Conv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weight = nn.Parameter(torch.Tensor(self.out_channels,self.in_channels,self.kernel_size,self.kernel_size))
        self.scale = (2 / (self.in_channels * self.kernel_size**2))**0.5
        nn.init.normal_(self.weight, 0., 1.)
        if bias:
            self.bias = nn.Parameter(torch.tensor(out_channels))
            nn.init.zeros_(self.bias)
        else: self.bias = None

    def forward(self, input):
        return F.conv2d(input,self.weight*self.scale,self.bias,self.stride,self.padding)

class FadeIn_Concat(nn.Module):
    # downsample = True means use average pooling for the previous layer for the discriminator
    def __init__(self,previous_model,next_model,downsample):
        super(FadeIn_Concat,self).__init__()
        self.alpha = 0.
        self.previous_model = previous_model
        self.next_model = next_model
        self.downsample = downsample

    def update_alpha(self,delta):
        self.alpha += delta
        self.alpha = max(0.,min(self.alpha,1.)) #Alpha in [0,1]

    #combine previous result (up or downsampled) with growed layer reasult as a weighted sum
    #weight alpha grows linear during the epochs (not learnable!)
    def forward(self, input):
        if self.downsample: #downsample input for discriminator
            prev_input = F.avg_pool2d(input,kernel_size=4,stride=2,padding=1)
            result = torch.add(self.previous_model(prev_input).mul(1. - self.alpha),self.next_model(input).mul(self.alpha))
        else: #Upsample prev_result for generator
            prev_result = F.interpolate(self.previous_model(input),scale_factor=2)
            result = torch.add(prev_result.mul(1. - self.alpha),self.next_model(input).mul(self.alpha))
        return result

#-------Functions to choose the corresponding method based on the hyperparameter-------

def Norm(method_str,num_channels):
    if method_str == 'batch':
        return nn.BatchNorm2d(num_channels)
    elif method_str == 'pixel':
        return PixelNorm()

def ConvTranspose2d(use_spectral,in_channels,out_channels,kernel_size,stride,padding,bias):
    conv = None
    if equalized_convolution:
        conv = Equalized_ConvTranspose2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=stride,padding=padding,bias=bias)
    else:
        conv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
    if use_spectral:
        conv = torch.nn.utils.spectral_norm(conv)
    return conv



def Conv2d(use_spectral,in_channels,out_channels,kernel_size,stride,padding,bias):
    conv = None
    if equalized_convolution:
        conv = Equalized_Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=stride,padding=padding,bias=bias)
    else:
        conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
    if use_spectral:
        conv = torch.nn.utils.spectral_norm(conv)
    return conv

#------generator and discriminator blocks and classes------

class generator(nn.Module): # inherit from nn.Module
    def __init__(self,device,nz,ProGAN=True):
        super(generator, self).__init__() # calls construcctor of superclass nn.Module
        self.isProGAN = ProGAN
        self.device = device
        self.nz = nz
        self.main = nn.Sequential()
        if ProGAN: #start with 4x4 architecture
            self.num_blocks = 2
            self.main.add_module('first_block',self.first_block())
            self.main.add_module('last_block',self.last_block(256))
        else: #start with fully DCGAN architecture
            self.num_blocks = 7
            self.main.add_module('first_block',self.first_block())
            self.main.add_module('intermediate_block_256_128',self.intermediate_block(256,128))
            self.main.add_module('intermediate_block_128_64', self.intermediate_block(128, 64))
            self.main.add_module('intermediate_block_64_32', self.intermediate_block(64, 32))
            self.main.add_module('intermediate_block_32_16', self.intermediate_block(32, 16))
            self.main.add_module('intermediate_block_16_8', self.intermediate_block(16, 8))
            self.main.add_module('last_block', self.last_block(8))

        self.main.apply(weights_init)

    def first_block(self):
        layers = []
        if norm_latent:
            layers += [PixelNorm()]
        layers += [
            ConvTranspose2d(generator_spectral_norm,in_channels=self.nz, out_channels=256, kernel_size=4, stride=1, padding=0, bias=False),
            Norm(generator_norm,256),
            nn.ReLU(True)
                   ]
        return nn.Sequential(*layers)

    def intermediate_block(self,in_channels,out_channels):
        layers = [
            ConvTranspose2d(generator_spectral_norm, in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2,padding=1, bias=False),
            Norm(generator_norm, out_channels),
            nn.ReLU(True)
        ]
        return nn.Sequential(*layers)

    def last_block(self,in_channels): #toRGB
        layers = [
            ConvTranspose2d(generator_spectral_norm, in_channels=in_channels, out_channels=3, kernel_size=4, stride=2, padding=1,bias=False),
            nn.Tanh()
        ]
        return nn.Sequential(*layers)

    def grow(self):
        assert self.isProGAN
        assert self.num_blocks < 7
        last_name, last_model = list(self.main.named_children())[-1]
        if last_name == 'fade_in':
            previous_model = last_model.next_model
        else:
            previous_model = self.main
        #--copy old blocks except last block--
        next_model = nn.Sequential()
        named_children = list(previous_model.named_children())
        for name,module in named_children:
            if not name=='last_block':
                next_model.add_module(name,module)
                next_model[-1].load_state_dict(module.state_dict()) #copy pretrained params
        #Add one additional intermediate layer and the coresponding last layer
        #print("Growing generator from "+str(self.num_blocks)+" to "+str(self.num_blocks+1)+" blocks.")
        in_channels_new = int(2**(10-self.num_blocks))
        out_channels_new = int(in_channels_new/2)
        next_model.add_module('intermediate_block_'+str(in_channels_new)+'_'+str(out_channels_new),self.intermediate_block(in_channels_new,out_channels_new))
        next_model.add_module('last_block',self.last_block(out_channels_new))
        if(smooth_growing):
            #combine both models to the new model
            new_model = nn.Sequential()
            fade_in_block = FadeIn_Concat(previous_model,next_model,downsample=False)
            new_model.add_module('fade_in',fade_in_block)
        else:
            new_model = next_model
        self.num_blocks += 1
        new_model.to(self.device)
        self.main = new_model

    def forward(self, input):
        return self.main(input)


class discriminator(nn.Module):
    def __init__(self,device,nz,ProGAN=True):
        super(discriminator, self).__init__()
        self.nz = nz
        self.device = device
        self.isProGAN = ProGAN
        self.main = nn.Sequential()
        if ProGAN:
            self.num_blocks = 2
            self.main.add_module('first_block', self.first_block())
            self.main.add_module('last_block', self.last_block(8,batchStdConcat=discriminator_batchstdconcat)) #kernel_size has to ba as big as the final resultion
        else:
            self.num_blocks = 7
            self.main.add_module('first_block',self.first_block())
            self.main.add_module('intermediate_block_8_16',self.intermediate_block(8,16))
            self.main.add_module('intermediate_block_16_32', self.intermediate_block(16, 32))
            self.main.add_module('intermediate_block_32_64', self.intermediate_block(32, 64))
            self.main.add_module('intermediate_block_64_128', self.intermediate_block(64, 128))
            self.main.add_module('intermediate_block_128_256', self.intermediate_block(128, 256))
            self.main.add_module('last_block',self.last_block(256,batchStdConcat=discriminator_batchstdconcat))

        self.main.apply(weights_init)

    def grow(self):
        assert self.isProGAN
        assert self.num_blocks < 7
        last_name, last_model = list(self.main.named_children())[-1]
        if last_name == 'fade_in':
            previous_model = last_model.next_model
        else:
            previous_model = self.main
        #--copy old blocks except last block--
        next_model = nn.Sequential()
        named_children = list(previous_model.named_children())
        for name,module in named_children:
            if not name=='last_block':
                next_model.add_module(name,module)
                next_model[-1].load_state_dict(module.state_dict()) #copy pretrained params
        #Add one additional intermediate layer and the coresponding last layer
        #print("Growing discriminator from "+str(self.num_blocks)+" to "+str(self.num_blocks+1)+" blocks.")

        in_channels_new = int(2**(self.num_blocks+1)) #= current_image_size input
        out_channels_new = int(in_channels_new*2)
        next_model.add_module('intermediate_block_'+str(in_channels_new)+'_'+str(out_channels_new),self.intermediate_block(in_channels_new,out_channels_new))
        next_model.add_module('last_block',self.last_block(out_channels_new,batchStdConcat=discriminator_batchstdconcat))
        if(smooth_growing):
            # combine both models to the new model
            new_model = nn.Sequential()
            fade_in_block = FadeIn_Concat(previous_model, next_model, downsample=True)
            new_model.add_module('fade_in', fade_in_block)
        else:
            new_model = next_model
        self.num_blocks += 1
        new_model.to(self.device)
        self.main = new_model

    def forward(self, input):
        return self.main(input)

    def first_block(self):
        layers = [
            Conv2d(discriminator_spectral_norm, in_channels=3, out_channels=8, kernel_size=4, stride=2, padding=1,bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        return nn.Sequential(*layers)

    def intermediate_block(self,in_channels,out_channels,batchStdConcat = False):
        layers = []
        if batchStdConcat:
            layers += [
                BatchStdConcat(),
                Conv2d(discriminator_spectral_norm, in_channels=in_channels+1, out_channels=out_channels, kernel_size=4,stride=2, padding=1, bias=False)
                       ]
        else:
            layers += [
            Conv2d(discriminator_spectral_norm, in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1,bias=False),
            ]
        layers += [
            Norm(discriminator_norm, out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        return nn.Sequential(*layers)

    def last_block(self,in_channels,batchStdConcat = False): #fromRGB
        layers = []
        if(batchStdConcat):
            layers += [
                BatchStdConcat(),
                Conv2d(discriminator_spectral_norm, in_channels=in_channels+1, out_channels=1, kernel_size=4, stride=1, padding=0,bias=False)
            ]
        else:
            layers += [
            Conv2d(discriminator_spectral_norm, in_channels=in_channels, out_channels=1, kernel_size=4, stride=1, padding=0,bias=False)
            ]
        layers += [
            nn.Sigmoid()
        ]
        return nn.Sequential(*layers)

#------- Wasserstein related things-------
class critic(nn.Module):
    def __init__(self,num_features,device):
        self.device = device
        self.num_blocks = 2
        super(critic, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(num_features,512), #3*256*256
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(512,256),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(256,1)
        )
        self.main.to(self.device)

    def forward(self,input):
        flattend = input.view(input.shape[0],-1) #spatial dimensions flattend
        return self.main(flattend)

    def replace_first_layer(self,num_features):
        new_model = nn.Sequential()
        new_model.add_module("Linear",nn.Linear(num_features,512))
        i = 0
        for name, module in self.main.named_children():
            if(i!=0): #
                new_model.add_module(name, module)
                new_model[-1].load_state_dict(module.state_dict())  # copy pretrained params
            i += 1
        self.num_blocks += 1
        self.main = new_model
        self.main.to(self.device)

"""
#generator 
        lst = [ # Init the archticture of the generator
            # input is Z, going into a transposed convolution
            #ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1,
            #bias=True, dilation=1, padding_mode='zeros', device=None, dtype=None)
            # nz x 1 x 1 -> 256 x 4 x 4
            ConvTranspose2d(generator_spectral_norm,in_channels=nz, out_channels=256, kernel_size=4, stride=1, padding=0, bias=False),
            Norm(generator_norm,256),
            nn.ReLU(True),
            # 256 x 4 x 4 -> 128 x 8 x 8
            ConvTranspose2d(generator_spectral_norm,in_channels = 256, out_channels = 128, kernel_size=4, stride=2, padding=1, bias=False),
            Norm(generator_norm,128),
            nn.ReLU(True),
            # 128 x 8 x 8 -> 64 x 16 x 16
            ConvTranspose2d(generator_spectral_norm,in_channels = 128, out_channels = 64, kernel_size=4, stride=2, padding=1, bias=False),
            Norm(generator_norm,64),
            nn.ReLU(True),
            # 64 x 16 x 16 -> 32 x 32 x 32
            ConvTranspose2d(generator_spectral_norm,in_channels = 64, out_channels = 32, kernel_size=4, stride=2, padding=1, bias=False),
            Norm(generator_norm,32),
            nn.ReLU(True),
            # 32 x 32 x 32 -> 16 x 64 x 64
            ConvTranspose2d(generator_spectral_norm,in_channels = 32, out_channels = 16, kernel_size=4, stride=2, padding=1, bias=False),
            Norm(generator_norm,16),
            nn.ReLU(True),
            # 16 x 64 x 64 -> 8 x 128 x 128
            ConvTranspose2d(generator_spectral_norm,in_channels = 16, out_channels = 8, kernel_size=4, stride=2, padding=1, bias=False),
            Norm(generator_norm,8),
            nn.ReLU(True),
            # 8 x 128 x 128 -> 3 x 256 x 256
            ConvTranspose2d(generator_spectral_norm,in_channels = 8, out_channels = 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        ]
        
#discriminator
 lst = [
            #torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
            #bias=True, padding_mode='zeros', device=None, dtype=None)
            # 3 x 256 x 256 -> 8 x 128 x 128
            Conv2d(discriminator_spectral_norm,in_channels=3, out_channels=8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 8 x 128 x 128 -> 16 x 64 x 64
            Conv2d(discriminator_spectral_norm,in_channels=8, out_channels=16, kernel_size=4, stride=2, padding=1, bias=False),
            Norm(discriminator_norm,16),
            nn.LeakyReLU(0.2, inplace=True),
            # 16 x 64 x 64 -> 32 x 32 x 32
            Conv2d(discriminator_spectral_norm,in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=1, bias=False),
            Norm(discriminator_norm,32),
            nn.LeakyReLU(0.2, inplace=True),
            # 32 x 32 x 32 -> 64 x 16 x 16
            Conv2d(discriminator_spectral_norm,in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False),
            Norm(discriminator_norm,64),
            nn.LeakyReLU(0.2, inplace=True),
            # 64 x 16 x 16 -> 128 x 8 x 8
            Conv2d(discriminator_spectral_norm,in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False),
            Norm(discriminator_norm,128),
            nn.LeakyReLU(0.2, inplace=True),
            # 128 x 8 x 8 -> 256 x 4 x 4
            Conv2d(discriminator_spectral_norm,in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False),
            Norm(discriminator_norm,256),
            nn.LeakyReLU(0.2, inplace=True),
            # 256 x 4 x 4 -> 1 x 1 x 1
            Conv2d(discriminator_spectral_norm,in_channels=256, out_channels=1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        ]
"""