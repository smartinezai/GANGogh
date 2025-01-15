
# GAN-powered impressionist art generation
<table>
<tr>
<td>
This project takes several impressoinist painters' work to train GANs that generate art based on their styles as well as providing a comprehensive performance comparison of some of the most used GAN models (DCGAN, ProGAN,
WGAN and LSGAN. We also tried to combine WGAN and ProGAN to a new model we called ProWGAN.</td>
</tr>
</table>


## Dataset
Using the wikiart Dataset, 10 impressionist painters were selected. You can access the exact data used as a zip file [here](https://drive.google.com/file/d/1nVjPtk6CgNlIoZ0fHwOLDAbdx0q0atJC/view?usp=drive_link).


## Quickstart guide
LSGAN.ipynb provies a plug-and-play demonstration that runs the LSGAN.py script for easier visualisation as well as training and visualising the results of any of the supported GAN models (by default DCGAN). To use a different GAN model, simply specify the "type" argument 
at the time of object creation (Available types are "DGCAN", "WGAN", "ProGAN", "ProWGAN").
### Basic idea behind GANs
Borrowing ideas from game theory, the main notion of GANs consists in taking 2 adversarial networks (a generator and a discriminator) that strive to reach opposite goals during training.
Doing what essentially amounts to a MinMax game, the discriminator is tasked with discerning whether a provided sample is a real painting or a fake image created by the generator.
As both models try to optimise the same objective function in different directions, the goal is to reach a state where the generator produces outputs that fool the discriminator.

![](https://raw.githubusercontent.com/smartinezai/GANGogh/daefc741255f13d229f76dab7f5fa85a04b61b69/simple_gan.png)

