import configparser
import os
from utils import util
from data.PokemonDataset import PokemonDataset
from torchvision import transforms
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from models.cdcgan import Generator, Discriminator, weights_init
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import torch.nn.functional as F
import random
import time

config = configparser.ConfigParser()
config.read(os.path.join("cfg", "config.ini"))
config = config['PARAMETERS']

def main():
    load_pretrained = False
    if os.path.isfile(os.path.join(config['pretrained'] + '_netG.pt')):
        load_pretrained = True
        netD_path = os.path.join(config['pretrained'] + '_netD.pt')
        netG_path = os.path.join(config['pretrained'] + '_netG.pt')
        current_epoch = int(config['pretrained'].split(os.path.sep)[-1].split("_")[0]) + 1
        current_iter = int(config['pretrained'].split(os.path.sep)[-1].split("_")[1])
        print(current_epoch, current_iter)
        print("pretrained")
    else:
        current_epoch = 0

    dataset = PokemonDataset(dataroot=config['dataroot'],
                            transform=transforms.Compose([
                                transforms.Resize(int(config['image_size'])),
                                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0),
                                transforms.RandomRotation(10),
                                transforms.RandomHorizontalFlip(p=0.5),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                            ]),
                            config=config)
    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size= int(config['batch_size']),
                                            shuffle=True, num_workers= int(config['workers']))

 
    device = torch.device("cuda:0" if (torch.cuda.is_available() and int(config['ngpu']) > 0) else "cpu")

    # Create the generator
    netG = Generator(config).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (int(config['ngpu']) > 1):
        netG = nn.DataParallel(netG, list(range(int(config['ngpu']))))

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    if load_pretrained:
        netG.load_state_dict(torch.load(netG_path))
    else:
        netG.apply(weights_init)
    netG.train()
    # Print the model
    print(netG)

    # Create the discriminator
    netD = Discriminator(config).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (int(config['ngpu']) > 1):
        netD = nn.DataParallel(netD, list(range(int(config['ngpu']))))

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    # netD.apply(weights_init)
    if load_pretrained:
        netD.load_state_dict(torch.load(netD_path))

    netD.train()
    # Print the model
    print(netD)

    label_nc = int(config['label_nc'])
    image_size = int(config['image_size'])
    # Initialize BCELoss function
    criterion = nn.BCELoss()

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(64, int(config['nz']), 1, 1, device=device)
    fixed_onehot = util.get_random_labels(64, label_nc, image_size, p=0.5)[0].to(device)

    # Establish convention for real and fake labels during training
    real_label = 0.9    # GAN tricks #1: label smoothing
    fake_label = 0

    # Setup Adam optimizers for both G and D
    # optimizerD = optim.Adam(netD.parameters(), lr=float(config['lr']), betas=(float(config['beta1']), 0.999))
    optimizerD = optim.Adam(filter(lambda p: p.requires_grad, netD.parameters()), lr=float(config['netD_lr']), betas=(float(config['beta1']), 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=float(config['netG_lr']), betas=(float(config['beta1']), 0.999))

    # Training Loop
    num_epochs = int(config['num_epochs'])
    nz = int(config['nz'])
    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    frames = []
    iters = 0
    if load_pretrained:
        iters = current_iter

    print("Starting Training Loop...")
    start_time = time.time()
    # For each epoch
    for epoch in range(current_epoch, num_epochs):
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            real_cpu = data[0].to(device)
            onehot_label = data[1].to(device)
            c_fill = data[2].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, device=device)
            # label = torch.rand(b_size,).uniform_(0.7, 0.9).to(device)   # label smoothing for real labels

            # Forward pass real batch through D
            output = netD(real_cpu, c_fill).view(-1)

            # GAN tricks #2: occasionally flip labels
            # label = util.flip_label(label, p=0.2).to(device)  # flip ~20% of the labels
            
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)

            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            c_onehot, c_fill = util.get_random_labels(b_size, label_nc, image_size, p=0.5)
            c_onehot = c_onehot.to(device)
            c_fill = c_fill.to(device)

            # Generate fake image batch with G
            fake = netG(noise, c_onehot)
            label.fill_(fake_label)

            # Classify all fake batch with D
            output = netD(fake.detach(), c_fill).view(-1)

            # GAN tricks #2: occasionally flip labels
            # label = util.flip_label(label, p=0.2).to(device)  # flip ~20% of the labels

            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Add the gradients from the all-real and all-fake batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake, c_fill).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # Output training stats
            if i % 50 == 0:
                end_time = time.time()
                duration = end_time - start_time
                print(f"{duration:.2f}s, [{epoch}/{num_epochs}][{i}/{len(dataloader)}]\tLoss_D: {errD.item():.4f}\t  \
                Loss_G: {errG.item():.4f}\tD(x): {D_x:.4f}\tD(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f}")
                start_time = time.time()

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % int(config['save_freq']) == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = netG(fixed_noise, fixed_onehot).detach().cpu()
                grid = vutils.make_grid(fake, padding=2, normalize=True)
                ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
                im = Image.fromarray(ndarr)
                im.save(os.path.join("output", f"epoch{epoch}_iter{iters}.png"))
                frames.append(im)
                torch.save(netD.state_dict(), os.path.join("output", f"{epoch}_{iters}_netD.pt"))
                torch.save(netG.state_dict(), os.path.join("output", f"{epoch}_{iters}_netG.pt"))

            iters += 1

    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses,label="G")
    plt.plot(D_losses,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join("output", "loss_curve.png"))
    frames[0].save(os.path.join('output', 'animation.gif'), format='GIF', append_images=frames[1:], save_all=True, duration=500, loop=0)

if __name__ == '__main__':
    main()