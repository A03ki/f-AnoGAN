import argparse
import os
import torch
import torch.autograd as autograd
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image

from model import Generator, Discriminator
from tools import SimpleDataset, load_mnist


"""
These codes are:
Copyright (c) 2018 Erik Linder-Norén
Licensed under MIT
(https://github.com/eriklindernoren/PyTorch-GAN/blob/master/LICENSE)
"""


def compute_gradient_penalty(D, real_samples, fake_samples, device):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.rand(*real_samples.shape[:2], 1, 1, device=device)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples)
    interpolates = autograd.Variable(interpolates, requires_grad=True)
    d_interpolates = D(interpolates)
    fake = torch.ones(*d_interpolates.shape, device=device)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(outputs=d_interpolates, inputs=interpolates,
                              grad_outputs=fake, create_graph=True,
                              retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.shape[0], -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def train_wgangp(opt, generator, discriminator,
                 dataloader, device, lambda_gp=10):
    generator.to(device)
    discriminator.to(device)

    optimizer_G = torch.optim.Adam(generator.parameters(),
                                   lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(),
                                   lr=opt.lr, betas=(opt.b1, opt.b2))

    os.makedirs("results/images", exist_ok=True)

    padding_epoch = len(str(opt.n_epochs))
    padding_i = len(str(len(dataloader)))

    batches_done = 0
    for epoch in range(opt.n_epochs):
        for i, (imgs, _)in enumerate(dataloader):

            # Configure input
            real_imgs = imgs.to(device)

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Sample noise as generator input
            z = torch.randn(imgs.shape[0], opt.latent_dim, device=device)

            # Generate a batch of images
            fake_imgs = generator(z)

            # Real images
            real_validity = discriminator(real_imgs)
            # Fake images
            fake_validity = discriminator(fake_imgs.detach())
            # Gradient penalty
            gradient_penalty = compute_gradient_penalty(discriminator,
                                                        real_imgs.data,
                                                        fake_imgs.data,
                                                        device)
            # Adversarial loss
            d_loss = (-torch.mean(real_validity) + torch.mean(fake_validity)
                      + lambda_gp * gradient_penalty)

            d_loss.backward()
            optimizer_D.step()

            optimizer_G.zero_grad()

            # Train the generator and output log every n_critic steps
            if i % opt.n_critic == 0:

                # -----------------
                #  Train Generator
                # -----------------

                # Generate a batch of images
                fake_imgs = generator(z)
                # Loss measures generator's ability to fool the discriminator
                # Train on fake images
                fake_validity = discriminator(fake_imgs)
                g_loss = -torch.mean(fake_validity)

                g_loss.backward()
                optimizer_G.step()

                print(f"[Epoch {epoch:{padding_epoch}}/{opt.n_epochs}] "
                      f"[Batch {i:{padding_i}}/{len(dataloader)}] "
                      f"[D loss: {d_loss.item():3f}] "
                      f"[G loss: {g_loss.item():3f}]")

                if batches_done % opt.sample_interval == 0:
                    save_image(fake_imgs.data[:25],
                               f"results/images/{batches_done:06}.png",
                               nrow=5, normalize=True)

                batches_done += opt.n_critic

    torch.save(generator.state_dict(), "results/generator")
    torch.save(discriminator.state_dict(), "results/discriminator")


def main(opt):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    (x_train, y_train), _ = load_mnist("dataset",
                                       training_label=opt.training_label,
                                       split_rate=opt.split_rate)
    train_mnist = SimpleDataset(x_train, y_train,
                                transform=transforms.Compose(
                                    [transforms.ToPILImage(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.5], [0.5])])
                                )
    train_dataloader = DataLoader(train_mnist, batch_size=opt.batch_size,
                                  shuffle=True)

    img_shape = (opt.channels, opt.img_size, opt.img_size)
    generator = Generator(img_shape, opt.latent_dim)
    discriminator = Discriminator(img_shape)

    train_wgangp(opt, generator, discriminator, train_dataloader, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=200,
                        help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002,
                        help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5,
                        help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999,
                        help="adam: decay of first order momentum of gradient")
    parser.add_argument("--latent_dim", type=int, default=100,
                        help="dimensionality of the latent space")
    parser.add_argument("--img_size", type=int, default=28,
                        help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=1,
                        help="number of image channels")
    parser.add_argument("--n_critic", type=int, default=5,
                        help="number of training steps for "
                             "discriminator per iter")
    parser.add_argument("--sample_interval", type=int, default=400,
                        help="interval betwen image samples")
    parser.add_argument("--training_label", type=int, default=0,
                        help="label for normal images")
    parser.add_argument("--split_rate", type=float, default=0.8,
                        help="rate of split for normal training data")
    opt = parser.parse_args()

    main(opt)
