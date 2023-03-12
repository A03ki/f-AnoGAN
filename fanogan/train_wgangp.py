import os
import torch
import torch.autograd as autograd
from torchvision.utils import save_image


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
                 train_dataloader, device, lambda_gp=10, valid_dataloader=None):
    generator.to(device)
    discriminator.to(device)

    optimizer_G = torch.optim.Adam(generator.parameters(),
                                   lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(),
                                   lr=opt.lr, betas=(opt.b1, opt.b2))

    os.makedirs("results/train_images", exist_ok=True)
    os.makedirs("results/valid_images", exist_ok=True)

    for epoch in range(opt.n_epochs):
        # Train mode
        _run_model_on_epoch(opt, generator, discriminator, train_dataloader,
                            "results/train_images/", lambda_gp, epoch, device,
                            optimizer_D=optimizer_D, optimizer_G=optimizer_G,
                            train_mode=True)
        # Validation mode
        if valid_dataloader is not None:
            _run_model_on_epoch(opt, generator, discriminator, valid_dataloader,
                                "results/valid_images/", lambda_gp, epoch,device)

    torch.save(generator.state_dict(), "results/generator")
    torch.save(discriminator.state_dict(), "results/discriminator")


def _run_model_on_epoch(opt, generator, discriminator, dataloader,
                        output_dirpath, lambda_gp, current_epoch, device,
                        optimizer_D=None, optimizer_G=None, train_mode=False):
    generator.train(train_mode)
    discriminator.train(train_mode)

    if train_mode:
        mode_tag = "[Train]"
    else:
        mode_tag = "[Valid]"

    padding_epoch = len(str(opt.n_epochs))
    padding_i = len(str(len(dataloader)))
    current_batches_done = (((len(dataloader) - 1) // opt.n_critic + 1)
                            * opt.n_critic * current_epoch)

    for i, (imgs, _)in enumerate(dataloader):

        # Configure input
        real_imgs = imgs.to(device)

        # ---------------------
        #  Train Discriminator
        # ---------------------

        if optimizer_D is not None:
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

        if optimizer_D is not None:
            d_loss.backward()
            optimizer_D.step()

        if optimizer_G is not None:
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

            if optimizer_G is not None:
                g_loss.backward()
                optimizer_G.step()

            print(f"{mode_tag} "
                  f"[Epoch {current_epoch:{padding_epoch}}/{opt.n_epochs}] "
                  f"[Batch {i:{padding_i}}/{len(dataloader)}] "
                  f"[D loss: {d_loss.item():3f}] "
                  f"[G loss: {g_loss.item():3f}]")

            if current_batches_done % opt.sample_interval == 0:
                save_image(fake_imgs.data[:25],
                           os.path.join(output_dirpath,
                                        f"{current_batches_done:06}.png"),
                           nrow=5, normalize=True)

            current_batches_done += opt.n_critic
