import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision4ad.datasets import MVTecAD

from fanogan.save_compared_images import save_compared_images

from model import Generator, Encoder


def main(opt):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    pipeline = [transforms.Resize([opt.img_size]*2),
                transforms.RandomHorizontalFlip()]
    if opt.channels == 1:
        pipeline.append(transforms.Grayscale())
    pipeline.extend([transforms.ToTensor(),
                     transforms.Normalize([0.5]*opt.channels, [0.5]*opt.channels)])

    transform = transforms.Compose(pipeline)
    mvtec_ad = MVTecAD(".", opt.dataset_name, train=False, transform=transform,
                       download=True)
    test_dataloader = DataLoader(mvtec_ad, batch_size=opt.n_grid_lines,
                                 shuffle=False)

    generator = Generator(opt)
    encoder = Encoder(opt)

    save_compared_images(opt, generator, encoder, test_dataloader, device)


"""
The code below is:
Copyright (c) 2018 Erik Linder-Norén
Licensed under MIT
(https://github.com/eriklindernoren/PyTorch-GAN/blob/master/LICENSE)
"""


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_grid_lines", type=int, default=10,
                        help="number of grid lines in the saved image")
    parser.add_argument("dataset_name", type=str,
                        choices=MVTecAD.available_dataset_names,
                        help="name of MVTec Anomaly Detection Datasets")
    parser.add_argument("--force_download", "-f", action="store_true",
                        help="flag of force download")
    parser.add_argument("--latent_dim", type=int, default=100,
                        help="dimensionality of the latent space")
    parser.add_argument("--img_size", type=int, default=64,
                        help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=3,
                        help="number of image channels (If set to 1, convert image to grayscale)")
    parser.add_argument("--n_iters", type=int, default=None,
                        help="value of stopping iterations")
    opt = parser.parse_args()

    main(opt)
