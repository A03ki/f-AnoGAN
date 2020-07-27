import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from fanogan.test_anomaly_detection import test_anomaly_detection

from model import Generator, Discriminator, Encoder
from tools import SimpleDataset, load_mnist


def main(opt):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    _, (x_test, y_test) = load_mnist("dataset",
                                     training_label=opt.training_label,
                                     split_rate=opt.split_rate)
    test_mnist = SimpleDataset(x_test, y_test,
                               transform=transforms.Compose(
                                   [transforms.ToPILImage(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.5], [0.5])])
                               )
    test_dataloader = DataLoader(test_mnist, batch_size=1, shuffle=False)

    generator = Generator(opt)
    discriminator = Discriminator(opt)
    encoder = Encoder(opt)

    test_anomaly_detection(opt, generator, discriminator, encoder,
                           test_dataloader, device)


"""
The code below is:
Copyright (c) 2018 Erik Linder-Norén
Licensed under MIT
(https://github.com/eriklindernoren/PyTorch-GAN/blob/master/LICENSE)
"""


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent_dim", type=int, default=100,
                        help="dimensionality of the latent space")
    parser.add_argument("--img_size", type=int, default=28,
                        help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=1,
                        help="number of image channels")
    parser.add_argument("--training_label", type=int, default=0,
                        help="label for normal images")
    parser.add_argument("--split_rate", type=float, default=0.8,
                        help="rate of split for normal training data")
    opt = parser.parse_args()

    main(opt)
