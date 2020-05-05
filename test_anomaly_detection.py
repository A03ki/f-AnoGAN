
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from model import Generator, Discriminator, Encoder
from tools import SimpleDataset, load_mnist


def test_anomaly_detection(opt, generator, discriminator, encoder,
                           dataloader, device, kappa=1.0):
    generator.load_state_dict(torch.load("results/generator"))
    discriminator.load_state_dict(torch.load("results/discriminator"))
    encoder.load_state_dict(torch.load("results/encoder"))

    generator.to(device).eval()
    discriminator.to(device).eval()
    encoder.to(device).eval()

    criterion = nn.MSELoss()

    with open("results/score.csv", "w") as f:
        f.write("label,img_distance,anomaly_score,z_distance\n")

    for i, (img, label) in enumerate(dataloader):

        real_img = img.to(device)

        real_z = encoder(real_img)
        fake_img = generator(real_z)
        fake_z = encoder(fake_img)

        real_feature = discriminator.forward_features(real_img)
        fake_feature = discriminator.forward_features(fake_img)

        # Scores for anomaly detection
        img_distance = criterion(fake_img, real_img)
        loss_feature = criterion(fake_feature, real_feature)
        anomaly_score = img_distance + kappa * loss_feature

        z_distance = criterion(fake_z, real_z)

        print(f"{label.item()}, {img_distance}, "
              f"{anomaly_score}, {z_distance}\n")

        with open("results/score.csv", "a") as f:
            f.write(f"{label.item()},{img_distance},"
                    f"{anomaly_score},{z_distance}\n")


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

    img_shape = (opt.channels, opt.img_size, opt.img_size)
    generator = Generator(img_shape, opt.latent_dim)
    discriminator = Discriminator(img_shape)
    encoder = Encoder(img_shape)

    test_anomaly_detection(opt, generator, discriminator, encoder,
                           test_dataloader, device)


if __name__ == "__main__":
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
