import torch
from torchvision import datasets


class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels, transform=None):
        self.transform = transform
        self.data = data
        self.labels = labels

    def __getitem__(self, idx):
        data = self.data[idx]
        labels = self.labels[idx]

        if self.transform:
            data = self.transform(data)

        return data, labels

    def __len__(self):
        return len(self.data)


def load_mnist(path, training_label=1, split_rate=0.8, download=True):
    train = datasets.MNIST(path, train=True, download=download)
    test = datasets.MNIST(path, train=False, download=download)
    _x_train = train.data[train.targets == training_label]
    x_train, x_test_normal = _x_train.split((int(len(_x_train) * split_rate)),
                                            dim=0)
    _y_train = train.targets[train.targets == training_label]
    y_train, y_test_normal = _y_train.split((int(len(_y_train) * split_rate)),
                                            dim=0)
    x_test = torch.cat([x_test_normal,
                        train.data[train.targets != training_label],
                        test.data], dim=0)
    y_test = torch.cat([y_test_normal,
                        train.targets[train.targets != training_label],
                        test.targets], dim=0)
    return (x_train, y_train), (x_test, y_test)
