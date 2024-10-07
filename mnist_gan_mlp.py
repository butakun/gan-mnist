import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision 
from torchvision.transforms import v2

class Discriminator(nn.Module):

    def __init__(self, img_size=28, hidden_size=64):
        super().__init__()

        self.model = nn.Sequential(
                nn.Flatten(),
                nn.Linear(img_size * img_size, hidden_size),
                nn.LeakyReLU(),
                nn.Dropout(),
                nn.Linear(hidden_size, hidden_size),
                nn.LeakyReLU(),
                nn.Dropout(),
                nn.Linear(hidden_size, 1),
                nn.Sigmoid()
                )

    def forward(self, x):
        return self.model(x)


class Generator(nn.Module):

    def __init__(self, img_size=28, latent_size=64, hidden_size=64):
        super().__init__()

        self.model = nn.Sequential(
                nn.Linear(latent_size, hidden_size),
                nn.LeakyReLU(),
                nn.Dropout(),
                nn.Linear(hidden_size, hidden_size * 4),
                nn.LeakyReLU(),
                nn.Dropout(),
                nn.Linear(hidden_size * 4, img_size * img_size),
                nn.Tanh(),
                nn.Unflatten(1, (img_size, img_size))
                )

    def forward(self, x):
        return self.model(x)


def train(weights_path, lr_d, lr_g, epochs):
    device = "mps"

    batch_size = 32

    transform = v2.Compose([v2.ToTensor(), v2.Normalize(mean=[0.5], std=[0.5])])
    train_mnist = torchvision.datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_mnist = torchvision.datasets.MNIST("./data", train=False, download=True, transform=transform)
    train_data_loader = DataLoader(dataset=train_mnist, batch_size=batch_size, shuffle=True)
    test_data_loader = DataLoader(dataset=test_mnist, batch_size=batch_size, shuffle=False)

    discriminator = Discriminator().to(device)

    latent_size = 64
    generator = Generator(latent_size=latent_size).to(device)

    loss_function = nn.BCELoss()

    optim_d = torch.optim.Adam(discriminator.parameters(), lr=lr_d)
    optim_g = torch.optim.Adam(generator.parameters(), lr=lr_g)

    for epoch in range(epochs):
        running_loss_train_d = 0.0
        running_loss_train_g = 0.0
        n_d, n_g = 0, 0
        for x_real, _ in train_data_loader:
            # discriminator train
            x_real = x_real.to(device)

            y_real = discriminator(x_real)
            y_label_real = torch.ones((batch_size, 1)).to(device)

            z = torch.randn((batch_size, latent_size)).to(device)
            x_fake = generator(z)
            y_fake = discriminator(x_fake)
            y_label_fake = torch.zeros((batch_size, 1)).to(device)

            y = torch.cat((y_real, y_fake))
            y_label = torch.cat((y_label_real, y_label_fake))
            loss = loss_function(y, y_label)
            optim_d.zero_grad()
            loss.backward()
            optim_d.step()

            running_loss_train_d += loss.item() * y.shape[0]
            n_d += y.shape[0]

            # generator train
            z = torch.randn((batch_size, latent_size)).to(device)
            x_fake = generator(z)
            y_fake = discriminator(x_fake)
            loss = loss_function(y_fake, y_label_real)
            optim_g.zero_grad()
            loss.backward()
            optim_g.step()

            running_loss_train_g += loss.item() * y_fake.shape[0]
            n_g += y_fake.shape[0]

        running_loss_train_d /= n_d
        running_loss_train_g /= n_g
        print(epoch, running_loss_train_d, running_loss_train_g)

    torch.save(discriminator.state_dict(), "d.pth")
    torch.save(generator.state_dict(), "g.pth")


def test_generator(weights_path):
    import matplotlib.pyplot as plt

    generator = Generator()
    generator.load_state_dict(torch.load(weights_path))
    generator.eval()

    N = 50
    batch_size = N * N
    latent_size = 64
    z = torch.randn((batch_size, latent_size))
    x = generator(z)
    print(x.shape)

    x -= x.min()
    x /= x.max()
    x = x.detach().numpy()
    print(x.shape)

    X = np.zeros((28 * N, 28 * N))
    for ii in range(batch_size):
        iy = ii // N
        ix = ii % N
        i0 = iy * 28 
        j0 = ix * 28
        i1 = i0 + 28
        j1 = j0 + 28
        X[i0:i1, j0:j1] = x[ii, :, :]

    print(z)
    plt.imshow(X, cmap="gray")
    plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("op", help="train|visualize")
    args = parser.parse_args()
    if args.op == "train":
        train(None, 0.001, 0.001, 500)
    elif args.op == "visualize":
        test_generator("g.pth")
    else:
        parser.print_help()
