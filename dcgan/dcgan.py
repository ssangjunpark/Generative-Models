import torch
import torch.nn as nn
import torchvision.datasets as datasets
from torchvision import transforms
import matplotlib.pyplot as plt

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.ConvTranspose2d(100, 512, (4, 4), (1, 1), (0, 0), bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, (4, 4), (2, 2), (1, 1), bias=True),
            nn.Tanh()
        )

    def forward(self, x):
        return self.seq(x)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(3, 64, (4, 4), (2, 2), (1, 1), bias=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1, (4, 4), (1, 1), (0, 0), bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.seq(x)
    

def train_dcgan(epochs=10, batch_size=256):
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_data = datasets.CelebA("./", split='train', download=False, transform=transform)
    test_data = datasets.CelebA("./", split='test', download=False, transform=transform)

    train = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
    test = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    optimizer_gen = torch.optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    optimizer_des = torch.optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))

    criterion = nn.BCELoss()

    for ep in range(epochs):
        generator.train()
        discriminator.train()
        train_loss_g = 0.0
        train_loss_d = 0.0

        for idx, (data, _) in enumerate(train):
            real = data.to(device)
            b_size = real.size(0)

            real_labels = torch.ones(b_size, device=device)
            fake_labels = torch.zeros(b_size, device=device)

            discriminator.zero_grad()
            out_real = discriminator(real).view(-1)
            loss_real = criterion(out_real, real_labels)

            noise = torch.randn(b_size, 100, 1, 1, device=device)
            fake_imgs = generator(noise)
            out_fake = discriminator(fake_imgs.detach()).view(-1)
            loss_fake = criterion(out_fake, fake_labels)

            loss_d = loss_real + loss_fake
            loss_d.backward()
            optimizer_des.step()

            generator.zero_grad()
            out_fake_for_gen = discriminator(fake_imgs).view(-1)
            loss_g = criterion(out_fake_for_gen, real_labels)
            loss_g.backward()
            optimizer_gen.step()

            train_loss_d += loss_d.item()
            train_loss_g += loss_g.item()
        

        avg_d = train_loss_d / len(train)
        avg_g = train_loss_g / len(train)
        print(f"Epoch {ep+1}/{epochs} | D_loss: {avg_d:.4f}, G_loss: {avg_g:.4f}")
    

        generator.eval()

        with torch.no_grad():
            noise = torch.randn(9, 100, 1, 1, device=device)
            out = generator(noise)
            out = out.detach().cpu()

        out = (out + 1.0) / 2.0
        out = out.permute(0, 2, 3, 1).numpy()

        f, axarr = plt.subplots(3, 3, figsize=(8, 8))
        axarr = axarr.flatten()
        for i in range(out.shape[0]):
            img = out[i]
            img = img.clip(0, 1)
            axarr[i].imshow(img)
            axarr[i].axis('off')

        plt.tight_layout()
        plt.savefig(f'./{ep+1}')

    torch.save(generator.state_dict(), 'generator.pth')
    torch.save(discriminator.state_dict(), 'discriminator.pth')

def inference_dcgan():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = Generator().to(device)
    state = torch.load('generator.pth', map_location=device)
    generator.load_state_dict(state)
    generator.eval()

    with torch.no_grad():
        noise = torch.randn(9, 100, 1, 1, device=device)
        out = generator(noise)

    out = out.detach().cpu()
    out = (out + 1.0) / 2.0
    out = out.permute(0, 2, 3, 1).numpy()

    f, axarr = plt.subplots(3, 3, figsize=(8, 8))
    axarr = axarr.flatten()
    for i in range(out.shape[0]):
        img = out[i]
        img = img.clip(0, 1)
        axarr[i].imshow(img)
        axarr[i].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    train_dcgan()
    # inference_dcgan()