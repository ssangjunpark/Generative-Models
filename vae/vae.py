import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision
import matplotlib.pyplot as plt

class Encoder(nn.Module):
    def __init__(self, dims): 
        super().__init__()
        nets = []

        for i in range(len(dims) - 1):
            nets.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                nets.append(nn.ReLU())

        self.fc = nn.Sequential(*nets)

    def forward(self, x):
        fc_out = self.fc(x)
        
        split = fc_out.shape[1] // 2
        mean = fc_out[:, :split]
        var = torch.exp(fc_out[:, split:])

        return mean, var

class Decoder(nn.Module):
    def __init__(self, dims): 
        super().__init__()
        nets = []

        for i in range(len(dims) - 1):
            nets.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                nets.append(nn.ReLU())

        self.fc = nn.Sequential(*nets)
        
    def forward(self, x):
        return torch.sigmoid(self.fc(x))

def train_vae(epochs=10, batch_size=256, latent_dim=20):
    train_data = datasets.MNIST("./", train=True, download=True, transform=torchvision.transforms.ToTensor())
    test_data = datasets.MNIST("./", train=False, download=True, transform=torchvision.transforms.ToTensor())

    train = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4) 
    test = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = Encoder([784, 512, 256, latent_dim * 2]).to(device)
    decoder = Decoder([latent_dim, 256, 512, 784]).to(device)
    
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3)
    
    for ep in range(epochs):
        train_loss = 0
        for idx, (data, target) in enumerate(train):
            data = data.view(data.size(0), -1).to(device)
            
            mean, var = encoder(data)
            z = mean + torch.sqrt(var) * torch.randn_like(mean)
            recon = decoder(z)
            
            recon_loss = torch.nn.functional.binary_cross_entropy(recon, data, reduction='sum') / data.size(0)
            kl_loss = -0.5 * torch.sum(1 + torch.log(var) - mean**2 - var) / data.size(0)
            loss = recon_loss + kl_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        print(f"Epoch {ep+1}/{epochs}, Loss: {train_loss / len(train):.4f}")
    
    torch.save(encoder.state_dict(), 'encoder.pth')
    torch.save(decoder.state_dict(), 'decoder.pth')

def inference_vae(num_samples=9, latent_dim=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    decoder = Decoder([latent_dim, 256, 512, 784]).to(device)
    decoder.load_state_dict(torch.load('decoder.pth', map_location=device))
    decoder.eval()
    
    with torch.no_grad():
        z = torch.randn(num_samples, latent_dim).to(device)
        samples = decoder(z)
        # breakpoint()
        samples = samples.view(num_samples, 28, 28, 1)
    
    samples = samples.cpu().numpy()
    
    f, axarr = plt.subplots(3, 3, figsize=(8, 8)) 
    axarr = axarr.flatten()
    for i in range(samples.shape[0]):
        axarr[i].imshow(samples[i].squeeze(), cmap='gray')
        axarr[i].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    train_vae()
    inference_vae()
