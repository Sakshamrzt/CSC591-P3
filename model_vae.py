"""
Port PyTorch Quickstart to NNI
==============================
This is a modified version of `PyTorch quickstart`_.

It can be run directly and will have the exact same result as original version.

Furthermore, it enables the ability of auto tuning with an NNI *experiment*, which will be detailed later.

It is recommended to run this script directly first to verify the environment.

There are 2 key differences from the original version:

1. In `Get optimized hyperparameters`_ part, it receives generated hyperparameters.
2. In `Train model and report accuracy`_ part, it reports accuracy metrics to NNI.

.. _PyTorch quickstart: https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
"""

# %%
import nni
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
# %%
# Hyperparameters to be tuned
# ---------------------------
# These are the hyperparameters that will be tuned.
params = {
    'features': 512,
    'lr': 0.001
}

# %%
# Get optimized hyperparameters
# -----------------------------
# If run directly, :func:`nni.get_next_parameter` is a no-op and returns an empty dict.
# But with an NNI *experiment*, it will receive optimized hyperparameters from tuning algorithm.
optimized_params = nni.get_next_parameter()
params.update(optimized_params)
print(params)

# %%
# Load dataset
# ------------
batch_size = 64
train_dataloader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True)
# %%
# Build model with hyperparameters
# --------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + std*eps

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

model = VAE().to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)

# %%
# Define train and test
# ---------------------
def train(model,optimizer):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_dataloader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    #     if batch_idx % args.log_interval == 0:
    #         print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
    #             epoch, batch_idx * len(data), len(train_loader.dataset),
    #             100. * batch_idx / len(train_loader),
    #             loss.item() / len(data)))

    # print('====> Epoch: {} Average loss: {:.4f} and recons avg. {:.4f} and KL loss avg {:.4f}'.format(
    #         epoch, train_loss / len(train_loader.dataset),recons_loss/len(train_loader.dataset),KL_loss/len(train_loader.dataset)))

def test(model):
    model.eval()
    test_loss = 0
    with torch.no_grad():
       for i, (data, _) in enumerate(test_dataloader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
    test_loss /= len(test_dataloader.dataset)
    return test_loss


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD

# %%
# Train model and report accuracy
# -------------------------------
# Report accuracy metrics to NNI so the tuning algorithm can suggest better hyperparameters.
epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train( model,  optimizer)
    loss = test(model)
    nni.report_intermediate_result(loss)
nni.report_final_result(loss)