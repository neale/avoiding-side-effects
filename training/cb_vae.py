###############################################################################

# CB-VAE code adapted from https://github.com/Robert-Aduviri/Continuous-Bernoulli-VAE

###############################################################################

#MIT License

#Copyright (c) 2019 Robert Aduviri

#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:

#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.

#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.a
###############################################################################

import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from torchvision.utils import save_image

class VAE(nn.Module):
    # VAE model 
    ## Architectured Based on Appendix by Authors
    ## https://arxiv.org/src/1907.06845v4/anc/cont_bern_aux.pdf
    def __init__(self, z_dim, input_dim):
        super(VAE, self).__init__()
        # Encoder layers
        self.z_dim = z_dim
        self.input_dim = input_dim
        self.fc1 = nn.Linear(input_dim*input_dim, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc3a1 = nn.Linear(256, 128)
        self.fc3b1 = nn.Linear(256, 128)
        self.fc3a2 = nn.Linear(128, z_dim)
        self.fc3b2 = nn.Linear(128, z_dim)
        # Decoder layers
        self.fc4 = nn.Linear(z_dim, 128)
        self.fc5 = nn.Linear(128, 256)
        self.fc6 = nn.Linear(256, 512)
        self.fc7 = nn.Linear(512, 512)
        self.fc8 = nn.Linear(512, input_dim*input_dim)
        
    def encode(self, x):
        #Recognition function
        h1 = F.elu(self.fc1(x))
        h2 = F.elu(self.fc2(h1))
        h3 = F.elu(self.fc3(h2))
        h3a1 = F.elu(self.fc3a1(h3))
        h3b1 = F.elu(self.fc3b1(h3))
        return self.fc3a2(h3a1), F.softplus(self.fc3b2(h3b1)) 

    def reparameterize(self, mu, std):
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        #Likelihood function
        h4 = F.elu(self.fc4(z))
        h5 = F.elu(self.fc5(h4))
        h6 = F.elu(self.fc6(h5))
        h7 = F.elu(self.fc7(h6))
        sigma = None
        return torch.sigmoid(self.fc8(h7)), sigma # Gaussian mean

    def forward(self, x):
        mu, std = self.encode(x.view(-1, self.input_dim*self.input_dim))
        z = self.reparameterize(mu, std)
        # Return decoding, mean and logvar
        return self.decode(z), mu, 2.*torch.log(std) 


def sumlogC( x , eps = 1e-5):
    '''
    Numerically stable implementation of 
    sum of logarithm of Continous Bernoulli
    constant C, using Taylor 3nd degree approximation
        
    Parameter
    ----------
    x : Tensor of dimensions (batch_size, dim)
        x takes values in (0,1)
    ''' 
    x = torch.clamp(x, eps, 1.-eps) 
    mask = torch.abs(x - 0.5).ge(eps)
    far = torch.masked_select(x, mask)
    close = torch.masked_select(x, ~mask)
    far_values =  torch.log( (torch.log(1. - far) - torch.log(far)).div(1. - 2. * far) )
    close_values = torch.log(torch.tensor((2.))) + torch.log(1. + torch.pow( 1. - 2. * close, 2)/3. )
    return far_values.sum() + close_values.sum()


def loss_vae(recon_x, x, mu, logvar):
    input_dim = x.size(-1)
    BCE = F.binary_cross_entropy(recon_x[0], x.view(-1, input_dim*input_dim), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


def loss_cbvae(recon_x, x, mu, logvar):
    input_dim = x.size(-1)
    BCE = F.binary_cross_entropy(recon_x[0], x.view(-1, input_dim*input_dim), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    LOGC = -sumlogC(recon_x[0])
    return BCE + KLD + LOGC


def encode_state(model, state, device):
    input_dim = state.size(-1)
    state = state.to(device)
    mu, std = model.encode(state.view(-1, input_dim*input_dim))
    z = model.reparameterize(mu, std)
    return z


def load_state_encoder(z_dim, path, input_dim, device):
    model = VAE(z_dim, input_dim).to(device)
    ckpt = torch.load(path)
    model.load_state_dict(ckpt)

def train_encoder(device, data, z_dim=16, training_epochs=200, exp='test', batch_size=128, log_interval=10):
    input_dim = data.size(-1)
    model = VAE(z_dim, input_dim).to(device)
    loss_fn = loss_cbvae
    batch_size = 64
    dataset = torch.utils.data.TensorDataset(data)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size, drop_last=True, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    total_batches = 0
    recon_batch = None

    for epoch in range(1, training_epochs + 1):
        model.train()
        train_loss = 0
        for batch_idx, data in enumerate(dataloader):
            data = torch.stack(data).float().squeeze(0)
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = loss_fn(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            total_batches += 1
            if total_batches % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]'.format(
                    epoch, batch_idx * len(data), len(dataloader.dataset),
                    100. * batch_idx / len(dataloader)))
    return model

