import torch
import torchvision
import torchvision.datasets as datasets
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from statistics import mean
from torch.utils.tensorboard import SummaryWriter

class VAE(nn.Module):

    def __init__(self,image_size=784,encoder_hidden=500, encoder_out=100, decoder_hidden=300):
        super().__init__()
        self.encoder = self.part1 = nn.Sequential(
            nn.Linear(image_size,encoder_hidden),
            nn.ReLU(),
            nn.Linear(encoder_hidden, encoder_hidden),
            nn.ReLU()
        )
        self.mu = nn.Linear(encoder_hidden,encoder_out)
        self.sigma = nn.Linear(encoder_hidden,encoder_out)
        
        self.decoder = nn.Sequential(
                nn.Linear(encoder_out,decoder_hidden),
                nn.ReLU(),
                nn.Linear(decoder_hidden, image_size),
                nn.Sigmoid()
        )

    def forward(self,x) :
        x = self.encoder(x)
        mu = self.mu(x)
        sigma = self.sigma(x)
        z = mu + sigma * torch.randn(mu.size())
        res = self.decoder(z)
        return res, mu, sigma
 



if __name__ == "__main__":
    mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
    mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=None)

    data_train = DataLoader(mnist_trainset)
    data_test = DataLoader(mnist_testset)

    tensor_train = data_train.dataset.data
    tensor_train = tensor_train.to(dtype=torch.float32)

    tr_train = tensor_train.reshape(tensor_train.size(0), -1)
    tr_train = tr_train/128
    
    targets_train = data_train.dataset.targets
    targets_train = targets_train.to(dtype=torch.long)

    tensor_test = data_test.dataset.data
    tensor_test = tensor_test.to(dtype=torch.float32)

    tr_test = tensor_test.reshape(tensor_test.size(0), -1)
    tr_test = tr_test/128
    
    targets_test = data_test.dataset.targets
    targets_test = targets_test.to(dtype=torch.long)

    BATCH_SIZE = 64

    train_ds = TensorDataset(tr_train, targets_train)
    train_dl = DataLoader(train_ds,batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

    test_ds = TensorDataset(tr_test,targets_test)
    test_dl = DataLoader(test_ds,batch_size=4, shuffle=True, drop_last=False)

    writer = SummaryWriter('runs/vae_experiments_1')
    
    
    lr=1e-3
    model=VAE(784,500,100,300)
    criterion=torch.nn.BCELoss() #binary cross entropy
    optimizer = torch.optim.Adam(params=model.parameters(),lr=lr)


    EPOCHS=10

    for k in range(EPOCHS):
        tab=[]
        for x,y in train_dl:
            model.train()
            #x=x.to(device)
            #x_temp=x.reshape(x.shape[0],x.shape[1]*x.shape[2]).float()
            y_pred, mu, sigma = model(x)
            BCE = criterion(y_pred,x)
            KLD = -0.5 * torch.sum(1 + sigma - mu.pow(2) - sigma.exp())
            loss = BCE + KLD
            tab.append(float(loss))
            loss.backward()
            optimizer.step()
        print(f"EPOCH {k} :{mean(tab)}")
        data_iter = iter(test_dl)
        images, label = data_iter.next()

        img_grid = torchvision.utils.make_grid(images.reshape((4,1,28,28)))

        writer.add_image('test_images', img_grid, k)
        
        images_vae, _, _= model(images)
        images_vae = images_vae.detach().reshape((4,1,28,28))

        img_grid_vae = torchvision.utils.make_grid(images_vae)
        
        writer.add_image('vae_images_from_test', img_grid_vae, k)

    writer.close()

        
    
        