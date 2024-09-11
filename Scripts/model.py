# Standard library
import numpy as np
# Pytorch library
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

class VAE(nn.Module):
    def __init__(self, input_dim,train_data, val_data, hidden1 = 128, hidden2 = 40,hidden3 = 10,hidden4 =10,
                code = 2, learning_rate=0.00001,batch_size=64, epochs=50,
                shuffle=True, thresh = 0.0001):
        
        """
        input_dim : flattened input vector length
        hidden1 : node number of hidden layer 1
        hidden2 : node number of hidden layer 2
        code : dimension of latent space
        learning_rate : name suggests
        thresh : thresh to compare while earlystopping
        train_data : trainning dataset
        val_data : validation dataset
        """
        
        super(VAE, self).__init__()
        
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.epochs = epochs
        self.shuffle = shuffle
        self.thresh = thresh
        self.learning_rate = learning_rate
        self.train_data = train_data
        self.val_data = val_data
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.hidden3 = hidden3
        self.hidden4 = hidden4
        
        torch.manual_seed(1)
        np.random.seed(1)
        self.learning_rate = learning_rate
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, self.hidden1),
            nn.GELU(),
            nn.Linear(self.hidden1, self.hidden2),
            nn.GELU(),
            nn.Linear(self.hidden2, self.hidden3),
            nn.GELU(),
            nn.Linear(self.hidden3, self.hidden4),
            nn.GELU(),
        )
        self.mu = nn.Linear(self.hidden4, code)
        self.logvar = nn.Linear(self.hidden4, code)
        self.decoder = nn.Sequential(
            nn.Linear(code, self.hidden4),
            nn.GELU(),
            nn.Linear(self.hidden4, self.hidden3),
            nn.GELU(),
            nn.Linear(self.hidden3, self.hidden2),
            nn.GELU(),
            nn.Linear(self.hidden2, self.hidden1),
            nn.GELU(),
            nn.Linear(self.hidden1, input_dim),
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def encode(self, x):
        encoded = self.encoder(x)
        mu = self.mu(encoded)
        logvar = self.logvar(encoded)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar, encoded
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        z, mu, logvar, _ = self.encode(x)
        decoded = self.decode(z)
        return decoded, mu, logvar

    def fit(self):
        # sourcery skip: low-code-quality
        if torch.cuda.is_available():
            self.to('cuda')
        train_dataset = TensorDataset(torch.Tensor(self.train_data))
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=self.shuffle)

        val_dataset = TensorDataset(torch.Tensor(self.val_data))
        val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=self.shuffle)
        
        criterion = nn.MSELoss(reduction='sum')
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        best_val_loss = float('inf')
        prv = float('inf')
        self.loss_train = []
        self.loss_val = []
        stop_counter = 0
        for epoch in range(self.epochs):
            train_loss = 0
            for x_batch, in train_dataloader:
                x_batch = x_batch.to('cuda') if torch.cuda.is_available() else x_batch
                optimizer.zero_grad()
                decoded, mu, logvar = self(x_batch)
                recon_loss = criterion(decoded, x_batch)
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + kl_loss
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            val_loss = 0
            for x_batch, in val_dataloader:
                x_batch = x_batch.to('cuda') if torch.cuda.is_available() else x_batch
                decoded, mu, logvar = self(x_batch)
                recon_loss = criterion(decoded, x_batch)
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + kl_loss
                val_loss += loss.item()
            val_loss = val_loss / len(val_dataloader)
            if abs(val_loss - prv) < self.thresh:
                break
            elif val_loss < best_val_loss:
                best_val_loss = val_loss
                stop_counter = 0
                # You can save the best model using torch.save(self.state_dict(), 'best_model.pt')
            else:
                stop_counter += 1
                if stop_counter >= 2:
                    print('Early stopping')
                    break
                    
            prv = val_loss
            self.loss_train.append(train_loss/(self.batch_size * len(train_dataloader)))
            self.loss_val.append(val_loss/self.batch_size)
            
            print(f'Epoch: {epoch + 1}/{self.epochs} | '
                  f'Train Loss: {train_loss / len(train_dataloader):.5f} | '
                  f'Val Loss: {val_loss:.5f} | ')  