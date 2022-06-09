import torch 
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn import preprocessing
import os
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# Hyperparameters
batch_size = 10
input_size = 7
hidden_size = 15
num_classes = 1
learning_rate = 0.001
epochs = 5


class BatteryDataSet(Dataset):
    
    def __init__(self):
        # Data loading
        # xy = dataset_raw.to_numpy(dtype=np.float32)
        xy = scaled_df_np
        self.x = torch.from_numpy(xy[:, 2:-2])
        self.y = torch.from_numpy(xy[:, [-1]])
        self.n_samples = xy.shape[0]
        
    def __getitem__(self, index):
        return self.x[index], self.y[index]
        
    def __len__(self):
        # len(Dataset)
        return self.n_samples


def classifyer(dataset, batch_size, shuffle_dataset=False):
    
    # get the dataset size
    dataset_len = len(dataset)
    
    dataset_size = torch.tensor([dataset_len])
    
    # get the indices
    indices = list(range(dataset_len))
    
    # percentage share of data set
    # train:        ~ 80 % 
    # test:         ~ 20 %
    
    # define borders
    first_split = int(torch.floor(0.8 * dataset_size))
    
    # set indices for train and test
    train_indices = indices[:first_split]
    test_indices = indices[first_split:]
    
    # shuffle the dataset
    if shuffle_dataset:
        np.random.seed()
        np.random.shuffle(train_indices)
        np.random.shuffle(test_indices)

    # set train dataset ot samplers and loader
    train_sampler = SubsetRandomSampler(train_indices)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
        
    # set test dataset ot samplers and loader
    test_sampler = SubsetRandomSampler(test_indices)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)
    
    return (train_loader, test_loader)


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.lstm1 = nn.LSTMCell(input_size, hidden_size)
        self.lstm2 = nn.LSTMCell(hidden_size, 20)
        self.lstm3 = nn.LSTMCell(20, 30)
        self.lstm4 = nn.LSTMCell(30, 40)
        self.linear = nn.Linear(40, num_classes)

    def forward(self, x):
        outputs = []
        n_samples = x.size(0)
        h_t = torch.zeros(n_samples, self.hidden_size, dtype=torch.float32)
        c_t = torch.zeros(n_samples, self.hidden_size, dtype=torch.float32)
        h_t2 = torch.zeros(n_samples, 20, dtype=torch.float32)
        c_t2 = torch.zeros(n_samples, 20, dtype=torch.float32)
        h_t3 = torch.zeros(n_samples, 30, dtype=torch.float32)
        c_t3 = torch.zeros(n_samples, 30, dtype=torch.float32)
        h_t4 = torch.zeros(n_samples, 40, dtype=torch.float32)
        c_t4 = torch.zeros(n_samples, 40, dtype=torch.float32)

        for input_t in x.split(7, dim=1):
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            h_t3, c_t3 = self.lstm3(h_t2, (h_t3, c_t3))
            h_t4, c_t4 = self.lstm4(h_t3, (h_t4, c_t4))
            
            output = self.linear(h_t4)
            outputs.append(output)
        
        outputs = torch.cat(outputs, dim=1)
        return outputs
    

# Training function

def train_loop(train_loader, model, loss_fn, optimizer):
    size = len(train_loader)
    for batch, (features, RUL) in enumerate(train_loader):
        
        # Forward path
        outputs = model(features)
        loss = loss_fn(outputs, RUL)
        
        # Backwards path
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # if batch % 100 == 0:
            # loss, current = loss.item(), batch*len(features)
            # print(f'loss: {loss:>7f} [{current:>5d}/{size:>5d}]')
              
def test_loop(dataloader, model, loss_fn):
    # size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    diff_list = []
    
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()

            # Difference between prediciotion and target
            diff = abs(y - pred) / y
            diff = diff.numpy()
            mean_diff = np.mean(diff)
            diff_list.append(mean_diff)
                

    test_loss /= num_batches
    difference_mean = np.mean(diff_list)
    print(f"Test: \n Avg Difference: {(100*difference_mean):>0.2f}%, Avg loss: {test_loss:>8f} \n")        

    min_diff_dict[t+1] = (difference_mean*100)
    min_diff_value = min(min_diff_dict.items(), key=lambda x:x[1])
    print("LOWEST DIFFERENCE AND EPOCH:")
    print(f"Epoch: {min_diff_value[0]}, diff: {min_diff_value[1]:>0.2f}%")
    plt.rcParams["figure.dpi"] = 600
    plt.scatter(t, difference_mean*100) 
    # plt.ylim(0, 60)
    plt.xlabel('Epoch')
    plt.ylabel('Target-Pred Difference (%)')
    
    #plt.rcParams["figure.dpi"] = 600
    #plt.scatter(t, test_loss)
    #plt.xlabel('Epoch')
   # plt.ylabel('loss (%)')
    #compariosion between predictiona and real value.
    
    
if __name__ == "__main__": 
    
    # Import data
    dataset_raw = pd.read_csv(os.getcwd() + '/Datasets/HNEI_Processed/Final Database.csv')
    dataset_raw.drop('Unnamed: 0', axis=1, inplace=True)
    
    # Feature scaling
    data = dataset_raw.values[:, :-1]    
    trans = MinMaxScaler()
    data = trans.fit_transform(data)
    dataset = pd.DataFrame(data)
    dataset_scaled = dataset.join(dataset_raw['RUL'])
    scaled_df_np = dataset_scaled.to_numpy(dtype=np.float32)
    
    
    dataset = BatteryDataSet()
    
    # Traind and test loader
    train_loader, test_loader = classifyer(dataset=dataset, batch_size=batch_size
                                                         , shuffle_dataset=True)
    # Init model
    model = LSTM(input_size, hidden_size, num_classes)
    
    # 
    loss_fn = nn.MSELoss()
    #loss_fn = nn.L1Loss()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 
    
    # Auxiliar dictionary to store epochs and difference values:
    min_diff_dict = {}

    
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        
        train_loop(train_loader, model, loss_fn, optimizer)
        
        test_loop(test_loader, model, loss_fn)
        
    print("Fertig!")                  
            
            
# Save model
# torch.save(NeuralNet.state_dict(), os.getcwd() + '/Datasets/FF_Net_1.pth')
            
            

