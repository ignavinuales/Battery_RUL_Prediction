import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from torch.utils.data.sampler import SubsetRandomSampler
import os
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# Hyperparameters
batch_size = 6
input_size = 7
hidden_size = 10
num_classes = 1
learning_rate = 0.0001
epochs = 500


class BatteryDataSet(Dataset):

    def __init__(self):
        # Data loading
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


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, 50)
        self.relu = nn.ReLU()
        self.l3 = nn.Linear(50, 20)
        self.relu = nn.ReLU()
        self.l4 = nn.Linear(20, 5)
        self.relu = nn.ReLU()
        self.l5 = nn.Linear(5, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        out = self.relu(out)
        out = self.l4(out)
        out = self.relu(out)
        out = self.l5(out)
        return out


# Training function
def train_loop(train_loader, model, loss_fn, optimizer):
    # size = len(train_loader)
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


# Test function
def test_loop(dataloader, model, loss_fn):
    num_batches = len(dataloader)
    test_loss = 0

    diff_list = []
    targets_list = []
    pred_list = []

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()

            # Difference between prediction and target
            diff = abs(y - pred) / y
            diff = diff.numpy()
            mean_diff = np.mean(diff)
            diff_list.append(mean_diff)

            # # Target vs prediction
            pred_np = pred.squeeze().tolist()
            target_np = y.squeeze().tolist()

            try:
                for i in pred_np:

                    pred_list.append(i)
                for i in target_np:
                    targets_list.append(i)
            except:
                pass

    # Average loss
    test_loss /= num_batches

    # Average difference
    difference_mean = np.mean(diff_list)

    # Print the average difference and average loss
    print(f"Test: \n Avg Difference: {(100*difference_mean):>0.2f}%, Avg loss: {test_loss:>8f} \n")

    # Minimum difference and its epoch
    min_diff_dict[t+1] = (difference_mean*100)
    min_diff_value = min(min_diff_dict.items(), key=lambda x:x[1])
    print("LOWEST DIFFERENCE AND EPOCH:")
    print(f"Epoch: {min_diff_value[0]}, diff: {min_diff_value[1]:>0.2f}%")

    # PLOT Target vs Prediction
    # if t % 10 == 0:

    plt.rcParams["figure.dpi"] = 600
    plt.scatter(targets_list, pred_list)
    plt.xlabel('Target', fontsize=10)
    plt.ylabel('Prediction', fontsize=10)
    plt.ylim(0, 1300)
    plt.title(f"Epoch {t+1}", fontsize=13)
    plt.show()


    # PLOT Difference

    # plt.scatter(t, difference_mean*100)
    # plt.ylim(0, 70)
    # plt.xlabel('Epoch')
    # plt.ylabel('Target-Pred Difference (%)')
    # plt.scatter(t, test_loss)


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

    # Load dataset
    dataset = BatteryDataSet()

    # Train and test loader
    train_loader, test_loader = classifyer(dataset=dataset, batch_size=batch_size
                                                         , shuffle_dataset=True)
    # Init model
    model = NeuralNet(input_size, hidden_size, num_classes)

    # Loss function
    loss_fn = nn.L1Loss()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Auxiliary dictionary to store epochs and difference values:
    min_diff_dict = {}

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")

        train_loop(train_loader, model, loss_fn, optimizer)

        test_loop(test_loader, model, loss_fn)

    print("Fertig!")


# Save model
# torch.save(NeuralNet.state_dict(), os.getcwd() + '/Datasets/FF_Net_1.pth')
            
            

