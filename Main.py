from sklearn.model_selection import train_test_split
import torch 
from torch import alpha_dropout, nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from torch.utils.data import DataLoader, TensorDataset


def create_network():
    
    class NeuralNetwork(nn.Module):
        
        def __init__(self) -> None:
            super().__init__()
            self.input = nn.Linear(2,64)
            self.hl1 = nn.Linear(64,32)
            self.bnorm1 = nn.BatchNorm1d(64)
            self.hl2 = nn.Linear(32,16)
            self.bnorm2 = nn.BatchNorm1d(32)
            self.output = nn.Linear(16,1)

        def forward(self,x):
          
            x = self.input(x)
            x = F.leaky_relu(x)

            # x = self.bnorm1(x)
            x = self.hl1(x)
            x = F.leaky_relu(x)

            # x = self.bnorm2(x)
            x = self.hl2(x)
            x = F.leaky_relu(x)

            x = self.output(x)

            return x

    neural_net = NeuralNetwork()
    return neural_net

def generate_data(a:int, b:int, samples:int = 1000):
    data = np.random.uniform(low = a, high = b,size = (samples,2))
    y = np.array([i[0]+i[1] for i in data])
    y = y.reshape((samples,1))

    return data, y

def main() -> None:
    data,y = generate_data(-1000,1000)

    data = stats.zscore(data)

    dataT = torch.tensor(data).float()
    yT = torch.tensor(y).float()
    
    train_data, test_data, train_y, test_y = train_test_split(dataT, yT, test_size=0.01)

    train_data = TensorDataset(train_data, train_y)
    batch_size = 10
    train_loader = DataLoader(train_data, batch_size=batch_size)

    epoch = 30
    lossfunction = nn.MSELoss()
    neural_net = create_network()
    optimizer = torch.optim.Adam(neural_net.parameters(),lr=0.01, amsgrad=True, weight_decay= 0.01  )

    dataT = torch.tensor(data)
    yT = torch.tensor(y)

    loss_list = list()

    for epochi in range(epoch):
        print(epochi)
        for X,y in train_loader:

            yHat = neural_net(X)
            
            loss = lossfunction(yHat,y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        yHat = neural_net(test_data)
        epoch_loss = torch.mean(((test_y-yHat)**2))
        print(epoch_loss.detach())
        loss_list.append(epoch_loss.detach())

    print("got out")

    yHat = neural_net(test_data)

    print(yHat)
    print(test_y)

    plt.plot(loss_list,'o')
    plt.show()

    
 

if __name__ == "__main__":
    main()
