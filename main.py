import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, in_features=8, h1=12, h2=14, out_features=2):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)

        return x

def graph(epochs: int, losses: list[float]) -> None:
    plt.plot(range(epochs), losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('loss_plot.png')
    plt.show()

torch.manual_seed(392)

model = Model()

dataframe = pd.read_csv('diabetes.csv')

X = dataframe.drop('Outcome', axis=1).values
y = dataframe['Outcome'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=392)

X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

epochs = 500
losses = []
print('Training...')
for i in range(1, epochs + 1):
    y_pred = model.forward(X_train)

    loss = criterion(y_pred, y_train)
    losses.append(loss)

    if i % 10 == 0:
        print(f'Epoch: {i} Loss: {loss.item()}')

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print('Training finished!\n')

graph(epochs, losses)

print(f'Testing...')
correct = 0
with torch.no_grad():
    for i, y in enumerate(X_test):
        y_val = model.forward(y)

        print(f'{i+1}.) {str(y_val)} \t {y_test[i]} \t {y_val.argmax().item()}')

        if y_val.argmax().item() == y_test[i]:
            correct += 1

print(f'{correct}/30 correct!')

torch.save(model.state_dict(), 'iris_model.pth')