import torch
from torch import Tensor
from torch.nn import Linear, MSELoss, functional as F
from torch.optim import SGD, Adam, RMSprop
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

def data_generator(data_size=50):
    inputs = []
    labels = []
    
    # loop data_size times to generate the data
    for ix in range(data_size):
        
        # generate a random number between 0 and 1000
        x = np.around(np.random.randint(1000) / 1000 - .5, 2)
        
        # calculate the y value using the function 8x^2 + 4x - 3
        y = np.around(10*x*x + (np.random.rand(1)-.5), 2)
        
        # append the values to our input and labels lists
        inputs.append([x])
        labels.append([y])
        
    return inputs, labels

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = Linear(1, 8)
        self.fc2 = Linear(8, 6)
        self.fc3 = Linear(6, 1)
        
    def forward(self, x):
        x = F.dropout(F.relu(self.fc1(x)), p=0.5)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
model = Net()
# define the loss function
critereon = MSELoss()
# define the optimizer
optimizer = SGD(model.parameters(), lr=0.01)


nb_epochs = 100
data_size = 1000

# create our training loop
for epoch in range(nb_epochs):
    X_train, y_train = data_generator(data_size)
    
    epoch_loss = 0;
    
    for ix in range(data_size):
        y_pred = model.forward(Variable(Tensor(X_train[ix])))
    
        loss = critereon(y_pred, Variable(Tensor(y_train[ix]), requires_grad=False))
        
        epoch_loss += loss.data.item()
    
    	# ensure that we zero all the gradients in the model, otherwise we will just be adding onto them and end up with HUGE gradients
        optimizer.zero_grad()
    
    	# compute the gradient of the loss with respect to the model parameters
        loss.backward()
    	
    	# make an update (step) on the parameters
        optimizer.step()
    
    print("Epoch: {} Loss: {}".format(epoch, epoch_loss/data_size))


# evaluation
X_test, y_test = data_generator(100)
y_pred = model(Variable(Tensor(X_test))).data.numpy()

# y_pred = [[np.around(i.item(), 3)] for i in model(Variable(Tensor(X_test))).data]
# print(prediction)
# print("Prediction: {}".format(y_pred))
# print("Expected: {}".format(y_test))

plt.scatter(X_test, y_test, marker='o')
plt.scatter(X_test, y_pred, marker='+')
plt.show()

print('w1, b1:\n\n')
print(model.fc1.weight.data.numpy(),'\n',model.fc1.bias.data.numpy())

print('\n\nw2, b2:\n\n')
print(model.fc2.weight.data.numpy(),'\n',model.fc2.bias.data.numpy())

print('\n\nw3, b3:\n\n')
print(model.fc3.weight.data.numpy(),'\n',model.fc3.bias.data.numpy())





