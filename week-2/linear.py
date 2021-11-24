import numpy
import torch
import numpy as np
import torch.nn.functional as F
#from utils import *
import matplotlib.pyplot as plt

x1 = np.arange(-10, 10.0).reshape((20,1))

x2 = np.arange(-20, 20.0, 2).reshape((20,1))

X = np.concatenate([x1, x2], axis=1)

Y = 2.5 + 3.59 * X[:, 0]  - 7.8 * X[:, 1]


#
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
#
# ax.scatter(x1, x2, Y)
#
# ax.set_xlabel('x1 ')
# ax.set_ylabel('x2 ')
# ax.set_zlabel('Y ')
#
# plt.show()

def normal_equation(X: np.ndarray, y: np.ndarray)->np.ndarray:
    theta = np.linalg.inv(X.T @ X) @ X.T @ y
    return theta

# class LinearRegression(torch.nn.Module):
#     def __init__(self):
#         super(LinearRegression, self).__init__()
#         self.weights = torch.randn(size=(2,1))
#         self.weights.requires_grad = True
#         self.biases = torch.zeros(size=(1,1))
#         self.biases.requires_grad = True
#         self.loss_fn = torch.nn.MSELoss()
#         self.optimizer = torch.optim.SGD(lr=1e-3, params=[self.weights, self.biases])
#
#     def make_data(self, X, Y):
#         self.X = torch.tensor(X, dtype=torch.float32)
#         self.X.requires_grad = True
#         self.Y = torch.tensor(Y, dtype=torch.float32).resize([20,1])
#         self.Y.requires_grad = True
#
#     def forward(self, x):
#         out = torch.matmul(x, self.weights) + self.biases
#         out.require_grad = True
#         return out
#
#     def train_step(self, x, y, lr=1e-3):
#         y_hat = self.forward(x)
#         loss = self.loss_fn(y_hat, y)
#
#         #loss.backward()
#         #self.optimizer.step()
#
#
#         #SGD - Stochastic Gradient Descent
#         grad_w = torch.autograd.grad(inputs=self.weights, outputs=loss, allow_unused=True, retain_graph=True)
#         grad_b = torch.autograd.grad(inputs=self.biases, outputs=loss, allow_unused=True, retain_graph=True)
#         #
#         self.weights = self.weights - lr*grad_w
#         self.biases = self.biases - lr*grad_b
#
#         return loss
#
#     def train(self, epochs: int):
#         """
#         Args: epochs: number of steps to train the model
#         """
#
#         for e in range(epochs):
#             loss =0
#             #for (x,y) in zip(self.X, self.Y):
#             running_loss = self.train_step(self.X, self.Y)
#             loss = running_loss
#             print(f"Epoch: {e+1} ------------ Loss = {loss}")
#         return loss


class LinearRegression(torch.nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linearlayer = torch.nn.Linear(in_features=2, out_features=1, bias=True)
        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = torch.optim.SGD(lr=1e-4, params=self.linearlayer.parameters())

    def make_data(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32).resize(20, 2)
        self.Y = torch.tensor(Y, dtype=torch.float32).resize(20, 1)

    def forward(self, x):
        #ut = torch.matmul(x, self.weights) + self.biases
        out = self.linearlayer(x)
        #out.require_grad = True
        return out

    def train_step(self, x, y, lr=1e-3):
        y_hat = self.forward(x)
        loss = self.loss_fn(y_hat, y)

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()


        # #SGD - Stochastic Gradient Descent
        # grad_w = torch.autograd.grad(inputs=self.weights, outputs=loss, allow_unused=True, retain_graph=True)
        # grad_b = torch.autograd.grad(inputs=self.biases, outputs=loss, allow_unused=True, retain_graph=True)
        # #
        # self.weights = self.weights - lr*grad_w
        # self.biases = self.biases - lr*grad_b

        return loss

    def train(self, epochs: int):
        """
        Args: epochs: number of steps to train the model
        """

        for e in range(epochs):
            loss =0
            #for (x,y) in zip(self.X, self.Y):
            running_loss = self.train_step(self.X, self.Y)
            loss = running_loss
            print(f"Epoch: {e+1} ------------ Loss = {loss}")

        print(self.linearlayer.state_dict())

        return loss


def main():
    linear_regressor = LinearRegression()
    linear_regressor.make_data(X, Y)
    #linear_regressor.train(1300)

    print(linear_regressor.forward(linear_regressor.X))
if __name__ == '__main__':
    main()



