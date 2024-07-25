import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Net(torch.nn.Module):
    def __init__(self, layers, activation):
        super(Net, self).__init__()

        input_size, output_size = 2, 1

        all_layers = []

        for i in layers[1:-1]:
            all_layers.append(nn.Linear(input_size, i))
            if activation == "relu":
                all_layers.append(nn.ReLU(inplace=True))
            if activation == "tanh":
                all_layers.append(nn.Tanh())
            input_size = i

        all_layers.append(nn.Linear(input_size, output_size))

        self.layers = nn.Sequential(*all_layers)
        
        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.00)

        self.layers.apply(init_weights)

    def forward(self, input):
        output = self.layers(input)
        return output

class ANNAbel:
    # Abel transform relates a projection of a field
    # like P(x,y) = A@f(x,y)
    # Here P is the objetive
    # A is the operator
    # and f the deconvolved values
    # Furthermore we need the coordinates x,y
    # this class functions for the more general case P = A@(f m)
    # where m is a multiplication factor for each f
    def __init__(self, 
                 coords, 
                 projected, 
                 operator, 
                 layers=[100]*3, 
                 activation="tanh", 
                 learning_rate=1e-3, 
                 device=device, 
                 mat = 1) -> None:
        self.obj_shape = projected.shape
        self.n_constant = projected.max()
        self.objective = projected/self.n_constant
        self.coords = coords
        self.operator = operator
        self.mat = mat
        
        self.net = Net(layers, activation).to(device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate)

        try:
            self.init_field = self.net(self.coords).reshape(self.obj_shape)
            self.init_proj = (self.operator@self.init_field.T).T
        except Exception as e:
            print("error:", e)
        
        self.history = {"fit":[],
                        "zero":[],
                        "loss":[]}
        self.best = self.init_field
        self.best_loss = 1e4
    
    def __call__(self, coords) -> np.array:
        out = self.net(coords)*self.n_constant
        return out.detach().cpu().numpy()
    
    def train(self, epochs, const = {"fit":1, "zero":0}, loss_fun= nn.MSELoss() ):
        mask = self.objective>0
        for _ in tqdm(range(epochs)):

            
            field_pred = self.net(self.coords).reshape(self.obj_shape)
            proj_pred = (self.operator@(field_pred*self.mat).T).T

            loss_data = loss_fun(self.objective,proj_pred) # close to data
            loss_zero = loss_fun(proj_pred[~mask],0*proj_pred[~mask]) # f=0 outside the mask

            LOSS  = loss_data*const["fit"]
            LOSS += loss_zero*const["zero"]
            
            self.history["fit"].append(loss_data.item())
            self.history["zero"].append(loss_zero.item())
            self.history["loss"].append(LOSS.item())
            
            if self.best_loss>loss_data.item():
                self.best = field_pred*self.n_constant
                self.best_loss = loss_data.item()
                        
            LOSS.backward()         # backpropagation, compute gradients

            self.optimizer.step()        # apply gradients
            self.optimizer.zero_grad()   # clear gradients for next train

    def history_plot(self, exclude=["fit","zero"]):
        plt.figure(figsize=(15,5))
        plt.semilogy()
        for e in self.history:
            if e in exclude: continue
            plt.plot(self.history[e], label=e)
        plt.xlabel("iter")
        plt.xlabel("loss")
        plt.legend()
        plt.grid()
        plt.show()      
