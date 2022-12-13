import h5py
# import numpy as np
import pandas as pd
fpath = "datasets/mps_heis_data_L40_new.h5"

import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader


import copy
try: 
    import intel_extension_for_pytorch as ipex
except  Exception as  ex:
    ipex = None
    print ("ipex not found")

# Read in Matt's data
with h5py.File(fpath, 'r') as f:
    data = torch.tensor(f['heis-bd-data'][:-4,:,:])


cmt='''
File mps_heis_data.h5 has one array of shape (10121, 50, 4)
- index 1 runs over the 10121 random instances of heisenberg spin-1/2 H's. Each H is of the form
H = \sum_{j, j+1} J_j S_j \dot S_{j+1} + \sum_j h_j S^z_j
	- the couplings J_j are random gaussian variables with mean -1 (antiferromagnetic) and variance 0.1
	- the fields h_j are random gaussian variables with mean 0 and variance 0.1
	- the chain is of length L = 50
- index 2 runs over the physical index {j} of the spin chain
- index 3 labels the stored data as follows:
	- k == 0: vector of the J_j's. This is of size 49 (open boundary conditions) so there is a zero prepended
		e.g. element (100, 29, 0) stores the value of J_{28} which is the coupling between the 28th and 29th qubit
	- k == 1: vector of the h_j's. This is of size 50
	- k == 2: vector of the *uncompressed* bond dimension of the ground state MPS |psi>
	- k == 3: vector of the *compressed* BD of an approximate ground state MPS |phi>

The compressed approximation |phi> satisfies |<phi | psi>| > 1 - 1e-6.

Notes on DMRG:
	- The DRMG algorithm that produced |psi> kept all singular values above 1e-12
	- Convergence was deemed achieved at an additive energy tolerance of 1e-4.
'''


# np.array(data)
# print(data)
Nsites = 40

# Get max bonds, for coarse-graining later
print( data.shape )
data[0,:,2]

max_dmrg_bond = torch.max(data[:,:,2])
print(f'max dmrg bond: {max_dmrg_bond}')

max_overlap_bond = torch.max(data[:,:,3])
print(f'max overlap bond: {max_overlap_bond}')

# Coarse-grain the data

nslots = 30.

def coarsegrain_dmrg_bonddims(inp):
    return torch.round( (nslots/max_dmrg_bond)*inp )#.type(torch.int)

def uncoarsegrain_dmrg_bonddims(inp):
    return torch.round( (max_dmrg_bond/nslots)*inp )#.type(torch.int)

def coarsegrain_overlap_bonddims(inp):
    return torch.round( (nslots/max_overlap_bond)*inp )#.type(torch.int)

def uncoarsegrain_overlap_bonddims(inp):
    return torch.round( (max_overlap_bond/nslots)*inp )#.type(torch.int)

'''
# Quick tests:
print( coarsegrain_dmrg_bonddims(np.array([10,20,30])) )
print( uncoarsegrain_dmrg_bonddims(np.array([10,20,30])) )
print( coarsegrain_overlap_bonddims(np.array([10,20,30])) )
print( uncoarsegrain_overlap_bonddims(np.array([10,20,30])) )
'''

# Coarse-graining implemented here
coarsegr_data = data.type(torch.float32)
coarsegr_data[:,:,2] = coarsegrain_dmrg_bonddims(data[:,:,2])
coarsegr_data[:,:,3] = coarsegrain_overlap_bonddims(data[:,:,2])


# Split into training and testing data (do 85%-15%)
frac_train = 0.85
size_train = int(frac_train*coarsegr_data.shape[0])
print(size_train)
training_data = coarsegr_data[:size_train,:,:]
testing_data  = coarsegr_data[size_train:,:,:]

print(training_data.shape)
print(testing_data.shape)

# class CustomImageDataset(Dataset):
#     def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
#         self.img_labels = pd.read_csv(annotations_file)
#         self.img_dir = img_dir
#         self.transform = transform
#         self.target_transform = target_transform

#     def __len__(self):
#         return len(self.img_labels)

#     def __getitem__(self, idx):
#         img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
#         image = read_image(img_path)
#         label = self.img_labels.iloc[idx, 1]
#         if self.transform:
#             image = self.transform(image)
#         if self.target_transform:
#             label = self.target_transform(label)
#         return image, label

# Need to define DataSet class
# Must *use* a DataLoader, but don't need new class

class HeisDMRGbondsDataset(Dataset):
    
    def __init__(self,inp_data, transform=None, target_transform=None):
        self.raw_data = inp_data
        
    def __len__(self):
        return self.raw_data.shape[0]
    
    def __getitem__(self,idx):
        
        did = 20
        
        hamdata = self.raw_data[idx, :, 0:2]
#         hamdata = torch.expand_dims(self.raw_data[idx, :, 0:2],axis=0)
#         hamdata = torch.array([self.raw_data[idx, :, 0:2]])

        dvals   = self.raw_data[idx, :, 2] #[idx, :, 2:3]
#         dval   = self.raw_data[idx, did, 2] #[idx, :, 2:3]
        
        return hamdata, dvals#.astype(int)
#         return hamdata, int(dval)
        
        
# class HeisOverlapBondsDataset(Dataset):
        
        
# 'reg' means 'regular bonds'
# Note 'as type' to go to 32-bit float
reg_training_data = HeisDMRGbondsDataset(training_data)
reg_testing_data  = HeisDMRGbondsDataset(testing_data)

# Make sure output looks correct
print( reg_training_data[0] )
print( reg_testing_data[0] )


# Create DataLoaders
batch_size = 64

# Create data loaders.
reg_train_dataloader = DataLoader(reg_training_data,batch_size=batch_size)
reg_test_dataloader = DataLoader(reg_testing_data,batch_size=batch_size)

for X, y in reg_test_dataloader:
    print(f"Shape of X: {X.shape} {X.dtype}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break


print(nslots)

# Create the model

print("try *not* doing nslots for now... try to learn d values directly...")

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
device = "xpu" if ipex is not None  else "cpu"
print(f"Using {device} device")

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(Nsites*2, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
#             nn.Linear(512, int(nslots) )
            nn.Linear(512, Nsites )
            
        )

    def forward(self, x):
#         print(f"x: {x}")
        x = self.flatten(x)
#         x = nn.Flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)


def custom_loss(y_pred,y_val):
    #loss = torch.sum(torch.absolute(y_pred - y_val))
    #loss = nn.L1Loss()(y_pred,y_val)
    loss = nn.MSELoss()(y_pred,y_val)
    return loss

loss_fn = custom_loss


# loss_fn = nn.CrossEntropyLoss() # <-- not appropriate for us
# loss_fn = nn.L1Loss
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-3) # lr = learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3) # lr = learning rate

# Train function
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 20 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
#             print('pred: ',X)
#             print('y: ',y)
            

# Test function
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            #correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            correct += ((pred -y).abs() < 0.5 ).count_nonzero().item()
    test_loss /= num_batches
    #correct /= size
    correct /= num_batches*X.shape[0]*X.shape[1]
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    

epochs = 25
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(reg_train_dataloader, model, loss_fn, optimizer)
#     test(reg_test_dataloader, model, loss_fn)
print("Training done!")

# torch.save(model.state_dict(), "model.pth")
# print("Saved PyTorch Model State to model.pth")

# model = NeuralNetwork()
# model.load_state_dict(torch.load("model.pth"))

test(reg_test_dataloader, model, loss_fn)
print("Training done!")

model.eval()
x, y = reg_testing_data[0][0], reg_testing_data[0][1] 
# print(x)
# print(y) # true value
with torch.no_grad():
#     pred = model(x)
    pred = model(x.unsqueeze(0).to(device))
    # print(pred) 
#     predicted, actual = pred[0].argmax(0), y # It's like one-hot. Position with *highest* value gives answer
    predicted, actual = pred[0], y
    print(f'Predicted: "{predicted}", Actual: "{actual}"')

