import torch
import torch.nn.functional as F
import torch.nn as nn

class autoencoderMLP4Layer(nn.Module):

	def __init__(self, N_input=784, N_bottleneck=8, N_output=784):
		super(autoencoderMLP4Layer, self).__init__()
		N2 = int(N_input/2)
		self.fc1 = nn.Linear(N_input, N2)        # input = 1x784, output = 1x392
		self.fc2 = nn.Linear(N2, N_bottleneck)  # output = 1xN
		self.fc3 = nn.Linear(N_bottleneck, N2)  # output = 1xN2
		self.fc4 = nn.Linear(N2, N_output)      # output = 1x784
		self.type = "MLP4"
		self.input_shape = (1, 28*28)

	def forward(self, X):
		return self.decode(self.encode(X))
	
	##### Step 6: Bottleneck Interpolation ################################################
	def encode(self, X):
		"""
		The encode method takes the same input as the existing forward method (i.e. the flattened image tensor), 
		and returns the bottleneck tensor.
		"""
		X = self.fc1(X)
		X = F.relu(X)
		X = self.fc2(X)
		X = F.relu(X)
		return X

	def decode(self, X):
		"""
		The decode method takes as input the bottleneck tensor, and returns the 
		reconstructed image. 
		"""
		X = self.fc3(X)
		X = F.relu(X)
		X = self.fc4(X)
		X = torch.sigmoid(X)
		return X



