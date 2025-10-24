import torch
import torch.nn.functional as F
import torch.nn as nn

class SnoutNetModel(nn.Module):

	def __init__(self):
		super(SnoutNetModel, self).__init__()
        # Convolution 1
		#    input = 277 x 277 x 3
		#    kernel = 3 x 3 x 3
		self.conv1 = nn.Conv2d(in_channels=1, out_channels=6,
							   kernel_size=5, stride=1, padding=2)
		
        # Convolution 2
        #    input 57 x 57 x 64
	    #    kernel = 3 x 3 x 64
		self.conv2 = nn.Conv2d(in_channels=1, out_channels=6,
							   kernel_size=5, stride=1, padding=2)
		
        # Convolution 3
        #   input = 15 x 15 x 128
		#   kernel = 3 x 3 x 128
		self.conv3 = nn.Conv2d(in_channels=1, out_channels=6,
							   kernel_size=5, stride=1, padding=2)
		
        # Fully Connected Layer 1
		#    input = flatten(4 x 4 x 256) = 1 x 4096
		self.fc1 = nn.Linear(4096, 1024)
		
        # Fully Connected Layer 2
		#    input = 1024
		self.fc2 = nn.Linear(1024, 1024)
		
		# Fully Connected Layer 3
		#    input = 1024
		self.fc3 = nn.Linear(1024, 2)
		

		
    #     # TODO Check these two below
	# 	self.type = "MLP4"
	# 	self.input_shape = (1, 28*28)

	def forward(self, X):
		return self.decode(self.encode(X))
	
	##### Step 6: Bottleneck Interpolation ################################################
	def regress(self, X):
		"""
		The encode method takes the same input as the existing forward method (i.e. the flattened image tensor), 
		and returns the bottleneck tensor.
		"""
		X = self.conv1(X)
		X = self.tanh(X)
		X = self.avgpool(X)
		
		X = self.conv2(X)
		X = self.tanh(X)
		X = self.avgpool(X)
		
		X = self.conv3(X)
		X = self.tanh(X)
		X = self.avgpool(X)
		
		X = X.reshape(X.shape[0], -1)
		X = self.fc1(X)
		X = F.relu(X)
		X = self.fc2(X)
		X = F.relu(X)
		X = self.fc3(X)
		return X


