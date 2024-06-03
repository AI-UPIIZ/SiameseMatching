import torch
import torch.nn as nn

from models.resnet import resnet18


class SiameseNetwork(nn.Module):
	def __init__(self):
		super(SiameseNetwork, self).__init__()
		self.cnn = resnet18(pretrained=True)
		self.cnn.fc = nn.Linear(self.cnn.fc.in_features, 256)

		self.fc1 = nn.Linear(256 * 2, 512)
		self.fc2 = nn.Linear(512, 256)
		self.fc3 = nn.Linear(256, 1)

	def forward_once(self, x):
		x = self.cnn(x)
		return x

	def forward(self, input1, input2):
		output1 = self.forward_once(input1)
		output2 = self.forward_once(input2)

		combined = torch.cat((output1, output2), 1)
		combined = nn.ReLU()(self.fc1(combined))
		combined = nn.ReLU()(self.fc2(combined))
		combined = torch.sigmoid(self.fc3(combined))

		return combined
