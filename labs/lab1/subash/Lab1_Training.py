import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

use_MNIST_data = True
use_cuda = False
epochs = 5

class Net_MNIST(nn.Module):
	def __init__(self):
		super(Net_MNIST, self).__init__()
		self.conv1 = nn.Conv2d(1, 20, 5, 1)
		self.conv2 = nn.Conv2d(20, 50, 5, 1)
		self.fc1 = nn.Linear(4*4*50, 500)
		self.fc2 = nn.Linear(500, 10)

	def forward(self, x):
			x = F.relu(self.conv1(x))
			x = F.max_pool2d(x, 2, 2)
			x = F.relu(self.conv2(x))
			x = F.max_pool2d(x, 2, 2)
			x = x.view(-1, 4*4*50)
			x = F.relu(self.fc1(x))
			x = self.fc2(x)
			return F.log_softmax(x, dim=1)



class Net_CIFAR10(nn.Module):
	def __init__(self):
		super(Net_CIFAR10, self).__init__()
		self.conv1 = nn.Conv2d(3, 6, 5)
		self.pool = nn.MaxPool2d(2, 2)
		self.conv2 = nn.Conv2d(6, 16, 5)
		self.fc1 = nn.Linear(16 * 5 * 5, 120)
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, 10)

	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		x = x.view(-1, 16 * 5 * 5)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return F.log_softmax(x, dim = 1)

def train(model, device, train_loader, optimizer, epoch, log_interval):
	model.train()
	for batch_idx, (data, target) in enumerate(train_loader):
			data, target = data.to(device), target.to(device)
			optimizer.zero_grad()
			output = model(data)
			loss = F.cross_entropy(output, target)
			loss.backward()
			optimizer.step()
			if batch_idx % log_interval == 0:
					print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
							epoch, batch_idx * len(data), len(train_loader.dataset),
							100. * batch_idx / len(train_loader), loss.item()))

def test(model, device, test_loader):
	model.eval()
	test_loss = 0
	correct = 0
	with torch.no_grad():
		for data, target in test_loader:
			data, target = data.to(device), target.to(device)
			output = model(data)
			test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
			pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
			correct += pred.eq(target.view_as(pred)).sum().item()

	test_loss /= len(test_loader.dataset)

	print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
		test_loss, correct, len(test_loader.dataset),
		100. * correct / len(test_loader.dataset)))

		
if __name__ == "__main__":
	torch.backends.cudnn.enabled = False
	use_cuda = use_cuda and torch.cuda.is_available()

	if use_MNIST_data:
			train_loader = torch.utils.data.DataLoader(
					datasets.MNIST('./MNIST_data', train=True, download=True,
											transform=transforms.Compose([
													transforms.ToTensor(),
													transforms.Normalize((0.1307,), (0.3081,))
											])),
					batch_size=64, shuffle=True)
			test_loader = torch.utils.data.DataLoader(
					datasets.MNIST('./MNIST_data', train=False, download=True,
											transform=transforms.Compose([
													transforms.ToTensor(),
													transforms.Normalize((0.1307,), (0.3081,))
											])),
		batch_size=64, shuffle=True)
	else:
			train_loader = torch.utils.data.DataLoader(
					datasets.CIFAR10(root='./CIFAR10_data', train=True,download=True,
											transform = transforms.Compose([
													transforms.ToTensor(),
													transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
											])),
					batch_size=64,shuffle=True)
			test_loader = torch.utils.data.DataLoader(
					datasets.CIFAR10(root='./CIFAR10_data', train=False,download=True,
											transform = transforms.Compose([
													transforms.ToTensor(),
													transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
											])),
					batch_size=64,shuffle=True)

	device = torch.device("cuda" if use_cuda else "cpu")
	net = Net_MNIST() if use_MNIST_data else Net_CIFAR10()
	model = net.to(device)
	optimizer = optim.SGD(model.parameters(), lr=0.5)

	print("training on data: {}".format("MNIST" if use_MNIST_data else "CIFAR10"))
	print("running on device: {}".format(device))
	print("running for epochs: {}".format(epochs))

	for epoch in range(1, epochs + 1):
					train(model, device, train_loader, optimizer, epoch, log_interval=10)
					test(model, device, test_loader)
