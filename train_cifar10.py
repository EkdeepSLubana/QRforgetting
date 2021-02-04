import torch
import torchvision
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
from models import *
from config import *
import os
import argparse

######### Parser #########
parser = argparse.ArgumentParser()
parser.add_argument("--seed", help="set random generator seed", default='0')
parser.add_argument("--download", help="download CIFAR-10?", default='False')
args = parser.parse_args()

######### Setup #########
torch.manual_seed(int(args.seed))
cudnn.deterministic = True
cudnn.benchmark = False
device='cpu'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if(device == 'cuda'):
		print("Backend:", device)
else:
	raise Exception("Please use a cuda-enabled GPU.")

if not os.path.isdir('pretrained'):
	os.mkdir('pretrained')

pretrained_root = 'pretrained/'
base_sched, base_epochs, wd = [1e-1, 1e-2, 1e-3], [25, 25, 10], 1e-4

######### Dataloaders #########
transform = transforms.Compose(
	[transforms.RandomHorizontalFlip(),
	 transforms.ToTensor(),
	 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	 ])
transform_test = transforms.Compose(
	[transforms.ToTensor(),
	 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	 ])

trainset = torchvision.datasets.CIFAR10(root='./../datasets/cifar10', train=True, download=(args.download=='True'), transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root='./../datasets/cifar10', train=False, download=(args.download=='True'), transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

######### Loss #########
criterion = nn.CrossEntropyLoss()

######### Optimizers #########
def get_optimizer(net, lr, wd):
	optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
	return optimizer

######### Training functions #########
# Training
def train(net):
	net.train()
	train_loss = 0
	correct = 0
	total = 0
	for batch_idx, (inputs, targets) in enumerate(trainloader):
		inputs, targets = inputs.to(device), targets.to(device)
		optimizer.zero_grad()
		outputs = net(inputs)
		loss = criterion(outputs, targets)
		loss.backward()
		optimizer.step()
		train_loss += loss.item()
		_, predicted = outputs.max(1)
		total += targets.size(0)
		correct += predicted.eq(targets).sum().item()
		progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
			% (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

# Testing
def test(net, T=1.0):
	global cfg_state
	net.eval()
	test_loss = 0
	correct = 0
	total = 0
	with torch.no_grad():
		for batch_idx, (inputs, targets) in enumerate(testloader):
			inputs, targets = inputs.to(device), targets.to(device)
			outputs = net(inputs)
			loss = criterion(outputs, targets)
			test_loss += loss.item()
			_, predicted = outputs.max(1)
			total += targets.size(0)
			correct += predicted.eq(targets).sum().item()
			progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
				% (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
	# Save checkpoint.
	global best_acc
	acc = 100.*correct/total
	if acc > best_acc:
		print('Saving..')
		state = {'net': net.state_dict()}
		torch.save(state, pretrained_root+'vanilla_cnn_seed_' + args.seed +'.pth')
		best_acc = acc

# Create model for evaluation
def create_model():
	net = torch.nn.DataParallel(Vanilla_cnn(num_classes=10))
	return net

######### Determine model, load, and train #########
net = create_model()

# Train 
print("\n------------------ Training base model ------------------\n")
best_acc = 0
lr_ind = 0
epoch = 0
optimizer = get_optimizer(net, lr=base_sched[lr_ind], wd=wd)
while(lr_ind < len(base_sched)):
	optimizer.param_groups[0]['lr'] = base_sched[lr_ind]
	print("\n--learning rate is {}".format(base_sched[lr_ind]))
	for n in range(base_epochs[lr_ind]):
		print('\nEpoch: {}'.format(epoch))
		train(net)
		epoch += 1
	lr_ind += 1
test(net)
print("Accuracy of trained model (best checkpoint): {:.2%}".format(best_acc / 100))