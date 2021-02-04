import os
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
import argparse

######### Parser #########
parser = argparse.ArgumentParser()
parser.add_argument("--pattern", help="pattern to divide tasks with", default='sequential', choices=['sequential', 'random', 'coarse_labels'])
parser.add_argument("--num_tasks", help="number of tasks to divide in", default='20')
parser.add_argument("--download", help="download the standard datasets?", default='False')
parser.add_argument("--seed", help="seed for different splitting of the dataset in random mode", default='0')
args = parser.parse_args()

######### Setup #########
np.random.seed(int(args.seed))
if not os.path.isdir("./../../datasets"):
	os.mkdir("./../../datasets")

pattern = args.pattern
data_pattern = os.path.join("./../../datasets/cifar_splits", pattern) # path for dataloaders 
num_tasks = int(args.num_tasks)
num_classes = 100 // num_tasks
if(pattern == 'coarse_labels'):
	num_tasks, num_classes = 20, 5

# General pattern directory
if not os.path.isdir(data_pattern):
	os.mkdir(data_pattern)

# Directories for task splits
for i in range(num_tasks):
	base_path = os.path.join(data_pattern, "task_"+str(i)) # path for dataloaders 
	if not os.path.isdir(base_path):
		os.mkdir(base_path)
		os.mkdir(base_path+"/train")
		os.mkdir(base_path+"/test")
		for j in range(num_classes):
			os.mkdir(base_path+"/train/"+str(j))
			os.mkdir(base_path+"/test/"+str(j))

######### Dataloaders #########
transform_tensor = transforms.Compose([transforms.ToTensor()])
trainset = torchvision.datasets.CIFAR100(root='./../../datasets/cifar100', train=True, download=(args.download=='True'), transform=transform_tensor)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=512, shuffle=False, num_workers=4)
testset = torchvision.datasets.CIFAR100(root='./../../datasets/cifar100', train=False, download=(args.download=='True'), transform=transform_tensor)
testloader = torch.utils.data.DataLoader(testset, batch_size=512, shuffle=False, num_workers=4)

######### Dicts for task splits #########
dict_splits = {}
if(pattern == 'sequential'):
	for task_id in range(num_tasks):
		dict_splits.update({task_id * num_classes + class_id: (task_id, class_id) for class_id in np.arange(num_classes)})
elif(pattern == 'random'):
	classes_permuted = np.random.permutation(np.arange(100))
	for task_id in range(num_tasks):
		task_classes = classes_permuted[task_id * num_classes: (task_id + 1) * num_classes]
		dict_splits.update({task_class: (task_id, class_id) for class_id, task_class in enumerate(task_classes)})
elif(pattern == 'coarse_labels'):
	coarse_labels = np.array([ 4,  1, 14,  8,  0,  6,  7,  7, 18,  3,  
							   3, 14,  9, 18,  7, 11,  3,  9,  7, 11,
							   6, 11,  5, 10,  7,  6, 13, 15,  3, 15,  
							   0, 11,  1, 10, 12, 14, 16,  9, 11,  5, 
							   5, 19,  8,  8, 15, 13, 14, 17, 18, 10, 
							   16, 4, 17,  4,  2,  0, 17,  4, 18, 17, 
							   10, 3,  2, 12, 12, 16, 12,  1,  9, 19,  
							   2, 10,  0,  1, 16, 12,  9, 13, 15, 13, 
							  16, 19,  2,  4,  6, 19,  5,  5,  8, 19, 
							  18,  1,  2, 15,  6,  0, 17,  8, 14, 13])
	tasks_order = [2, 17, 10, 7, 13, 15, 16, 1, 0, 12, 8, 11, 14, 4, 3, 6, 5, 9, 19, 18]

	for task_id in range(num_tasks):
		current_task = tasks_order[task_id]
		task_classes = np.argwhere(coarse_labels==current_task)
		dict_splits.update({task_class[0]: (task_id, class_id) for class_id, task_class in enumerate(task_classes)})
else:
	raise Exception("Pattern for task splits not identified. Check again.")

# Train data
global_ind = 0
for x, y in trainloader:
	for ind in range(x.shape[0]):
		current_task, current_class = dict_splits[y[ind].item()]
		save_path = data_pattern + "/task_"+str(current_task)+"/train/"+str(current_class)+"/"+str(global_ind)+".png"  
		save_image(x[ind], save_path)
		global_ind += 1
	
# Test data
global_ind = 0
for x, y in testloader:
	for ind in range(x.shape[0]):
		current_task, current_class = dict_splits[y[ind].item()]
		save_path = data_pattern + "/task_"+str(current_task)+"/test/"+str(current_class)+"/"+str(global_ind)+".png" 
		save_image(x[ind], save_path)
		global_ind += 1