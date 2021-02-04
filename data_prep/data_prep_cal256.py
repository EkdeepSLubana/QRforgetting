import os
import numpy as np
import argparse
from tqdm import tqdm
import scipy.io
from shutil import copyfile

######### Parser #########
parser = argparse.ArgumentParser()
parser.add_argument("--pattern", help="pattern to divide tasks with", default='sequential', choices=['sequential', 'random'])
parser.add_argument("--num_tasks", help="number of tasks to divide in", default='32')
parser.add_argument("--seed", help="seed for different splitting of the dataset in random mode", default='0')
args = parser.parse_args()

######### Setup #########
np.random.seed(int(args.seed))
if not os.path.isdir("./../../datasets/cal256_splits"):
	os.mkdir("./../../datasets/cal256_splits")

pattern = args.pattern
data_pattern = os.path.join("./../../datasets/cal256_splits", pattern) # path for dataloaders 
num_tasks = int(args.num_tasks)
num_classes = 256 // num_tasks

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

######### Dicts for task splits #########
dict_splits = {}
if(pattern == 'sequential'):
	for task_id in range(num_tasks):
		dict_splits.update({task_id * num_classes + class_id: (task_id, class_id) for class_id in np.arange(num_classes)})
elif(pattern == 'random'):
	classes_permuted = np.random.permutation(np.arange(256))
	for task_id in range(num_tasks):
		task_classes = classes_permuted[task_id * num_classes: (task_id + 1) * num_classes]
		dict_splits.update({task_class: (task_id, class_id) for class_id, task_class in enumerate(task_classes)})
else:
	raise Exception("Pattern for task splits not identified. Check again.")

# data paths
base_path = '../datasets/256_ObjectCategories/'
categories = os.listdir(base_path)
categories.remove('257.clutter')

label = 0
for task_name in tqdm(categories):
	task_path = base_path + task_name
	task_files = os.listdir(task_path)
	current_task, current_class = dict_splits[label]
	num_train = int(0.8 * len(task_files))

	for im_ind in range(len(task_files)):
		file_path = task_path + "/" + task_files[im_ind]
		# Train data
		if(im_ind <= num_train):
			save_path = data_pattern + "/task_"+str(current_task)+"/train/"+str(current_class)+"/"+task_files[im_ind]  
			copyfile(file_path, save_path)
		
		# Test data
		if(im_ind > num_train):
			save_path = data_pattern + "/task_"+str(current_task)+"/test/"+str(current_class)+"/"+task_files[im_ind] 
			copyfile(file_path, save_path)

	label += 1