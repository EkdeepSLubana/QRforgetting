import os
import numpy as np
import argparse
from tqdm import tqdm
import scipy.io
from shutil import copyfile

######### Parser #########
parser = argparse.ArgumentParser()
parser.add_argument("--pattern", help="pattern to divide tasks with", default='sequential', choices=['sequential', 'random'])
parser.add_argument("--num_tasks", help="number of tasks to divide in", default='17')
parser.add_argument("--seed", help="seed for different splitting of the dataset in random mode", default='0')
args = parser.parse_args()

######### Setup #########
np.random.seed(int(args.seed))
if not os.path.isdir("./../../datasets/flowers_splits"):
	os.mkdir("./../../datasets/flowers_splits")

pattern = args.pattern
data_pattern = os.path.join("./../../datasets/flowers_splits", pattern) # path for dataloaders 
num_tasks = int(args.num_tasks)
num_classes = 102 // num_tasks

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
	classes_permuted = np.random.permutation(np.arange(102))
	for task_id in range(num_tasks):
		task_classes = classes_permuted[task_id * num_classes: (task_id + 1) * num_classes]
		dict_splits.update({task_class: (task_id, class_id) for class_id, task_class in enumerate(task_classes)})
else:
	raise Exception("Pattern for task splits not identified. Check again.")

labels_path = "./../../datasets/flowers/imagelabels.mat"
labels = scipy.io.loadmat(labels_path)['labels'][0] - 1
ids_path = "./../../datasets/flowers/setid.mat"
ids = scipy.io.loadmat(ids_path)
train_ids, test_ids, val_ids = ids['tstid'][0], ids['trnid'][0], ids['valid'][0]

# Train data
global_ind = 0
for im_ind in tqdm(range(1, 8190)):
	if((im_ind in train_ids) or (im_ind in val_ids)):
		im_str = str(im_ind)
		for n in range(5 - len(im_str)):
			im_str = '0' + im_str

		file_path = "./../../datasets/flowers/102flowers/image_" + im_str + ".jpg"
		label = labels[im_ind-1]

		current_task, current_class = dict_splits[label]
		save_path = data_pattern + "/task_"+str(current_task)+"/train/"+str(current_class)+"/"+str(global_ind)+".jpg"  
		copyfile(file_path, save_path)
		global_ind += 1
	
# Test data
global_ind = 0
for im_ind in tqdm(range(1, 8190)):
	if(im_ind in test_ids):
		im_str = str(im_ind)
		for n in range(5 - len(im_str)):
			im_str = '0' + im_str

		file_path = "./../../datasets/flowers/102flowers/image_" + im_str + ".jpg"
		label = labels[im_ind-1]

		current_task, current_class = dict_splits[label]
		save_path = data_pattern + "/task_"+str(current_task)+"/test/"+str(current_class)+"/"+str(global_ind)+".png" 
		copyfile(file_path, save_path)
		global_ind += 1