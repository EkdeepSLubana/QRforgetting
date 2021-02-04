import torch
import torchvision
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
from models import *
from utils import *
from cka import cka
import copy
import numpy as np
import pickle as pkl
import os
import argparse

######### Parser #########
parser = argparse.ArgumentParser()
parser.add_argument("--seed", help="setup the random seed", default='0')
parser.add_argument("--dataset", help="dataset used", default='cifar', choices=['cifar', 'flowers', 'cal256'])
parser.add_argument("--split_pattern", help="pattern to divide tasks with", default='sequential', choices=['sequential', 'random', 'coarse_labels'])
parser.add_argument("--batch_size", help="batch size for dataloaders", default='10')
parser.add_argument("--train_type", help="train nets with which method", choices=['plain', 'online_qreg', 'online_emr', 'emr', 'qreg'])
parser.add_argument("--grid_search", help="grid search for hyperparameters", default='False', choices=['True', 'False'])
parser.add_argument("--importance_defin", help="importance definition to be used", default='None', choices=['ewc', 'mas', 'SI', 'RWalk', 'vanilla', 'rand'])
parser.add_argument("--importance_momentum", help="momentum for importance update", default='0.8')
parser.add_argument("--use_flip", help="flip minibatch for online methods", default='False')
parser.add_argument("--lr_config", help="learning rate", default='0')
parser.add_argument("--reg_coeff", help="regularization coefficient", default='0')
parser.add_argument("--print_cka", help="print cka similarities after every task?", default='False', choices=['True', 'False'])
parser.add_argument("--save_results", help="save results for later analysis", default='False', choices=['True', 'False'])
parser.add_argument("--use_pretrained", help="use pretrained CIFAR-10 model", default='True')
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

if not os.path.isdir('cl_models'):
	os.mkdir('cl_models')
if not os.path.isdir('results'):
	os.mkdir('results')
if not os.path.isdir('grid_search'):
	os.mkdir('grid_search')
cl_root = 'cl_models/'

######### Setup the framework from argparser variables #########
use_pretrained = (args.use_pretrained=='True')
pretrained_path = 'pretrained/vanilla_cnn_seed_0.pth'
dataset = args.dataset + '_splits'
if not os.path.isdir('results/'+dataset):
	os.mkdir('results/'+dataset)
if not os.path.isdir('grid_search/'+dataset):
	os.mkdir('grid_search/'+dataset)
split_pattern = args.split_pattern
num_tasks = len(os.listdir('./../datasets/'+dataset+'/'+split_pattern))
total_classes = 100 if (args.dataset == 'cifar') else 102 if (args.dataset == 'flowers') else 256
num_classes = total_classes // num_tasks
batch_size = int(args.batch_size)
cl_epochs = 1 # This can be changed if more than one epoch of training is desired
if(split_pattern == 'coarse_labels'):
	if(args.dataset == 'cifar'):
		pass
	else:
		raise Exception("Coarse labels are only valid for CIFAR-100.")
train_type = args.train_type
grid_search = (args.grid_search == 'True')
importance_defin = args.importance_defin # how to define importance
importance_momentum = float(args.importance_momentum) 
importance_basis = 'loss' if ((importance_defin == 'SI') or (importance_defin == 'ewc') or (importance_defin == 'RWalk')) else 'logits'
use_flip = (args.use_flip == 'True')
print_cka = (args.print_cka == 'True')
save_results = (args.save_results == 'True')

if(grid_search):
	start_task, end_task = 0, 3
	lr_options = [0.03, 0.01, 0.003, 0.001]
	reg_options = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
	print_cka = False
else:
	start_task, end_task = 3, num_tasks
	lr_options = [float(args.lr_config)]
	reg_options = [float(args.reg_coeff)]
num_tasks = end_task - start_task

hyperparams = []
if(train_type=='emr' or train_type=='online_emr'):
	for l in lr_options:
		hyperparams.append((l, 0))
else:
	for l in lr_options:
		for r in reg_options:
			hyperparams.append((l, r))

print("\n------------------ Setup For Training ------------------\n")
print("Dataset:", args.dataset)
print("Data split pattern:", split_pattern)
print("Training type:", train_type)
print("Importance definition:", importance_defin)
print("Importance momentum:", importance_momentum)
print("Importance Basis:", importance_basis)
print("Flip minibatch:", use_flip)
print("Batch size:", batch_size)
print("Learning rate:", lr_options)
if(train_type == 'emr' or train_type == 'online_emr'):
	pass
else:
	print("Regularization constant:", reg_options)
print("Optimizer:", SGD)
print("Number of tasks:", end_task - start_task)
print("Number of classes per task:", num_classes)
print("Pretrained model:", use_pretrained)
print("Save results:", save_results)

######### Loss #########
criterion = nn.CrossEntropyLoss()

######### Dataloaders #########
def get_dataloader(task_id, split_pattern):
	transform_train = transforms.Compose(
		[transforms.Resize((32,32)),
		 # transforms.RandomHorizontalFlip(),
		 transforms.ToTensor(),
		 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
		 ])
	transform_test = transforms.Compose(
		[transforms.Resize((32,32)),
		 transforms.ToTensor(),
		 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
		 ])

	trainset = datasets.ImageFolder(root='./../datasets/'+dataset+'/'+split_pattern+'/task_'+str(task_id)+'/train/', transform=transform_train)
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
	testset = datasets.ImageFolder(root='./../datasets/'+dataset+'/'+split_pattern+'/task_'+str(task_id)+'/test/', transform=transform_test)
	testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)
	return trainloader, testloader

######### Function to initialize a new model #########
def create_model(num_classes=10):
	net = torch.nn.DataParallel(Vanilla_cnn(num_classes=num_classes))
	return net

######### Functions needed for calculating importance #########
def clone_model_params(model):
	params = []
	for param in model.parameters():
		if not param.requires_grad:
			continue
		params.append(param.detach().clone())
	return params

def diff_params(net_curr, net_prev):
	diff = []
	for (param1, param2) in zip(net_curr.parameters(), net_prev.parameters()):
		if not param1.requires_grad:
			continue
		diff.append((param1 - param2).detach().clone())
	return diff

def clone_model_grads(model):
	grads = []
	for param in model.parameters():
		if not param.requires_grad:
			continue
		grads.append(0. if param.grad is None else param.grad.detach().clone() + 0.)
	return grads

def update_importance_vanilla_or_rand(net, importance_defin, prev_imp=None, tid=0):
	l_params = clone_model_params(net)
	if(importance_defin=='vanilla'):
		nlist = [(1/2) * torch.ones_like(l_params[ind]) for ind in range(len(l_params)-2)]
	elif(importance_defin=='rand'):
		nlist = [torch.rand_like(l_params[ind]) for ind in range(len(l_params)-2)]

	if(tid == start_task):
		new_imp = nlist
	else:
		new_imp = [(n_imp + p_imp) / 2 for n_imp, p_imp in zip(nlist, prev_imp)]
	return new_imp

######### Training functions #########
### Plain fine-tuning ###
def plain_train(net_curr, dataloader, task_id=0):
	net_curr.train()
	train_loss = 0
	correct = 0
	total = 0
	for batch_idx, (inputs, targets) in enumerate(dataloader):
		inputs, targets = inputs.to(device), targets.to(device)

		### SGD step on current minibatch 
		optimizer.zero_grad()
		outputs = net_curr(inputs)
		loss = criterion(outputs, targets)
		loss.backward()
		optimizer.step()

		train_loss += loss.item()
		_, predicted = outputs.max(1)
		total += targets.size(0)
		correct += predicted.eq(targets).sum().item()
		progress_bar(batch_idx, len(dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
			% (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

		### SGD step on flipped minibatch 
		if(use_flip):
			inputs = torch.flip(inputs, [3]).to(device)
			optimizer.zero_grad()
			outputs = net_curr(inputs)	
			loss = criterion(outputs, targets)		
			loss.backward()
			optimizer.step()

### Quadratic regularization (Use this implementation for online methods: EWC, MAS, SI, RWalk) ###
def online_qreg_train(net_curr, net_prev, dataloader, epoch, importance_defin='ewc', task_id=0, reg_coeff=1):
	global original_params, importance_list, wk_running, fisher_running
	net_curr.train()
	train_loss = 0
	correct = 0
	total = 0
	start_params = clone_model_params(net_curr)

	for batch_idx, (inputs, targets) in enumerate(dataloader):
		inputs, targets = inputs.to(device), targets.to(device)

		### SGD step on current minibatch 
		optimizer.zero_grad()
		outputs = net_curr(inputs)
		if(task_id == start_task):
			loss = criterion(outputs, targets)
		else:
			forgetting_loss = 0
			for ind, l in enumerate([0, 2, 5, 7, 10, 12]):
				forgetting_loss += (importance_list[2*ind] * ((net_curr.module.features[l].weight - net_prev.module.features[l].weight.detach())).pow(2)).sum()
				forgetting_loss += (importance_list[2*ind+1] * ((net_curr.module.features[l].bias - net_prev.module.features[l].bias.detach())).pow(2)).sum()
			loss = criterion(outputs, targets) + reg_coeff * forgetting_loss
		loss.backward()
		optimizer.step()

		train_loss += loss.item()
		_, predicted = outputs.max(1)
		total += targets.size(0)
		correct += predicted.eq(targets).sum().item()
		progress_bar(batch_idx, len(dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
			% (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

		if(importance_basis == 'loss'):
			next_params, next_grads = clone_model_params(net_curr), clone_model_grads(net_curr)

		elif(importance_basis == 'logits'):
			outputs = net_curr.module.features(inputs.to(device))
			loss = outputs.norm().pow(2)
			loss.backward()
			next_params, next_grads = clone_model_params(net_curr), clone_model_grads(net_curr)

		if(importance_defin == 'ewc'):
			wk_running = [(importance_momentum) * (next_g).pow(2) + (1-importance_momentum) * wk_old for (next_g, wk_old) in zip(next_grads, wk_running)]
		elif(importance_defin == 'mas'):
			wk_running = [(importance_momentum) * (next_g).abs() + (1-importance_momentum) * wk_old for (next_g, wk_old) in zip(next_grads, wk_running)]
		elif(importance_defin == 'SI'):
			wk_running = [(importance_momentum) * (next_g * (next_p - start_p)).abs() + (1-importance_momentum) * wk_old for (next_g, next_p, start_p, wk_old) in zip(next_grads, next_params, start_params, wk_running)]
		elif(importance_defin == 'RWalk'):
			wk_running = [importance_momentum * (next_g * (next_p - start_p)).abs() + (1-importance_momentum) * wk_old for (next_g, next_p, start_p, wk_old) in zip(next_grads, next_params, start_params, wk_running)]
			fisher_running = [importance_momentum * (next_g * next_p).pow(2) + (1-importance_momentum) * f_old for (next_g, next_p, f_old) in zip(next_grads, next_params, fisher_running)]

		### SGD step on flipped minibatch 
		if(use_flip):
			inputs = torch.flip(inputs, [3]).to(device)
			optimizer.zero_grad()
			outputs = net_curr(inputs)	
			if(task_id == start_task):
				loss = criterion(outputs, targets)
			else:
				forgetting_loss = 0
				for ind, l in enumerate([0, 2, 5, 7, 10, 12]):
					forgetting_loss += (importance_list[2*ind] * ((net_curr.module.features[l].weight - net_prev.module.features[l].weight.detach())).pow(2)).sum()
					forgetting_loss += (importance_list[2*ind+1] * ((net_curr.module.features[l].bias - net_prev.module.features[l].bias.detach())).pow(2)).sum()
				loss = criterion(outputs, targets) + reg_coeff * forgetting_loss
			loss.backward()
			optimizer.step()

			if(importance_basis == 'loss'):
				next_params, next_grads = clone_model_params(net_curr), clone_model_grads(net_curr)

			elif(importance_basis == 'logits'):
				outputs = net_curr.module.features(inputs.to(device))
				loss = outputs.norm().pow(2)
				loss.backward()
				next_params, next_grads = clone_model_params(net_curr), clone_model_grads(net_curr)

			if(importance_defin == 'ewc'):
				wk_running = [(importance_momentum) * (next_g).pow(2) + (1-importance_momentum) * wk_old for (next_g, wk_old) in zip(next_grads, wk_running)]
			elif(importance_defin == 'mas'):
				wk_running = [(importance_momentum) * (next_g).abs() + (1-importance_momentum) * wk_old for (next_g, wk_old) in zip(next_grads, wk_running)]
			elif(importance_defin == 'SI'):
				wk_running = [(importance_momentum) * (next_g * (next_p - start_p)).abs() + (1-importance_momentum) * wk_old for (next_g, next_p, start_p, wk_old) in zip(next_grads, next_params, start_params, wk_running)]
			elif(importance_defin == 'RWalk'):
				wk_running = [importance_momentum * (next_g * (next_p - start_p)).abs() + (1-importance_momentum) * wk_old for (next_g, next_p, start_p, wk_old) in zip(next_grads, next_params, start_params, wk_running)]
				fisher_running = [importance_momentum * (next_g * next_p).pow(2) + (1-importance_momentum) * f_old for (next_g, next_p, f_old) in zip(next_grads, next_params, fisher_running)]

### Quadratic regularization (Use this implementation for the proposed, offline methods: Vanilla, Random) ###
def qreg_train(net_curr, net_prev, dataloader, epoch, task_id=0, reg_coeff=1):
	global original_params, importance_list
	net_curr.train()
	train_loss = 0
	correct = 0
	total = 0
	for batch_idx, (inputs, targets) in enumerate(dataloader):
		inputs, targets = inputs.to(device), targets.to(device)

		### SGD step on current minibatch 
		optimizer.zero_grad()
		outputs = net_curr(inputs)
		if(task_id == start_task):
			loss = criterion(outputs, targets)
		else:
			forgetting_loss = 0
			for ind, l in enumerate([0, 2, 5, 7, 10, 12]):
				forgetting_loss += (importance_list[2*ind] * ((net_curr.module.features[l].weight - net_prev.module.features[l].weight.detach())).pow(2)).sum()
				forgetting_loss += (importance_list[2*ind+1] * ((net_curr.module.features[l].bias - net_prev.module.features[l].bias.detach())).pow(2)).sum()
			loss = criterion(outputs, targets) + reg_coeff * forgetting_loss
		loss.backward()
		optimizer.step()

		train_loss += loss.item()
		_, predicted = outputs.max(1)
		total += targets.size(0)
		correct += predicted.eq(targets).sum().item()
		progress_bar(batch_idx, len(dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
			% (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

		### SGD step on flipped minibatch 
		if(use_flip):
			inputs = torch.flip(inputs, [3]).to(device)
			optimizer.zero_grad()
			outputs = net_curr(inputs)	
			if(task_id == start_task):
				loss = criterion(outputs, targets)
			else:
				forgetting_loss = 0
				for ind, l in enumerate([0, 2, 5, 7, 10, 12]):
					forgetting_loss += (importance_list[2*ind] * ((net_curr.module.features[l].weight - net_prev.module.features[l].weight.detach())).pow(2)).sum()
					forgetting_loss += (importance_list[2*ind+1] * ((net_curr.module.features[l].bias - net_prev.module.features[l].bias.detach())).pow(2)).sum()
				loss = criterion(outputs, targets) + reg_coeff * forgetting_loss
			loss.backward()
			optimizer.step()

### Explicit movement regularization (Use this implementation for online methods: EWC, MAS, SI, RWalk) ###
def online_emr_train(net_curr, net_prev, dataloader, epoch, importance_defin='SI', prev_imp=None, task_id=0, use_flip=True):
	global original_params, importance_list, wk_running, fisher_running
	net_curr.train()
	train_loss = 0
	correct = 0
	total = 0
	start_params = clone_model_params(net_curr)

	for batch_idx, (inputs, targets) in enumerate(dataloader):
		inputs, targets = inputs.to(device), targets.to(device)

		### SGD step on current minibatch 
		optimizer.zero_grad()
		outputs = net_curr(inputs)
		loss = criterion(outputs, targets)		
		loss.backward()
		optimizer.step()

		train_loss += loss.item()
		_, predicted = outputs.max(1)
		total += targets.size(0)
		correct += predicted.eq(targets).sum().item()
		progress_bar(batch_idx, len(dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
			% (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

		if(importance_basis == 'loss'):
			next_params, next_grads = clone_model_params(net_curr), clone_model_grads(net_curr)

		elif(importance_basis == 'logits'):
			outputs = net_curr.module.features(inputs.to(device))
			loss = outputs.norm().pow(2)
			loss.backward()
			next_params, next_grads = clone_model_params(net_curr), clone_model_grads(net_curr)

		if(importance_defin == 'ewc'):
			wk_running = [(importance_momentum) * (next_g).pow(2) + (1-importance_momentum) * wk_old for (next_g, wk_old) in zip(next_grads, wk_running)]
		elif(importance_defin == 'mas'):
			wk_running = [(importance_momentum) * (next_g).abs() + (1-importance_momentum) * wk_old for (next_g, wk_old) in zip(next_grads, wk_running)]
		elif(importance_defin == 'SI'):
			wk_running = [(importance_momentum) * (next_g * (next_p - start_p)).abs() + (1-importance_momentum) * wk_old for (next_g, next_p, start_p, wk_old) in zip(next_grads, next_params, start_params, wk_running)]
		elif(importance_defin == 'RWalk'):
			wk_running = [importance_momentum * (next_g * (next_p - start_p)).abs() + (1-importance_momentum) * wk_old for (next_g, next_p, start_p, wk_old) in zip(next_grads, next_params, start_params, wk_running)]
			fisher_running = [importance_momentum * (next_g * next_p).pow(2) + (1-importance_momentum) * f_old for (next_g, next_p, f_old) in zip(next_grads, next_params, fisher_running)]

		### Weighted average step
		if(task_id > start_task and (batch_idx%1==0)):
			if(importance_defin == 'SI'):
				diff_list = diff_params(net_curr, net_prev)
				new_imp = [(wk / (d.pow(2) + 1e-10)) for (wk, d) in zip(wk_running, diff_list)] 
			elif(importance_defin == 'RWalk'):
				diff_list = diff_params(net_curr, net_prev)
				new_imp = [(f + ((wk) / (((f) * d.pow(2)) + 1e-10))) for (wk, f, d) in zip(wk_running, fisher_running, diff_list)]
			else:
				new_imp = [wk for wk in wk_running]

			lind = 0
			for (mod_new, mod) in zip(net_curr.modules(), net_prev.modules()):
				if(isinstance(mod_new, nn.Conv2d)):
					alp = ((new_imp[lind].pow(1/2)) / (new_imp[lind].pow(1/2) + prev_imp[lind].pow(1/2) + 1e-20))
					mod_new.weight.data = alp * mod_new.weight.data.detach().clone() + (1 - alp) * mod.weight.data.detach().clone()
					lind += 1
					alp = ((new_imp[lind].pow(1/2)) / (new_imp[lind].pow(1/2) + prev_imp[lind].pow(1/2) + 1e-20))
					mod_new.bias.data = alp * mod_new.bias.data.detach().clone() + (1 - alp) * mod.bias.data.detach().clone()
					lind += 1

		### SGD step on flipped minibatch 
		if(use_flip):
			inputs = torch.flip(inputs, [3]).to(device)
			optimizer.zero_grad()
			outputs = net_curr(inputs)
			loss = criterion(outputs, targets)		
			loss.backward()
			optimizer.step()

			if(importance_basis == 'loss'):
				next_params, next_grads = clone_model_params(net_curr), clone_model_grads(net_curr)

			elif(importance_basis == 'logits'):
				outputs = net_curr.module.features(inputs.to(device))
				loss = outputs.norm().pow(2)
				loss.backward()
				next_params, next_grads = clone_model_params(net_curr), clone_model_grads(net_curr)

			if(importance_defin == 'ewc'):
				wk_running = [(importance_momentum) * (next_g).pow(2) + (1-importance_momentum) * wk_old for (next_g, wk_old) in zip(next_grads, wk_running)]
			elif(importance_defin == 'mas'):
				wk_running = [(importance_momentum) * (next_g).abs() + (1-importance_momentum) * wk_old for (next_g, wk_old) in zip(next_grads, wk_running)]
			elif(importance_defin == 'SI'):
				wk_running = [(importance_momentum) * (next_g * (next_p - start_p)).abs() + (1-importance_momentum) * wk_old for (next_g, next_p, start_p, wk_old) in zip(next_grads, next_params, start_params, wk_running)]
			elif(importance_defin == 'RWalk'):
				wk_running = [importance_momentum * (next_g * (next_p - start_p)).abs() + (1-importance_momentum) * wk_old for (next_g, next_p, start_p, wk_old) in zip(next_grads, next_params, start_params, wk_running)]
				fisher_running = [importance_momentum * (next_g * next_p).pow(2) + (1-importance_momentum) * f_old for (next_g, next_p, f_old) in zip(next_grads, next_params, fisher_running)]

			### Weighted average step
			if(task_id > start_task):
				if(importance_defin == 'SI'):
					diff_list = diff_params(net_curr, net_prev)
					new_imp = [(wk / (d.pow(2) + 1e-10)) for (wk, d) in zip(wk_running, diff_list)] 
				elif(importance_defin == 'RWalk'):
					diff_list = diff_params(net_curr, net_prev)
					new_imp = [(f + ((wk) / (((f) * d.pow(2)) + 1e-10))) for (wk, f, d) in zip(wk_running, fisher_running, diff_list)]
				else:
					new_imp = [wk for wk in wk_running]

				lind = 0
				for (mod_new, mod) in zip(net_curr.modules(), net_prev.modules()):
					if(isinstance(mod_new, nn.Conv2d)):
						alp = ((new_imp[lind].pow(1/2)) / (new_imp[lind].pow(1/2) + prev_imp[lind].pow(1/2) + 1e-20))
						mod_new.weight.data = alp * mod_new.weight.data.detach().clone() + (1 - alp) * mod.weight.data.detach().clone()
						lind += 1
						alp = ((new_imp[lind].pow(1/2)) / (new_imp[lind].pow(1/2) + prev_imp[lind].pow(1/2) + 1e-20))
						mod_new.bias.data = alp * mod_new.bias.data.detach().clone() + (1 - alp) * mod.bias.data.detach().clone()
						lind += 1

### Explicit movement regularization (Use this implementation for the proposed, offline methods: Vanilla, Random) ###
def emr_train(net_curr, net_prev, dataloader, epoch, importance_defin, prev_imp, task_id=0, use_flip=True):
	net_curr.train()
	train_loss = 0
	correct = 0
	total = 0

	for batch_idx, (inputs, targets) in enumerate(dataloader):
		inputs, targets = inputs.to(device), targets.to(device)

		### SGD step on current minibatch 
		optimizer.zero_grad()
		outputs = net_curr(inputs)
		loss = criterion(outputs, targets)		
		loss.backward()
		optimizer.step()

		train_loss += loss.item()
		_, predicted = outputs.max(1)
		total += targets.size(0)
		correct += predicted.eq(targets).sum().item()

		progress_bar(batch_idx, len(dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
			% (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
		
		### Weighted average step 
		if(task_id > start_task):
			# Calculate new importances
			new_imp = cal_importance(net=net_curr, importance_defin=importance_defin, dataloader=dataloader, n_classes=num_classes)

			lind = 0
			for (mod_new, mod) in zip(net_curr.modules(), net_prev.modules()):
				if(isinstance(mod_new, nn.Conv2d)):
					alp = ((new_imp[lind].pow(1/2)) / (new_imp[lind].pow(1/2) + prev_imp[lind].pow(1/2) + 1e-20))
					mod_new.weight.data = alp * mod_new.weight.data.detach().clone() + (1 - alp) * mod.weight.data.detach().clone()
					lind += 1
					alp = ((new_imp[lind].pow(1/2)) / (new_imp[lind].pow(1/2) + prev_imp[lind].pow(1/2) + 1e-20))
					mod_new.bias.data = alp * mod_new.bias.data.detach().clone() + (1 - alp) * mod.bias.data.detach().clone()
					lind += 1

		### SGD step on flipped minibatch 
		if(use_flip):
			inputs = torch.flip(inputs, [3]).to(device)
			optimizer.zero_grad()
			outputs = net_curr(inputs)
			loss = criterion(outputs, targets)		
			loss.backward()
			optimizer.step()

			### Weighted average step 
			if(task_id > start_task):
				# Calculate new importances
				new_imp = cal_importance(net=net_curr, importance_defin=importance_defin, dataloader=dataloader, n_classes=num_classes)

				lind = 0
				for (mod_new, mod) in zip(net_curr.modules(), net_prev.modules()):
					if(isinstance(mod_new, nn.Conv2d)):
						alp = ((new_imp[lind].pow(1/2)) / (new_imp[lind].pow(1/2) + prev_imp[lind].pow(1/2) + 1e-20))
						mod_new.weight.data = alp * mod_new.weight.data.detach().clone() + (1 - alp) * mod.weight.data.detach().clone()
						lind += 1
						alp = ((new_imp[lind].pow(1/2)) / (new_imp[lind].pow(1/2) + prev_imp[lind].pow(1/2) + 1e-20))
						mod_new.bias.data = alp * mod_new.bias.data.detach().clone() + (1 - alp) * mod.bias.data.detach().clone()
						lind += 1

######### Evaluation functions #########
### Calculate test accuracy (used at the end of every task to compute original task accuracy) ###
def eval(net, testloader, task_id=0, save=False, lr_method=0, reg_constant=0):
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
			if(not save):
				progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
				% (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

	# Save best checkpoint
	if(save):
		acc = 100.*correct/total
		print('\nSaving...', end="")
		state = {'net': net.state_dict()}
		torch.save(state, cl_root+'{data_name}_train_{ttype}_defin_{defin}_lr_{lr}_reg_{reg}_tasks_{ntasks}_taskid_{tid}_seed_{sid}.pth'.format(data_name=dataset, ttype=train_type, defin=importance_defin, lr=lr_method, reg=reg_constant, ntasks=num_tasks, tid=task_id, sid=int(args.seed)))
		return acc

### Calculate accuracy on a given dataloader (will be used for calculating accuracies on previously learned tasks) ###
def cal_acc(net, use_loader):
	net.eval()
	test_loss = 0
	correct = 0
	total = 0
	with torch.no_grad():
		for batch_idx, (inputs, targets) in enumerate(use_loader):
			inputs, targets = inputs.to(device), targets.to(device)
			outputs = net(inputs)
			loss = criterion(outputs, targets)
			test_loss += loss.item()
			_, predicted = outputs.max(1)
			total += targets.size(0)
			correct += predicted.eq(targets).sum().item()
	return 100.*(correct / total)

### Rewinding functions (useful in function ``update_results'') ###
def rewind_conv(net, net_base):
	for mod, mod_base in zip(net.modules(), net_base.modules()):
		if(isinstance(mod, nn.Conv2d)):
			mod.weight.data = mod_base.weight.data.detach().clone()
			mod.bias.data = mod_base.bias.data.detach().clone()
	return net

def net_rewinding(net_features, net_classifier):
	net_rewind = create_model(num_classes=num_classes)
	for (mod_rewind, mod_features) in zip(net_rewind.modules(), net_features.modules()):
		if(isinstance(mod_rewind, nn.Conv2d)):
			mod_rewind.weight.data = mod_features.weight.data.clone()
			mod_rewind.bias.data = mod_features.bias.data.clone()

	for (mod_rewind, mod_classifier) in zip(net_rewind.modules(), net_classifier.modules()):
		if(isinstance(mod_rewind, nn.Linear)):
			mod_rewind.weight.data = mod_classifier.weight.data.clone()
			mod_rewind.bias.data = mod_classifier.bias.data.clone()
	return net_rewind

### Function to accumulate training progress during the continual learning process ### 
def update_results(net_curr, upper_id, split_pattern, task_final=False, lr_method=0, reg_constant=0):
	av_tracc, av_teacc = 0, 0
	print("\n")
	for tid in range(start_task, upper_id+1):
		### net_classifier ###
		net_classifier = create_model(num_classes=num_classes)
		net_path = cl_root+'{data_name}_train_{ttype}_defin_{defin}_lr_{lr}_reg_{reg}_tasks_{ntasks}_taskid_{tid}_seed_{sid}.pth'.format(data_name=dataset, ttype=train_type, defin=importance_defin, lr=lr_method, reg=reg_constant, ntasks=num_tasks, tid=tid, sid=int(args.seed))
		net_dict = torch.load(net_path)
		net_classifier.load_state_dict(net_dict['net'])
	
		### Dataloaders ###
		trloader, teloader = get_dataloader(tid, split_pattern=split_pattern)

		### rewind net ###
		net_r = net_rewinding(net_features=net_curr, net_classifier=net_classifier)

		tracc = cal_acc(net_r.eval(), trloader)
		teacc = cal_acc(net_r.eval(), teloader)

		if(teacc > stat[tid]['max_acc']):
			stat[tid]['max_acc'] = teacc
		if(task_final):
			stat[tid]['final_acc'] = teacc

		av_tracc += tracc
		av_teacc += teacc

		print("Task "+str(tid)+" results:", end=" ")
		print("Train: {:.2f}".format(tracc), end="; ")
		print("  Test: {:.2f}".format(teacc))		
		
	print("\nAverage Train Accuracy: {:.2f}".format(av_tracc / (upper_id+1 - start_task)))
	print("Average Test Accuracy: {:.2f}".format(av_teacc / (upper_id+1 - start_task)))

	return av_tracc / (upper_id+1 - start_task), av_teacc / (upper_id+1 - start_task)

### Layer-wise CKA estimator ### 
def layerwise_cka(net, net_base, task_id=1):
	if(print_cka):
		print("\n------------------ CKA similarity between Task 0 and Task {:d} ------------------".format(task_id))
	orig_loader, _ = get_dataloader(0, split_pattern=split_pattern)
	for n, (X1, _) in enumerate(orig_loader):
		if(n==0):
			X = X1.clone()
		else:
			torch.cat((X, X1), dim=0)
		if(n * batch_size > 500):
			break
	with torch.no_grad():
		lind = 0
		for mod, mod_base in zip(net.module.features, net_base.module.features):
			lind += 1
			if(isinstance(mod, nn.Conv2d)):
				# Current model's gram matrix w.r.t ReLU features
				f_curr = (net.module.features[0:lind+1](X.to(device))).reshape(batch_size, -1)
				gram_curr = torch.matmul(f_curr, f_curr.T).cpu().numpy()
				# Original model's gram matrix w.r.t ReLU features
				f_orig = (net_base.module.features[0:lind+1](X.to(device))).reshape(batch_size, -1)
				gram_orig = torch.matmul(f_orig, f_orig.T).cpu().numpy()
				# CKA
				cka_val = cka(gram_curr, gram_orig, debiased=True)
				stat[task_id]['cka'].append(cka_val)
				if(print_cka):
					print("Layer {:d}: {:.3f}".format(lind-1, cka_val))


######### Continual Learning setup is established here #########
try_config = {c:True for c in hyperparams}

for (lr_method, reg_constant) in hyperparams: 
	if(not try_config[(lr_method, reg_constant)]):
		continue

	### Initialize model and importance list ###
	net_prev = create_model(num_classes=num_classes)
	net_curr = create_model(num_classes=num_classes)
	importance_list = []
	if(train_type == 'online_qreg' or train_type == 'online_emr'):
		temp = clone_model_params(net_curr)
		wk_running = [torch.tensor(0).to(device) for i in temp]
		fisher_running = [torch.tensor(0).to(device) for i in temp]
		importance_list = [torch.tensor(0).to(device) for i in temp]
		del temp
	stat = {task_id:{'orig_acc': 0, 'final_acc': 0, 'max_acc': 0, 'cka': []} for task_id in range(start_task, end_task)}

	### Use pretrained model ###
	if(use_pretrained):
		print("\n------------------ Loading pretrained model ------------------\n")
		net_pretrained = create_model(num_classes=10)
		net_dict = torch.load(pretrained_path)
		net_pretrained.load_state_dict(net_dict['net'])
		net_prev = rewind_conv(net_prev, net_pretrained)
		net_curr = rewind_conv(net_curr, net_pretrained)
		del net_dict, net_pretrained

	######### Continual Learning process begins here #########
	for task_id in range(start_task, end_task):
		print("\n------------------ Task ID: {tid} ------------------\n".format(tid=task_id))

		### Dataloaders ###
		curr_trainloader, curr_testloader = get_dataloader(task_id, split_pattern=split_pattern)

		### Optimizer ###
		if((train_type == 'plain') or (task_id == start_task)):
			optimizer = optim.SGD(net_curr.parameters(), lr=lr_method, momentum=0.9, weight_decay=1e-4)
		else:
			optimizer = optim.SGD(net_curr.parameters(), lr=lr_method, momentum=0.9, weight_decay=0)

		### Train ###
		epoch = 0

		if((train_type == 'plain') or (train_type == 'emr') or (train_type == 'online_emr')):
			print("\n--Training at {lr} learning rate for {n} epochs".format(lr=lr_method, n=cl_epochs))
		else:
			print("\n--Training at {lr} learning rate and {reg} regularization constant for {n} epochs".format(lr=lr_method, reg=reg_constant, n=cl_epochs))

		for n in range(cl_epochs):
			print('\nEpoch: {}'.format(epoch))

			# Train
			if(train_type == 'plain'):
				plain_train(net_curr, dataloader=curr_trainloader, task_id=task_id)

			elif(train_type == 'qreg'):
				qreg_train(net_curr=net_curr, net_prev=net_prev, dataloader=curr_trainloader, epoch=epoch, task_id=task_id, reg_coeff=reg_constant)

			elif(train_type == 'online_qreg'):
				online_qreg_train(net_curr=net_curr, net_prev=net_prev, dataloader=curr_trainloader, epoch=epoch, importance_defin=importance_defin, 
									task_id=task_id, reg_coeff=reg_constant)

			elif(train_type == 'emr'):
				emr_train(net_curr=net_curr, net_prev=net_prev, dataloader=curr_trainloader, epoch=epoch, importance_defin=importance_defin, 
									prev_imp=importance_list, task_id=task_id, use_flip=use_flip)

			elif(train_type == 'online_emr'):
				online_emr_train(net_curr=net_curr, net_prev=net_prev, dataloader=curr_trainloader, epoch=epoch, importance_defin=importance_defin, 
									prev_imp=importance_list, task_id=task_id, use_flip=use_flip)

			epoch += 1

		task_acc = eval(net_curr, testloader=curr_testloader, task_id=task_id, save=True, lr_method=lr_method, reg_constant=reg_constant) # save current task's model

		### Update importance ###
		if(train_type == 'plain'):
			pass
		elif(train_type == 'online_emr' or train_type == 'online_qreg'):
			print(" Updating importance... ", end="")
			if(importance_defin == 'SI'):
				diff_list = diff_params(net_curr, net_prev)
				temp_list = [(wk / (d.pow(2) + 1e-10)) for (wk, d) in zip(wk_running, diff_list)] 
				del diff_list
			elif(importance_defin == 'RWalk'):
				diff_list = diff_params(net_curr, net_prev)
				temp_list = [(f + ((wk) / ((f * d.pow(2)) + 1e-10))) for (wk, f, d) in zip(wk_running, fisher_running, diff_list)]
				del diff_list
			else:
				temp_list = [wk for wk in wk_running]

			importance_list = ([(s + t) / 2 for (s, t) in zip(importance_list, temp_list)])
			wk_running = [torch.tensor(0).to(device) for i in wk_running]
			del temp_list

		else:
			print(" Updating importance... ", end="")
			importance_list = update_importance_vanilla_or_rand(net=net_curr, importance_defin=importance_defin, prev_imp=importance_list if task_id > start_task else None, tid=task_id)
		print("Done.")

		### Rewind and update accuracy numbers ###
		stat[task_id]['orig_acc'] = task_acc
		if(save_results and (not grid_search)):
			print("\n------------------ Progress Check ------------------")
			av_train_acc, av_test_acc = update_results(net_curr=net_curr, upper_id=task_id, split_pattern=split_pattern, task_final=(task_id==end_task-1), lr_method=lr_method, reg_constant=reg_constant)

		if(task_id > start_task and (not grid_search)):
			layerwise_cka(net_curr, net_orig, task_id=task_id)
		else:
			net_orig = copy.deepcopy(net_curr)

		### Cache current network ###
		net_prev = copy.deepcopy(net_curr)

	######### Print and save final results #########
	print("\n------------------ Final Stats ------------------")
	if(not save_results or grid_search):
		av_train_acc, av_test_acc = update_results(net_curr=net_curr, upper_id=task_id, split_pattern=split_pattern, task_final=(task_id==end_task-1), lr_method=lr_method, reg_constant=reg_constant)

	av_forgetting = 0
	for task_id in range(start_task, end_task):
		print("Task "+str(task_id)+":")
		print("\tOriginal Accuracy: {:.2f}".format(stat[task_id]['orig_acc']))
		print("\tFinal Accuracy: {:.2f}".format(stat[task_id]['final_acc']))
		if(save_results):
			print("\tForgetting: {:.2f}".format(stat[task_id]['max_acc'] - stat[task_id]['final_acc']))
			av_forgetting += stat[task_id]['max_acc'] - stat[task_id]['final_acc']
		print("\tCKA similarity:")
		for lind, sim in enumerate(stat[task_id]['cka']):
			print("\t\tConv {:d}: {:.3f}".format(lind, sim))

	print("\n------------------ Average Results ------------------\n")
	print("Train Accuracy: {:.2f}".format(av_train_acc))
	stat['Av_train'] = av_train_acc
	print("Test Accuracy: {:.2f}".format(av_test_acc))
	stat['Av_test'] = av_test_acc
	if(save_results):
		print("Forgetting: {:.2f} \n".format(av_forgetting / num_tasks))
		stat['Av_forgetting'] = av_forgetting / num_tasks
	stat['lr'] = lr_method
	stat['reg'] = reg_constant

	if(av_test_acc < ((100 / num_classes) + 2)):
		if(train_type=='emr' or train_type=='online_emr'):
			pass
		else:
			for r_next in reg_options:
				if(r_next > reg_constant):
					try_config[(lr_method, r_next)] = False

	if(save_results):
		results_loc = './results/' + dataset + '/'
		results_loc += train_type + '_'
		if(use_flip):
			results_loc += 'flip_'
		results_loc += args.importance_defin + '_'
		results_loc += 'num_tasks_' + str(num_tasks) + '_'
		results_loc += 'LR_' + str(lr_method) + '_'
		results_loc += 'Reg_' + str(reg_constant) + '_'
		results_loc += 'seed_' + args.seed
		results_loc += '.pkl'

		with open(results_loc, 'wb') as f:
			pkl.dump(stat, f)

	if(grid_search):
		grid_search_loc = './grid_search/' + dataset + '/'
		grid_search_loc += train_type + '_'
		if(use_flip):
			grid_search_loc += 'flip_'
		grid_search_loc += args.importance_defin + '_'
		grid_search_loc += 'num_tasks_' + str(num_tasks) + '_'
		grid_search_loc += 'LR_' + str(lr_method) + '_'
		grid_search_loc += 'Reg_' + str(reg_constant) + '_'
		grid_search_loc += 'seed_' + args.seed
		grid_search_loc += '.pkl'

		with open(grid_search_loc, 'wb') as f:
			pkl.dump(stat, f)