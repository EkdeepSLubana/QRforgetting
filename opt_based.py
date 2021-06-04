import torch
import torchvision
from torchvision import datasets, models, transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from models import *
from config import *
from cka import cka
from ptflops import get_model_complexity_info
import copy
import numpy as np
import pickle as pkl
import os
import argparse
from adabelief_pytorch import AdaBelief

######### Parser #########
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", help="model architecture", default='vanilla_cnn', choices=['vanilla_cnn', 'resnet-18'])
parser.add_argument("--seed", help="setup the random seed", default='0')
parser.add_argument("--use_pretrained", help="use pretrained CIFAR-10 model", default='True')
parser.add_argument("--dataset", help="dataset used", default='cifar', choices=['cifar', 'flowers', 'cal256'])
parser.add_argument("--split_pattern", help="pattern to divide tasks with", default='coarse_labels', choices=['sequential', 'random', 'coarse_labels'])
parser.add_argument("--batch_size", help="batch size for dataloaders", default='128')
parser.add_argument("--optimizer", help="optimizer for training", default='SGD', choices=['SGD', 'AdaBelief'])
parser.add_argument("--use_AGC", help="use AGC?", default='False', choices=['True', 'False'])
parser.add_argument("--train_type", help="train nets with which method", choices=['agem', 'ogd', 'er_reservoir'])
parser.add_argument("--grid_search", help="grid search for hyperparameters", default='False', choices=['True', 'False'])
parser.add_argument("--lr_config", help="learning rate", default='use_grid')
parser.add_argument("--buffer_size", help="buffer size", default='500')
parser.add_argument("--print_cka", help="print task-wise cka similarities", default='False', choices=['True', 'False'])
parser.add_argument("--save_results", help="save CL specific stats for later analysis", default='False', choices=['True', 'False'])
args = parser.parse_args()

######### Setup #########
torch.manual_seed(int(args.seed))
cudnn.deterministic = True
cudnn.benchmark = False
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
model_name = args.model # model architecture
use_pretrained = (args.use_pretrained=='True')
if(model_name=='vanilla_cnn'):
	pretrained_path = 'pretrained/vanilla_cnn_temp_1.0_seed_0_plain.pth'
else:
	pretrained_path = 'pretrained/resnet-18_temp_1.0_seed_0.pth'
dataset = args.dataset + '_splits'
split_pattern = args.split_pattern
num_tasks = len(os.listdir('./../datasets/'+dataset+'/'+split_pattern))
total_classes = 100 if (args.dataset == 'cifar') else 102 if (args.dataset == 'flowers') else 256
num_classes = total_classes // num_tasks
batch_size = int(args.batch_size)
opt_type = args.optimizer
use_AGC = (args.use_AGC == 'True')
if(split_pattern == 'coarse_labels'):
	num_tasks, num_classes = 20, 5
train_type = args.train_type
grid_search = (args.grid_search == 'True')
wd = wd_cl # weight decay
buffer_size = int(args.buffer_size) # buffer size
print_cka = (args.print_cka == 'True')
save_results = (args.save_results == 'True')

# Hyperparameters
cl_epochs = cl_epochs_config
wd = wd_cl

if(grid_search):
	start_task, end_task = 0, 3
	lr_options = [0.1, 0.03, 0.01, 0.003, 0.001]
	reg_options = [0]
	print_cka = False
	save_results = True
elif(args.lr_config=='use_grid'):
	with open("./hyperparams/" + dataset + '/' + train_type + "_None" + ".params", 'rb') as f:
		coeffs = pkl.load(f)['Test']
	start_task, end_task = 3, num_tasks
	lr_options = [coeffs['LR']]
	reg_options = [0]
else:
	start_task, end_task = 3, num_tasks
	lr_options = [cl_sched_config] if (args.lr_config=='use_config') else [float(args.lr_config)]
	reg_options = [0]
num_tasks = end_task - start_task

hyperparams = []
for l in lr_options:
	for r in reg_options:
		hyperparams.append((l, r))

print("\n------------------ Setup For Training ------------------\n")
print("Model architecture:", model_name)
print("Training type:", train_type)
print("Buffer size:", buffer_size)
print("Data split pattern:", split_pattern)
print("Batch size:", batch_size)
print("Optimizer:", opt_type)
print("Weight decay for CL:", wd)
print("Number of tasks:", num_tasks)
print("Number of classes per task:", num_classes)
print("Pretrained model:", use_pretrained)
print("Save stats:", save_results)

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

######### Basic functions #########
def clone_model_params(model):
	params = []
	for param in model.parameters():
		if not param.requires_grad:
			continue
		params.append(param.detach().clone())
	return params

def clone_model_grads(model):
	grads = []
	for param in model.parameters():
		if not param.requires_grad:
			continue
		grads.append(0. if param.grad is None else param.grad.detach().clone() + 0.)
	return grads

######### Initialization / training functions #########
# Create model
def create_model(name, num_classes=5, num_tasks=20, is_pretrained=False):
	if(name == 'vanilla_cnn' and is_pretrained):
		net = torch.nn.DataParallel(Vanilla_cnn(num_classes=num_classes))
	elif(name == 'vanilla_cnn'):
		net = torch.nn.DataParallel(Vanilla_cnn_multiclassifier(num_classes=num_classes, num_tasks=num_tasks))
	elif(name == 'resnet-18' and is_pretrained):
		net = torch.nn.DataParallel(ResNet18(num_classes=num_classes))
	elif(name == 'resnet-18'):
		net = torch.nn.DataParallel(ResNet18_multiclassifier(num_classes=num_classes, num_tasks=num_tasks))
	return net

# Training
def buffer_update(mem_buffer, label_buffer, tid_buffer, dataloader, buffer_size, task_id):
	samples_per_task = buffer_size // (task_id - start_task + 1)
	ref_indices = torch.randperm(buffer_size)[0:samples_per_task]
	update_rounds = (samples_per_task // batch_size) + 1

	dataloader_iter = iter(dataloader)
	updates_finished = 0
	while (updates_finished < samples_per_task):
		X, Y = next(dataloader_iter)
		for i in range(batch_size):
			updates_finished += 1
			if(updates_finished == samples_per_task):
				break
			mem_buffer[ref_indices[updates_finished]] = X[i]
			label_buffer[ref_indices[updates_finished]] = Y[i]
			tid_buffer[ref_indices[updates_finished]] = task_id
	return mem_buffer, label_buffer

# AGC
def AGC(net, optimizer):
	eta = optimizer.param_groups[0]['lr']
	lambd = optimizer.param_groups[0]['weight_decay']
	beta = optimizer.param_groups[0]['momentum']
	threshold = np.sqrt(2 * lambd / (eta * (1 + beta)))

	for mod in net.modules():
		if(isinstance(mod, nn.Conv2d)):
			g_norms = torch.norm(mod.weight.grad.data.reshape(mod.weight.shape[0], -1), dim=1)
			p_norms = torch.norm(mod.weight.data.reshape(mod.weight.shape[0], -1), dim=1)
			ratios = torch.div(g_norms, p_norms + 1e-8) 
			multiplier = (ratios < threshold) * 1
			multiplier = multiplier + (1 - multiplier) * threshold * (torch.div(p_norms, g_norms + 1e-8))
			mod.weight.grad.data = mod.weight.grad.data * multiplier.view(-1,1,1,1)								
	return net

def agem_train(net_curr, mem_buffer, label_buffer, tid_buffer, dataloader, epoch, task_id=0, buffer_size=200, batch_size=10):
	net_curr.train()
	train_loss = 0
	correct = 0
	total = 0
	for batch_idx, (inputs, targets) in enumerate(dataloader):
		if(task_id > start_task):
			optimizer.zero_grad()
			ref_indices = torch.randperm(buffer_size - batch_size)
			ref_samples_X, ref_samples_Y,  ref_samples_tid = (mem_buffer[ref_indices[0]: ref_indices[0]+batch_size]).to(device), (label_buffer[ref_indices[0]: ref_indices[0]+batch_size]).to(device), (tid_buffer[ref_indices[0]: ref_indices[0]+batch_size]).to(device)
			outputs = net_curr(ref_samples_X, ref_samples_tid, num_classes=num_classes)
			loss = criterion(outputs, ref_samples_Y)
			loss.backward()
			ref_grads = clone_model_grads(net_curr)

		optimizer.zero_grad()
		inputs, targets = inputs.to(device), targets.to(device)
		outputs = net_curr(inputs, (task_id * torch.ones(inputs.shape[0], dtype=torch.long)).to(device), num_classes=num_classes)
		loss = criterion(outputs, targets)
		loss.backward()
		if(use_AGC):
			net_curr = AGC(net=net_curr, optimizer=optimizer)
		step_grads = clone_model_grads(net_curr)

		if(task_id > start_task):
			inner_prod = 0
			ref_norm = 0
			for g_step, g_ref in zip(step_grads, ref_grads):
				inner_prod += (g_step * g_ref).sum()
				ref_norm += g_ref.norm().pow(2)

			if(inner_prod < 0):
				lind = 0
				for mod in net_curr.modules():
					if isinstance(mod, nn.Conv2d):
						mod.weight.grad -= (inner_prod / ref_norm) * ref_grads[lind]
						lind += 1
						try:
							mod.bias.grad -= (inner_prod / ref_norm) * ref_grads[lind]
						except:
							pass
					elif isinstance(mod, nn.BatchNorm2d):
						mod.weight.grad -= (inner_prod / ref_norm) * ref_grads[lind]
						lind += 1
						mod.bias.grad -= (inner_prod / ref_norm) * ref_grads[lind]
						lind += 1

		optimizer.step()

		train_loss += loss.item()
		_, predicted = outputs.max(1)
		total += targets.size(0)
		correct += predicted.eq(targets).sum().item()
		progress_bar(batch_idx, len(dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
			% (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def er_reservoir_train(net_curr, mem_buffer, dataloader, epoch, task_id=0, buffer_size=200, batch_size=10):
	net_curr.train()
	train_loss = 0
	correct = 0
	total = 0
	for batch_idx, (inputs, targets) in enumerate(dataloader):
		if(task_id > start_task):
			optimizer.zero_grad()
			ref_indices = torch.randperm(buffer_size - batch_size)
			ref_samples_X, ref_samples_Y,  ref_samples_tid = (mem_buffer[ref_indices[0]: ref_indices[0]+batch_size]).to(device), (label_buffer[ref_indices[0]: ref_indices[0]+batch_size]).to(device), (tid_buffer[ref_indices[0]: ref_indices[0]+batch_size]).to(device)
			outputs = net_curr(ref_samples_X, ref_samples_tid, num_classes=num_classes)
			loss = criterion(outputs, ref_samples_Y)
			loss.backward()
			if(use_AGC):
				net_curr = AGC(net=net_curr, optimizer=optimizer)
			optimizer.step()

		optimizer.zero_grad()
		inputs, targets = inputs.to(device), targets.to(device)
		outputs = net_curr(inputs, (task_id * torch.ones(inputs.shape[0], dtype=torch.long)).to(device), num_classes=num_classes)
		loss = criterion(outputs, targets)
		loss.backward()
		if(use_AGC):
			net_curr = AGC(net=net_curr, optimizer=optimizer)
		optimizer.step()

		train_loss += loss.item()
		_, predicted = outputs.max(1)
		total += targets.size(0)
		correct += predicted.eq(targets).sum().item()
		progress_bar(batch_idx, len(dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
			% (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

# eval
def eval(net, testloader, task_id=0, T=1.0, save=False):
	net.eval()
	test_loss = 0
	correct = 0
	total = 0
	with torch.no_grad():
		for batch_idx, (inputs, targets) in enumerate(testloader):
			inputs, targets = inputs.to(device), targets.to(device)
			outputs = net(inputs, (task_id * torch.ones(inputs.shape[0], dtype=torch.long)).to(device), num_classes=num_classes)
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
		torch.save(state, cl_root+'{mod_name}_{data_name}_train_{ttype}_lr_{lr}_tasks_{ntasks}_taskid_{tid}_seed_{sid}.pth'.format(mod_name=model_name, data_name=dataset, ttype=train_type, lr=lr_method, ntasks=num_tasks, tid=task_id, sid=int(args.seed)))
		return acc

# Calculate accuracy on a given dataloader
def cal_acc(net, use_loader, task_id=0):
	net.eval()
	test_loss = 0
	correct = 0
	total = 0
	with torch.no_grad():
		for batch_idx, (inputs, targets) in enumerate(use_loader):
			inputs, targets = inputs.to(device), targets.to(device)
			outputs = net(inputs, (task_id * torch.ones(inputs.shape[0], dtype=torch.long)).to(device), num_classes=num_classes)
			loss = criterion(outputs, targets)
			test_loss += loss.item()
			_, predicted = outputs.max(1)
			total += targets.size(0)
			correct += predicted.eq(targets).sum().item()
	return 100.*(correct / total)

######### Rewinding functions #########
def rewind_conv(net, net_base):
	for mod, mod_base in zip(net.modules(), net_base.modules()):
		if(isinstance(mod, nn.Conv2d)):
			mod.weight.data = mod_base.weight.data.detach().clone()
			try:
				mod.bias.data = mod_base.bias.data.detach().clone()
			except:
				pass
		elif(isinstance(mod, nn.BatchNorm2d)):
			mod.weight.data = mod_base.weight.data.detach().clone()
			mod.bias.data = mod_base.bias.data.detach().clone()
			mod.running_mean.data = mod_base.running_mean.data.clone()
			mod.running_var.data = mod_base.running_var.data.clone()
	return net

def net_rewinding(net_features, net_classifier):
	net_rewind = create_model(name=model_name, num_classes=num_classes, num_tasks=num_tasks+start_task)
	for (mod_rewind, mod_features) in zip(net_rewind.modules(), net_features.modules()):
		if(isinstance(mod_rewind, nn.Conv2d)):
			mod_rewind.weight.data = mod_features.weight.data.clone()
			try:
				mod.bias.data = mod_base.bias.data.detach().clone()
			except:
				pass

		elif(isinstance(mod_rewind, nn.BatchNorm2d)):
			mod_rewind.weight.data = mod_features.weight.data.clone()
			mod_rewind.bias.data = mod_features.bias.data.clone()
			mod_rewind.running_mean.data = mod_features.running_mean.data.clone()
			mod_rewind.running_var.data = mod_features.running_var.data.clone()

	for (mod_rewind, mod_classifier) in zip(net_rewind.modules(), net_classifier.modules()):
		if(isinstance(mod_rewind, nn.Linear)):
			mod_rewind.weight.data = mod_classifier.weight.data.clone()
			mod_rewind.bias.data = mod_classifier.bias.data.clone()
	return net_rewind

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
				gram_curr = torch.matmul(f_curr, f_curr.t()).cpu().numpy()
				# Original model's gram matrix w.r.t ReLU features
				f_orig = (net_base.module.features[0:lind+1](X.to(device))).reshape(batch_size, -1)
				gram_orig = torch.matmul(f_orig, f_orig.t()).cpu().numpy()
				# CKA
				cka_val = cka(gram_curr, gram_orig, debiased=True)
				stat[task_id]['cka'].append(cka_val)
				if(print_cka):
					print("Layer {:d}: {:.3f}".format(lind-1, cka_val))

def update_results(net_curr, upper_id, split_pattern, task_final=False, lr_method=0):
	av_tracc, av_teacc = 0, 0
	print("\n")
	for tid in range(start_task, upper_id+1):
		### net_classifier ###
		net_classifier = create_model(name=model_name, num_classes=num_classes, num_tasks=num_tasks+start_task)
		net_path = cl_root+'{mod_name}_{data_name}_train_{ttype}_lr_{lr}_tasks_{ntasks}_taskid_{tid}_seed_{sid}.pth'.format(mod_name=model_name, data_name=dataset, ttype=train_type, lr=lr_method, ntasks=num_tasks, tid=tid, sid=int(args.seed))
		net_dict = torch.load(net_path)
		net_classifier.load_state_dict(net_dict['net'])
	
		### Dataloaders ###
		trloader, teloader = get_dataloader(tid, split_pattern=split_pattern)

		### rewind net ###
		net_r = net_rewinding(net_features=net_curr, net_classifier=net_classifier)

		tracc = cal_acc(net_r.eval(), trloader, task_id=tid)
		teacc = cal_acc(net_r.eval(), teloader, task_id=tid)

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

for (lr_method, reg_constant) in hyperparams: 

	### Initialize model ###
	net_curr = create_model(name=model_name, num_classes=num_classes, num_tasks=num_tasks+start_task)
	stat = {task_id:{'orig_acc': 0, 'final_acc': 0, 'max_acc': 0, 'cka': []} for task_id in range(start_task, end_task)}

	### Use pretrained model ###
	if(use_pretrained):
		print("\n------------------ Loading pretrained model ------------------\n")
		net_pretrained = create_model(name=model_name, num_classes=10, is_pretrained=True)
		net_dict = torch.load(pretrained_path)
		net_pretrained.load_state_dict(net_dict['net'])
		net_curr = rewind_conv(net_curr, net_pretrained)
		del net_dict, net_pretrained

	######### CL process begins here #########
	for task_id in range(start_task, end_task):
		print("\n------------------ Task ID: {tid} ------------------\n".format(tid=task_id))

		### Dataloaders ###
		curr_trainloader, curr_testloader = get_dataloader(task_id, split_pattern=split_pattern)
		if(task_id == start_task):
			dataloader_iter = iter(curr_trainloader)
			X, Y = next(dataloader_iter)
			mem_buffer, label_buffer, tid_buffer = torch.zeros(buffer_size, X.shape[1], X.shape[2], X.shape[3]), torch.zeros(buffer_size, dtype=torch.long), torch.zeros(buffer_size, dtype=torch.long)
			del dataloader_iter

		### Optimizer ###
		if(task_id == start_task):
			if(opt_type=='SGD'):
				optimizer = optim.SGD(net_curr.parameters(), lr=0, momentum=0.9, weight_decay=1e-4)
			elif(opt_type=='AdaBelief'):
				optimizer = AdaBelief(net_curr.parameters(), lr=1e-3, eps=1e-8, betas=(0.9,0.999), weight_decay=wd_cl, weight_decouple = True, rectify = False)
		else:
			if(opt_type=='SGD'):
				optimizer = optim.SGD(net_curr.parameters(), lr=0, momentum=0.9, weight_decay=wd_cl)
			elif(opt_type=='AdaBelief'):
				optimizer = AdaBelief(net_curr.parameters(), lr=1e-3, eps=1e-8, betas=(0.9,0.999), weight_decay=wd_cl, weight_decouple = True, rectify = False)

		### Train ###
		epoch = 0
		optimizer.param_groups[0]['lr'] = lr_method

		print("\n--Training at {lr} learning rate for {n} epochs".format(lr=lr_method, n=cl_epochs))

		for n in range(cl_epochs):
			print('\nEpoch: {}'.format(epoch))

			# Train
			if(train_type == 'agem'):
				agem_train(net_curr=net_curr, mem_buffer=mem_buffer, label_buffer=label_buffer, tid_buffer=tid_buffer, dataloader=curr_trainloader, epoch=epoch, task_id=task_id, buffer_size=buffer_size, batch_size=batch_size)

			elif(train_type == 'ogd'):
				ogd_train(net_curr=net_curr, mem_buffer=mem_buffer, dataloader=curr_trainloader, epoch=epoch, task_id=task_id, buffer_size=buffer_size, batch_size=batch_size)

			elif(train_type == 'er_reservoir'):
				er_reservoir_train(net_curr=net_curr, mem_buffer=mem_buffer, dataloader=curr_trainloader, epoch=epoch, task_id=task_id, buffer_size=buffer_size, batch_size=batch_size)

			# # Eval
			# eval(net_curr, testloader=curr_testloader, task_id=task_id, save=False)

			epoch += 1
		task_acc = eval(net_curr, testloader=curr_testloader, task_id=task_id, save=True) # save current task's model

		### Update buffer ###
		print(" Updating buffer... ", end="")
		mem_buffer, label_buffer = buffer_update(mem_buffer=mem_buffer, label_buffer=label_buffer, tid_buffer=tid_buffer, dataloader=curr_trainloader, buffer_size=buffer_size, task_id=task_id)
		print("Done.")

		### Update accuracy numbers ###
		stat[task_id]['orig_acc'] = task_acc
		if(save_results and (not grid_search)):
			print("\n------------------ Progress Check ------------------")
			av_train_acc, av_test_acc = update_results(net_curr=net_curr, upper_id=task_id, split_pattern=split_pattern, task_final=(task_id==end_task-1), lr_method=lr_method)

		if(task_id > start_task and (not grid_search)):
			pass #layerwise_cka(net_curr, net_orig, task_id=task_id)
		else:
			net_orig = copy.deepcopy(net_curr)

	######### Print and save final stats #########
	print("\n------------------ Final Stats ------------------")
	if(not save_results or grid_search):
		av_train_acc, av_test_acc = update_results(net_curr=net_curr, upper_id=task_id, split_pattern=split_pattern, task_final=(task_id==end_task-1), lr_method=lr_method)

	av_forgetting = 0
	for task_id in range(start_task, end_task):
		print("Task "+str(task_id)+":")
		print("\tOriginal Accuracy: {:.2f}".format(stat[task_id]['orig_acc']))
		print("\tMaximum Accuracy: {:.2f}".format(stat[task_id]['max_acc']))
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

	if(save_results):
		results_loc = './results/' + dataset + '/'
		results_loc += (model_name + '_')
		results_loc += train_type + '_None_'
		results_loc += 'num_tasks_' + str(num_tasks) + '_'
		results_loc += 'LR_' + str(lr_method) + '_'
		results_loc += 'Reg_' + str(reg_constant) + '_'
		results_loc += 'seed_' + args.seed
		results_loc += '.pkl'

		with open(results_loc, 'wb') as f:
			pkl.dump(stat, f)

	if(grid_search):
		grid_search_loc = './grid_search/' + dataset + '/'
		grid_search_loc += (model_name + '_')
		grid_search_loc += train_type + '_None_'
		grid_search_loc += 'num_tasks_' + str(num_tasks) + '_'
		grid_search_loc += 'LR_' + str(lr_method) + '_'
		grid_search_loc += 'Reg_' + str(reg_constant) + '_'
		grid_search_loc += 'seed_' + args.seed
		grid_search_loc += '.pkl'

		with open(grid_search_loc, 'wb') as f:
			pkl.dump(stat, f)