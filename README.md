# Explicit Movement Regularization for Continual Learning

Codebase for the paper ["Rethinking Quadratic Regularizers: Explicit Movement Regularization for Continual Learning."](https://arxiv.org/abs/2102.02805)

## Requirements

The code requires:

* Python 3.6 or higher

* Pytorch 1.4 or higher

To install other dependencies, the following command can be used (uses pip):

```setup
./requirements.sh
```

## Organization
The datasets and code are stored separately with the relative path shown below. 

For a given dataset *X* (e.g., CIFAR-100), the script data_prep_X.py can be used to divide the dataset into separate tasks. The task-wise splits will be stored in the directory *X_splits*. All models trained during the continual learning process will be stored in the directory **cl_models**. Stats collected for train/test/forgetting/cka are stored in the directory **results**. If grid search is applied, the results are stored in the directory **grid_search**. Any pretrained models are stored in the directory **pretrained**.

```bash
EMR
│   README.md
└───datasets
│   │   X
│   │   X_splits
│
└───codebase
│   └───cl_models
│   └───grid_search
│   └───results
│   └───pretrained
│   └───data_prep
│   │   │   data_prep_cifar.py
│   │   │   data_prep_flowers.py
│   │   │   data_prep_cal256.py
│   │   
│   │   reg_based.py
│   │   opt_based.py
│   │   models.py
│   │   cka.py
│   │   train_cifar10.py
│   │   utils.py
```

## Codebase
Code for different continual learning methods are in the following modules:

* **reg_based.py**: Use for training with regularization-based continual learning methods (*Quadratic regularization* / *Explicit movement regularization*). Run *python reg_based.py --help* for further details about usage.

* **opt_based.py**: Use for training with replay-based continual learning methods (*A-GEM* / *ER-Reservoir*). Run *python opt_based.py --help* for further details about usage.

## Example execution 
To train a model using regularization-based methods (e.g., Explicit movement regularization) for a specific importance definition (e.g., ewc) and specific dataset (e.g., CIFAR-100) at a given learning rate (e.g., 0.001), run the following command:

```Regularization-based
python reg_based.py --dataset=cifar --train_type=online_emr --importance_defin=ewc --lr_config=0.001
```

Similarly, for replay-based methods (e.g., A-GEM), run the following command:
```Regularization-based
python opt_based.py --dataset=cifar --train_type=agem --lr_config=0.001
```
