# Deep Learning Pipeline with Pytorch & DVC
## CIFAR-10 Image Classification

### Project overview
This project implements a complete and reproducible Deep learning pipeline in PyTorch. The goal was to build a modular training system that: 
- Handles data loading properly
- Trains and evaluates a nerual network model
- Tracks datasets using DVC(not git)
- Runs reproducibly via a single entrypoint (main.py)
- Includes multiple controlled experiments
- Documents hyperparameter impact on performance 

The dataset used is CIFAR-10, a standard image classification benchmark consisting of 60.000 32x32 RGB images across 10 classes. 

### What I built 
The project implements: 
- A custom dataset pipeline using `torchvision.datasets.CIFAR10`
- A modular CNN model (`model.py`)
- A training loop separated from evaluation logic (`train.py`)
- Configuration driven training via `params.yaml`
- Experiment tracking through DVC
- Reproducible execution via `main.py`

The final system separates: 
- Code
- Data 
- Experiment results

This reflects a real world ML engineering workflow rather than a single traning script.

### Project structure
```
ML-FRAMEWORK-MLOPS25/
│
├── .dvc/                      # DVC internal metadata
│   ├── cache/
│   ├── tmp/
│   ├── config
│   └── .gitignore
│
├── data/
│   ├── raw/                   # CIFAR-10 dataset (tracked by DVC, not Git)
│   ├── raw.dvc                # DVC tracking file for raw dataset
│   └── .gitignore
│
├── runs/                      # Experiment outputs (ignored by Git)
│   ├── exp1_baseline/
│   ├── exp2_low_lr_more_epochs/
│   └── exp3_big_batch/
│
├── src/                       # Source code (modular design)
│   ├── config.py              # Loads parameters from params.yaml
│   ├── dataset.py             # Data loading and transformations
│   ├── model.py               # CNN model definition
│   ├── train.py               # Training and evaluation logic
│   ├── utils.py               # Utilities (seed, device handling)
│   └── __pycache__/           # Python cache files
│
├── main.py                    # Entry point (runs full training pipeline)
├── params.yaml                # Experiment configuration
├── dvc.yaml                   # DVC pipeline definition
├── .dvcignore
├── .gitignore
├── .python-version
├── pyproject.toml             # Project dependencies (uv)
├── uv.lock                    # Locked dependency versions
└── README.md                  # Project documentation
```

This structure shows: 
- Proper separation of data and code 
- DVC-based dataset versioning
- Modular PyTorch design
- Reproducible experiment setup
- Clean root level entrypoint. 

It follows a simplified production style ML layout rather than a single script notebook workflow. 

### Dataset handling

CIFAR-10 is downloaded using `torchvision.datasets.CIFAR10` into:
data/raw/

The dataset is versioned using DVC:
`dvc add data/raw`

This creates:

data/raw.dvc

The .dvc file is tracked by Git, while the raw dataset is not. 

This ensures:
- No raw data is pushed to GitHub
- Data can be restored using `dvc pull`

### Model architecture 
I implemented a smal Convulutioanl Neural Network: 
- 3 convolutional blocks
- Max pooling 
- Fully connected classifier
- Dropout for regularization

The model outputs logits for 10 classes

Loss function: `CrossEntryLoss`
Optimizer: `Adam`

### Training pipeline 
The training pipeline: 
1. Loads configuration from params.yaml
2. Sets deterministic seed 
3. Loads dataset and DataLoaders
4. Initializes model 
5. Trains for configured number of epochs 
6. Evaluates on test set
7. Saves metrics to `runs/'experiments_name'/metrics.json`

Everything runs through: 
`uv run python main.py`
This ensures full reproducibility. 

### Experiments
I conducted three controlled experiments 

| Experiment | Epochs | Batch Size | LR     | Test Accuracy |
| ---------- | ------ | ---------- | ------ | ------------- |
| Baseline   | 5      | 128        | 0.001  | 0.74          |
| Low LR     | 10     | 128        | 0.0005 | 0.76          |
| Big Batch  | 10     | 256        | 0.001  | 0.75          |

### Analysis of results 
1. Lower learning rate + more epochs(best reults)
The experiment using a smaller learning rate (0.0005) and more epochs achieved the highest test accuracy(0.76).

This suggests: 
- Smaller update steps allowed smoother convergence
- Longer training enabled better optimization
- Generalization improved slightly 

2. Larger batch size did not improve performance
Increasing batch size to 256 did not improve accuracy. 

This aligns with known deep learning behavior:
- Larger batches produce more stable gradients
- However, they may reduce generalization
- Smaller batches often introduce beneficial noise in optimization

3. Performance difference are moderate
The improvement from 0.74 to 0.76 corresponds to roughly 200 additional correct classifications out of 10.000 test images. This is a meusurable but not a dramatic improvement. 
Overall, learning rate had a stronger effect than batch size in this setup. 

### Final model 
The final selected configuration is: 
Epochs: 10
Batch size: 128 
Learning rate: 0.0005

Running: 
`uv run python main.py`
trains and evaluates this final model en-to-end. 

### Challenges encountered 
1. Proper DVC usage
It was neccesary to ensure:
- Raw data is tracked by DVC
- .dvc files are commited to Git
- Raw data is not pushed to GitHub
This required careful .gitginore configuration

2. Modularization
Separating responsibilities into: 
- dataset.py 
- model.py 
- train.py
- main.py
prevent monolithic code and improved calrity. 

3. Reproducibility
Ensuring deterministic behavior required: 
- Setting seeds
- Centralizing configuration 
- Avoiding hardcoded parameters

### Reproducing the project 
1. Clone repository
2. Install enviroment 
`uv sync`
3. Pull dataset 
`dvc pull`
4. Train final model 
`uv run python main.py`

### What this project demonstrates 
This project demonstrates understanding of: 
- Deep learning fundamentals
- CNN architecture for image classification 
- Proper data versioning with dvc
- Experiment tracking 
- Modular software design 
- Reproducible ML pipelines 

The focus was not only on achieving accuracy, but on building a structured and reproducible ML system. 

### Future improvments 
Potential extension include:
- Learning rate schedulers
- Data augmentation tuning
- Deeper architectures (e.g., Resnet)
- TensorBoard intergration
- Automated hyperparameter search 

### Conclussion
I successfully built a modular, reproducible deep larning pipelines uisng PyTorch and DVC. 

The experiments demonstrates that learing rate selection had the largest impact on model performance within this configuration. 

The final modell achieves 76% test accuracy on CIFAR-10 using a simple CNN architecture. 

The primary achievement of this project is not the accuracy score itself, buth the implementation of structured, reproducible ML workflow. 