# PointMLP for Point Cloud Displacement Prediction

This project implements a point cloud displacement prediction model based on PointMLP architecture.

## Project Structure 
pointmlp_displacement/
├── README.md
├── .gitignore
├── models/ # Model definitions
│ ├── __init__.py
│ ├── pointmlp.py # Original PointMLP implementation
│ └── pointmlp_displacement.py # Modified PointMLP for displacement
├── data_utils/ # Data loading utilities
│ ├── __init__.py
│ └── liver_dataset.py # Dataset loader for liver/vessel data
├── train_pointmlp_displacement.py # Training script
└── test_pointmlp_displacement.py # Testing script

## Requirements

- Python 3.7+
- PyTorch 1.7+
- numpy
- tqdm

## Installation

bash
git clone https://github.com/zachery-mai/pointmlp_displacement.git
cd pointmlp_displacement

## Data Preparation

Place your data in the following structure:
data/data_1/
├── liver_displacement_/.txt # Liver point cloud files
├── vessels_displacement_/.txt # Vessel point cloud files
├── liverpoints_train.txt # Training split for liver
├── vesselspoints_train.txt # Training split for vessels
├── liverpoints_test.txt # Testing split for liver
└── vesselspoints_test.txt # Testing split for vessels

## Usage

### Training
bash
python train_pointmlp_displacement.py --log_dir ./log_dir --use_cpu
Key arguments:
- `--log_dir`: Name of experiment
- `--batch_size`: Batch size for training (default: 6)
- `--epoch`: Number of epochs (default: 50)
- `--learning_rate`: Learning rate (default: 0.001)
- `--num_point`: Number of points in point cloud (default: 1024)

### Testing
bash
python test_pointmlp_displacement.py --log_dir ./log_dir --use_cpu

## Model Architecture

The model uses PointMLP as backbone and adds:
1. Dual input branches for liver and vessel point clouds
2. Feature fusion module
3. Displacement prediction head

## Data Format

Input:
- Liver point cloud: [B, 3, N]
- Vessel point cloud: [B, 3, N]

Output:
- Displacement vectors: [B, N, 3]

