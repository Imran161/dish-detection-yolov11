import torch
import os
from ultralytics import YOLO
from PIL import Image
import requests
from roboflow import Roboflow
import optuna
import matplotlib.pyplot as plt
import joblib
import argparse
from ultralytics.utils import SETTINGS

from optuna.visualization.matplotlib import plot_optimization_history
from optuna.visualization.matplotlib import plot_param_importances

from augment import get_aug_params
from utils import get_device

SETTINGS['tensorboard'] = True


MODEL = YOLO('yolo11n.pt')
DATA_PATH = "dish-detection-3/data.yaml"

DEVICE = get_device()
print(f"Using device: {DEVICE}")



def train_model(params, trial=None):
    if trial:
        experiment_name = f'exp_{trial.number:03d}'
        
        best_dir = os.path.join('outputs/train', experiment_name)
    else:
        experiment_name = 'default'
        
    train_args = {
            'data': os.path.abspath(DATA_PATH),
            'epochs': params.get('epochs', 200),
            'imgsz': params.get('imgsz', 640),
            'batch': params.get('batch', 16),
            'device': DEVICE,
            'lr0': params.get('lr0', 0.001),
            'weight_decay': params.get('weight_decay', 0.0005),
            'project': 'outputs/train',
            'name': experiment_name,
            'plots': True,
            "visualize": True,
        }
    
    train_args.update(get_aug_params())
    train_args.update(params)
    return MODEL.train(**train_args)