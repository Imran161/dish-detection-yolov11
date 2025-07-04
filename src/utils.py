import os
import torch

def get_device():
    device = os.getenv('DEVICE', 'cuda:0').lower()
    
    if device == 'cpu':
        return 'cpu'
    
    if 'cuda' in device:
        if torch.cuda.is_available():
            return device
        print("CUDA not available, switching to CPU")
    
    return 'cpu'