# Models Directory

This directory contains saved model weights from training experiments.

## File Naming Convention
- `[model_name]_bs[batch_size]_ep[epochs]_weights.pth`
- Example: `simple_cnn_bs20_ep15_weights.pth`

## Loading Models
```python
import torch
from models import SimpleCNN

# Load saved weights
model = SimpleCNN(num_classes=2)
model.load_state_dict(torch.load('models/simple_cnn_bs20_ep15_weights.pth'))
model.eval()