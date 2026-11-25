![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)
![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)
![Medical AI](https://img.shields.io/badge/Medical-AI-red.svg)
![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)

# FructureBOT - X-ray Fracture Detection with Deep Learning

A comprehensive deep learning project for automated fracture detection in X-ray images, featuring extensive model comparisons, hyperparameter experiments, and explainable AI techniques for medical diagnosis.

## Project Overview

This project implements and systematically compares multiple state-of-the-art deep learning architectures for binary fracture classification in X-ray images. The goal is to assist medical professionals in quickly and accurately identifying bone fractures through automated analysis with explainable predictions.

### Key Features

- **Multi-Architecture Comparison**: Custom CNN, ResNet-50, and EfficientNet-B0
- **Comprehensive Experiments**: Multiple batch sizes, epochs, and data augmentation strategies
- **Medical AI Focus**: Class-weighted training prioritizing fracture detection (2:1 weight ratio)
- **Explainable AI**: LIME (Local Interpretable Model-agnostic Explanations) for medical validity
- **Professional Structure**: Modular code blocks with extensive documentation
- **Automated Evaluation**: Confusion matrices, classification reports, and performance metrics

## Architecture Comparison

### Models Implemented:

1. **Custom CNN** - 4-layer architecture with BatchNorm and Dropout
   - Features: 32, 64, 128, 256 filters with MaxPooling
   - Classifier: AdaptiveAvgPool + 512-unit dense layer
   - Purpose: Baseline comparison and interpretability

2. **ResNet-50** - Deep residual network with transfer learning
   - Pre-trained on ImageNet, fine-tuned for fracture detection
   - Modified final layer for binary classification
   - Purpose: State-of-the-art performance with skip connections

3. **EfficientNet-B0** - Compound scaling optimization
   - Pre-trained weights with efficient architecture
   - Optimal balance of accuracy, speed, and model size
   - Purpose: Modern efficient architecture benchmark

## Dataset & Preprocessing

- **Source**: Medical X-ray images from bone fracture dataset
- **Classes**: Fractured vs Normal (non-fractured) bones
- **Image Size**: 224×224 pixels (standard CNN input)
- **Normalization**: ImageNet statistics for transfer learning compatibility
- **Split Strategy**: Random validation/test splits (60/40) with configurable seeds

### Data Augmentation (Configurable):
- Random horizontal flip (50% probability)
- Random rotation (±10 degrees)
- Color jitter (brightness & contrast ±20%)
- Controlled toggle for augmentation on/off experiments

## Experimental Design

The project includes comprehensive hyperparameter experiments:

### Experiment Variables:
- **Models**: Simple CNN, ResNet-50, EfficientNet-B0
- **Batch Sizes**: [20, 40] (configurable)
- **Epochs**: [15, 30] (configurable)
- **Learning Rate**: 0.001 (Adam optimizer)
- **Data Augmentation**: On/Off toggle
- **Class Weights**: 2:1 (fractured:normal) for medical priorities

### Medical-Focused Training:
- **Weighted Loss Function**: Penalizes missed fractures 2× more than false positives
- **Early Stopping**: Based on validation accuracy
- **Learning Rate Scheduling**: StepLR with γ=0.1 every 7 epochs

## Results & Performance

The notebook generates comprehensive performance analysis including:

### Metrics Tracked:
- **Accuracy**: Training, validation, and test accuracy
- **Precision/Recall**: Specifically for fracture detection
- **F1-Score**: Balanced metric for medical applications
- **Confusion Matrices**: Visual error analysis
- **Training Time**: Efficiency comparison

### Visualization Dashboard:
- Training/validation loss curves
- Accuracy progression over epochs
- Side-by-side confusion matrices
- Model comparison charts
- LIME explanation visualizations

## Explainable AI (XAI)

### LIME Integration:
- **Local Explanations**: Individual prediction interpretability
- **Medical Validation**: Anatomical relevance analysis
- **Visual Explanations**: Highlighted regions supporting decisions
- **Clinical Guidelines**: Framework for medical interpretation

### Medical Interpretation Framework:
- **Good Signs**: Focus on bone structures, cortical lines, joint spaces
- **Concerning Signs**: Attention to text, equipment, or artifacts
- **Evaluation Criteria**: Anatomical relevance, medical logic, consistency

## Installation & Setup

### Prerequisites
```bash
Python 3.8+
PyTorch 2.0+
torchvision
CUDA-capable GPU (recommended)
```

### Required Packages
```bash
# Core ML libraries
pip install torch torchvision
pip install numpy pandas matplotlib seaborn
pip install scikit-learn tqdm

# Image processing and XAI
pip install Pillow opencv-python
pip install lime scikit-image

# Jupyter environment
pip install jupyter ipykernel

# Data handling
pip install gdown  # For Google Drive dataset download
```

### Google Colab Setup
```python
# Install in Colab
!pip install lime opencv-python scikit-image gdown

# Enable GPU
# Runtime > Change runtime type > Hardware accelerator: GPU
```

## Usage


### Dataset Setup
The notebook automatically downloads and processes the dataset:
```python
# Google Drive dataset download
file_id = '1pfyAN5XE4ULgnx9NHfMCgw9QK6KCee_l'
# Automatic extraction and folder renaming
```

### Experiment Configuration
```python
# Modify these variables for different experiments:
MODELS_TO_TRAIN = ['Simple CNN', 'ResNet-50', 'EfficientNet-B0']
BATCH_SIZES = [16, 32, 64]
EPOCHS_LIST = [10, 20, 30]
LEARNING_RATE = 0.001
RANDOM_SEED = 42  # For reproducibility
```

## Project Structure

```
FructureBOT/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── AI_PROJECT_REFINED.ipynb      # Main analysis notebook
├── models/                            # Saved model weights (see models/README.md)
└── results/                           # Experimental results (see results/README.md)
```

## Methodology

### Training Pipeline:
1. **Data Preparation**: Automatic download, extraction, and preprocessing
2. **Model Initialization**: Architecture-specific setup with transfer learning
3. **Training Loop**: Progress tracking with tqdm, validation monitoring
4. **Evaluation**: Comprehensive metrics calculation and visualization
5. **Model Saving**: Automatic weight persistence for each experiment

### Medical AI Considerations:
- **Class Imbalance**: Addressed through weighted loss functions
- **Clinical Relevance**: Emphasis on fracture recall over overall accuracy
- **Interpretability**: LIME explanations for clinical decision support
- **Validation**: Medical interpretation guidelines for AI explanations

## Key Findings & Insights

Based on experimental results:

### Model Performance Insights:
- **Transfer Learning Advantage**: Pre-trained models (ResNet, EfficientNet) typically outperform custom CNN
- **Batch Size Impact**: Smaller batches (20) often provide better generalization
- **Augmentation Benefits**: Data augmentation generally improves robustness
- **Training Efficiency**: EfficientNet offers best accuracy-to-time ratio

### Medical AI Insights:
- **Explainability Matters**: LIME reveals whether models focus on anatomically relevant features
- **Class Weighting Crucial**: 2:1 weighting significantly improves fracture detection recall
- **Clinical Validation**: Manual review of LIME explanations essential for medical deployment

## Future Improvements

- [ ] **Advanced Architectures**: Vision Transformers (ViT), DenseNet variants
- [ ] **Multi-Class Extension**: Specific fracture type classification
- [ ] **Ensemble Methods**: Combining multiple model predictions
- [ ] **Advanced XAI**: GradCAM, Integrated Gradients, SHAP
- [ ] **Clinical Integration**: DICOM support, radiologist workflow integration
- [ ] **Deployment Pipeline**: Web interface, mobile application, hospital system API
- [ ] **Larger Datasets**: Multi-institutional validation
- [ ] **Real-time Processing**: Optimization for clinical deployment

## Technical Documentation

### Block Structure:
- **Block 0**: Environment setup and dataset cleaning
- **Block A**: Experiment configuration and parameters
- **Block B**: Data loading and preprocessing pipeline
- **Block C**: Model architecture definitions
- **Block D**: Training and evaluation functions
- **Block E**: Comprehensive experiment execution
- **Block F**: Results analysis and visualization
- **LIME Block**: Explainable AI implementation

### Design Patterns:
- **Modular Architecture**: Each block serves specific purpose
- **Configuration-Driven**: Easy parameter modification
- **Comprehensive Logging**: Detailed progress and results tracking
- **Medical Focus**: Class weighting and clinical metrics priority

## Medical Validation Guidelines

### Anatomical Focus Areas:
- Cortical bone lines and discontinuities
- Joint spaces and alignment
- Bone density variations
- Common fracture locations (distal radius, metacarpals)

### Red Flags for AI Explanations:
- Focus on text annotations or labels
- Attention to medical equipment
- Background artifacts or positioning aids
- Non-anatomical features

## Contributing

Contributions welcome! Please focus on:
- **Medical Accuracy**: Ensure clinical relevance
- **Code Quality**: Follow medical AI best practices
- **Documentation**: Clear methodology description
- **Validation**: Thorough testing with medical datasets

## Acknowledgments

- **Medical Community**: For fracture detection clinical guidelines
- **PyTorch Team**: For deep learning framework
- **LIME Developers**: For explainable AI capabilities
- **Dataset Contributors**: For providing medical imaging data

## Contact & Citation

**Author**: Kostas Plomaritis  
**Institution**: [AUTh University - BioMedical Engineering]  
**Email**: [kplomari@gmail.com]  
**LinkedIn**: [-]

**Project Repository**: [https://github.com/kostas-g33579/FructureBOT](https://github.com/kostas-g33579/FructureBOT)

### Citation
```bibtex
@software{fracture_bot_2025,
  title={FructureBOT: Deep Learning for X-ray Fracture Detection with Explainable AI},
  author={Kostas G},
  year={2025},
  url={https://github.com/kostas-g33579/FructureBOT}
}
```

---

> **Medical Disclaimer**: This tool is for research purposes only and should not replace professional medical diagnosis. Always consult qualified healthcare professionals for medical decisions.
