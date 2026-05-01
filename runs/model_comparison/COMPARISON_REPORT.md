# Model Comparison Study: Swin vs ViT vs CNN

## Executive Summary

This study compares three model architectures trained on the same lens detection dataset with identical hyperparameters.

### Models Evaluated

- **Swin**: Hierarchical Vision Transformer (swin_tiny_patch4_window7_224)
- **ViT**: Standard Vision Transformer (vit_small_patch16_224)
- **CNN**: Convolutional Neural Network (resnet50)

### Key Findings

#### Best Validation AUC
- Swin: 0.9918
- ViT: 0.9900
- CNN: 0.9800

**Winner**: Swin

#### Training Efficiency
- Swin: 35.6 minutes (converged at epoch 10)
- ViT: 32.2 minutes (converged at epoch 5)
- CNN: 50.0 minutes (converged at epoch 9)

**Fastest**: ViT

#### Low False-Positive Performance (Critical for Surveys)

**TPR @ FPR = 1e-3** (1 false positive per 1000 negatives)
- Swin: 0.9993
- ViT: 0.9993
- CNN: 1.0000

**TPR @ FPR = 1e-4** (1 false positive per 10,000 negatives)
- Swin: 0.9990
- ViT: 0.9993
- CNN: 1.0000

### Configuration (Fair Comparison)

All models trained with identical hyperparameters:
- Batch size: 16
- Learning rate: 0.0001
- Weight decay: 0.0001
- Epochs: 10
- Seed: 42 (reproducibility)
- Image size: 224

### Dataset

- Training samples: 28154
- Validation samples: 6032

### Conclusion

Swin achieved the best AUC (0.9918) on this lens detection task.
However, for survey-scale applications where false positives are costly, the TPR@FPR scores are critical.
