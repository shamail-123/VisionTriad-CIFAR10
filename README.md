# VisionTriad-CIFAR10
ViT vs ResNet (transfer) vs CNN+MLP on CIFAR-10
VisionTriad-CIFAR10 — ViT vs ResNet (Transfer) vs CNN+MLP on CIFAR-10

Classify CIFAR-10 images using three approaches:

a Vision Transformer (ViT) implemented from scratch,

a hybrid CNN→MLP architecture (no Transformer),

a transfer-learned ResNet (pretrained on ImageNet, frozen backbone + new head).

This repo includes standardized preprocessing/augmentations, training with early stopping and LR scheduling, rich evaluation (accuracy, precision, recall, F1), confusion matrices, error analysis, performance benchmarking (training time, memory, inference speed), deployment, and tidy experiments logging.

##Features

CIFAR-10 preprocessing & augmentations (random crop, flip, normalization)

ViT from scratch: patchifying, learnable positional embedding, pre-norm blocks, MHA, GELU MLP, Dropout, residuals

Hybrid CNN→MLP baseline using patch features learned via conv blocks

ResNet-18/34/50 transfer learning with frozen trunk + trainable classifier head

Early stopping, cosine LR schedule with warmup, AMP/mixed precision, gradient clipping

Metrics: accuracy (top-1), macro precision/recall/F1, confusion matrix

Visualizations: train/val curves, confusion matrix heatmap, correct vs incorrect grids

Benchmarks: training wall-clock, GPU memory peak, inference latency/throughput

Optional Optuna hyperparameter sweeps

Simple Streamlit or Flask app for local demo & comparisons

Reproducibility (seeds, deterministic flags)
