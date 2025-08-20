import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import os
import pickle
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

import matplotlib.pyplot as plt

CIFAR10_LABELS = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

class CNN_MLP(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN_MLP, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        return self.classifier(self.features(x))

def load_model(model_name):
    if model_name == "resnet":
        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, 10)
        model.load_state_dict(torch.load("resnet_best.pth", map_location='cpu'), strict=False)
    elif model_name == "cnn_mlp":
        model = CNN_MLP()
        model.load_state_dict(torch.load("cnn_mlp_best.pth", map_location='cpu'))
    else:
        raise ValueError("Invalid model name")
    model.eval()
    return model

def load_cifar10_test_images(dataset_path):
    with open(os.path.join(dataset_path, 'test_batch'), 'rb') as f:
        data = pickle.load(f, encoding='bytes')
    images = data[b'data'].reshape((10000, 3, 32, 32)).astype("uint8")
    labels = data[b'labels']
    return images, labels

def predict(model, image_tensor):
    with torch.no_grad():
        output = model(image_tensor.unsqueeze(0))
        _, predicted = torch.max(output, 1)
    return predicted.item()

def prepare_images_for_visualization(model, images, labels, output_dir='static/images'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    results = []
    for idx in range(50):
        img_np = np.transpose(images[idx], (1, 2, 0))
        img_tensor = transform(img_np)
        pred_label = predict(model, img_tensor)
        correct = pred_label == labels[idx]

        pred_class = CIFAR10_LABELS[pred_label]
        true_class = CIFAR10_LABELS[labels[idx]]

        save_path = os.path.join(output_dir, f"img_{idx}.png")
        plt.figure(figsize=(2, 2))
        plt.imshow(img_np)
        plt.axis('off')
        plt.title(f"P: {pred_class}\nT: {true_class}", fontsize=8, color='green' if correct else 'red')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

        results.append((f"img_{idx}.png", pred_class, true_class, correct))
    return results
