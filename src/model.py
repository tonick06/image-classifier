"""Transfer-learning model builders.

Loads a pretrained backbone, optionally freezes it, and swaps the final
layer for a fresh head sized to your number of classes.
"""
import torch.nn as nn
from torchvision import models


def build_model(model_name: str, num_classes: int, freeze_backbone: bool = True):
    name = model_name.lower()

    if name == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_features, head = m.fc.in_features, "fc"
    elif name == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        in_features, head = m.fc.in_features, "fc"
    elif name == "mobilenet_v3_large":
        m = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        in_features, head = m.classifier[-1].in_features, "classifier"
    elif name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        in_features, head = m.classifier[-1].in_features, "classifier"
    else:
        raise ValueError(
            f"Unknown model_name '{model_name}'. Use resnet18, resnet50, "
            "mobilenet_v3_large or efficientnet_b0."
        )

    if freeze_backbone:
        for p in m.parameters():
            p.requires_grad = False

    # Replace the classification head (its new params are trainable by default).
    if head == "fc":
        m.fc = nn.Linear(in_features, num_classes)
    else:
        m.classifier[-1] = nn.Linear(in_features, num_classes)

    return m
