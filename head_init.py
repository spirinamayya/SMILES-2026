"""
head_init.py — Final layer initialization (student-implemented).

Students: Implement `init_last_layer` to control how the new classification
head is initialized before fine-tuning begins. The skeleton below uses
Kaiming uniform weights and zero bias — you are expected to experiment with
alternatives (e.g. Xavier, orthogonal, small-scale random, learned bias init).
"""

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.models as models

from augmentation import get_transforms

def init_last_layer(layer: nn.Linear) -> None:
    """Initialize the weights and bias of the final classification layer in-place.

    This function is called once during model construction (see model.py).
    Modify it to experiment with different initialization strategies and observe
    their effect on the "initialized head" evaluation checkpoint.

    Args:
        layer: The ``nn.Linear`` layer that serves as the new CIFAR100 head.
               Modifies the layer in-place; return value is ignored.

    Student task:
        Replace or extend the skeleton below. Some strategies to consider:
          - ``nn.init.xavier_uniform_``  — preserves variance across layers
          - ``nn.init.orthogonal_``      — encourages diverse feature directions
          - Small-scale init (e.g. scale weights by 0.01) — conservative start
          - Non-zero bias init           — useful when class priors are known
    """
    # -------------------------------------------------------------------------
    # STUDENT: Replace or extend the initialization below.
    # -------------------------------------------------------------------------
    dataset = datasets.CIFAR100(
        root="./data",
        train=True,
        transform=get_transforms(train=False),
        download=False,
    )

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Identity()
    model.eval()

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
    )

    all_x = []
    all_y = []
    model.eval()
    with torch.no_grad():
        for image, target in dataloader:
            all_x.append(model(image))
            all_y.append(target)

    x = torch.cat(all_x, dim=0)
    y = torch.cat(all_y, dim=0)

    x_centered = x - x.mean(dim=0, keepdim=True)
    x_augmented = torch.cat([x_centered, torch.ones(x_centered.shape[0], 1)], dim=1)

    Y = torch.nn.functional.one_hot(y, num_classes=layer.out_features).float()
    Y = 0.95 * Y  + (1.0 - Y) * (0.05 / max(1, layer.out_features - 1))

    reg_term = torch.eye(layer.in_features + 1) * 1000
    reg_term[-1, -1] = 0.0

    answer = torch.linalg.solve(x_augmented.T @ x_augmented + reg_term, x_augmented.T @ Y)

    weight = answer[:-1].T
    bias = answer[-1] - (x.mean(dim=0, keepdim=True) @ weight.T).squeeze(0)
    bias = bias - bias.mean()

    with torch.no_grad():
        layer.weight.copy_(weight)
        layer.bias.copy_(bias)

    # -------------------------------------------------------------------------
