import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from train_test_model import train_model_with_early_stop, plot_loss_accuracy, binary_evaluation_metrics, binary_confusion_metrics
from torchmetrics.classification import Accuracy
from torchinfo import summary
from helper import get_mean_std

#setting up device
device = "cuda" if torch.cuda.is_available() else "cpu"

#dataset paths
train_dir = ""
test_dir = ""

#mean and std for transform
mean, std = get_mean_std(train_dir)

#defining a transform
transform = transforms.Compose([
    transforms.Resize(size=(224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean.tolist(), std=std.tolist())
])

#loading data
data = datasets.ImageFolder(root=train_dir,
                            transform=transform)

test_data = datasets.ImageFolder(root=test_dir,
                                 transform=transform)

#split data tor train and validation datasets
indices = list(range(len(data)))

train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=42)

train_data = Subset(data, train_indices)
val_data = Subset(data, val_indices)

print(f"Classes: {test_data.class_to_idx}")
print(f"Train data size : {len(train_data)}")
print(f"Validation data size : {len(val_data)}")
print(f"Test data size : {len(test_data)}")

#dataloaders
num_workers = 4

train_dataLoader = DataLoader(dataset=train_data,
                              batch_size=32,
                              shuffle=True,
                              num_workers=num_workers)
val_dataLoader = DataLoader(dataset=val_data,
                            batch_size=32,
                            shuffle=False,
                            num_workers=num_workers)
test_dataLoader = DataLoader(dataset=test_data,
                             batch_size=32,
                             shuffle=False,
                             num_workers=num_workers
                             )

images, labels = next(iter(train_dataLoader))
print(f"Batch shape : {images.shape}")
print(f"Labels shape : {labels.shape}")

#CNN model architecture

class CNNModel_V0(nn.Module):

  def __init__(self):
    super().__init__()

    self.conv_block_1 = nn.Sequential(
        nn.Conv2d(in_channels=3,
                  out_channels=16,
                  kernel_size=3,
                  stride=2,
                  padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2,
                     stride=2)
    )

    self.conv_block_2 = nn.Sequential(
        nn.Conv2d(in_channels=16,
                  out_channels=32,
                  kernel_size=3,
                  stride=2,
                  padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2,
                     stride=2)
    )

    self.conv_block_3 = nn.Sequential(
        nn.Conv2d(in_channels=32,
                  out_channels=64,
                  kernel_size=3,
                  stride=2,
                  padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2,
                     stride=2)
    )

    self.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features=64*3*3,
                  out_features=2)
    )

  def forward(self, x):
    return self.classifier(self.conv_block_3(self.conv_block_2(self.conv_block_1(x))))
  
#initiating a model
model_0 = CNNModel_V0().to(device)

#obtaining a summary of the model
summary(model_0,input_size=[1,3,224,224])

#loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_0.parameters(), lr=0.001)

# accuracy function
acc_fn = Accuracy(task="multiclass", num_classes=2)
acc_fn.to(device)

#training model
train_losses, train_accuracies, val_losses, val_accuracies, model_weights, val_predictions, val_targets = train_model_with_early_stop(model=model_0,
                                                                                         train_dataLoader=train_dataLoader,
                                                                                         val_dataLoader=val_dataLoader,
                                                                                         loss_function=loss_fn,
                                                                                         optimizer=optimizer,
                                                                                         accuracy_function=acc_fn,
                                                                                         epoches=50,
                                                                                         early_stop=10,
                                                                                         device=device)

plot_loss_accuracy(train_losses=train_losses,
                   val_losses=val_losses,
                   train_accuracies=train_accuracies,
                   val_accuracies=val_accuracies,
                   fig_name="Pneumonia_CNN_loss_curves")

binary_evaluation_metrics(predictions=val_predictions,
                          target=val_targets)

binary_confusion_metrics(predictions=val_predictions,
                         target=val_targets,
                         fig_name="Pneumonia_CNN_confusion_matrix")