import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from torchmetrics.classification import Accuracy
from helper import train_model_with_early_stop, plot_loss_accuracy, binary_evaluation_metrics, binary_confusion_metrics

#setting up device
device = "cuda" if torch.cuda.is_available() else "cpu"

#dataset paths
train_dir = "Replace with dataset path"

#loading the pre-trained ResNet50 model
weights = models.ResNet50_Weights.DEFAULT

transform = weights.transforms()

#loading data
data = datasets.ImageFolder(root=train_dir,
                            transform=transform)

#split data tor train and validation datasets
indices = list(range(len(data)))

train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=42)

train_data = Subset(data, train_indices)
val_data = Subset(data, val_indices)

print("Pneumonia dataset details for ResNet model")
print(f"Classes: {data.class_to_idx}")
print(f"Train data size : {len(train_data)}")
print(f"Validation data size : {len(val_data)}")

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

#initiating the model
model = models.resnet50(weights=weights).to(device)

#freezing the model
for params in model.parameters():
  params.requires_grad = False

#updating the classifer
torch.manual_seed(42)
torch.cuda.manual_seed(42)
output_shape = 2
model.fc = nn.Linear(in_features=model.fc.in_features,
                     out_features=output_shape).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(params = model.parameters(),
                             lr=0.001)
acc_fn = Accuracy(task="multiclass", num_classes=2).to(device)

train_losses, train_accuracies, val_losses, val_accuracies, val_predictions, val_targets = train_model_with_early_stop(model=model,
                                                                                         train_dataLoader=train_dataLoader,
                                                                                         val_dataLoader=val_dataLoader,
                                                                                         loss_function=loss_fn,
                                                                                         optimizer=optimizer,
                                                                                         accuracy_function=acc_fn,
                                                                                         epoches=50,
                                                                                         early_stop=10,
                                                                                         device=device,
                                                                                         model_name="Pneumonia_ResNet50_model")

plot_loss_accuracy(train_losses=train_losses,
                   val_losses=val_losses,
                   train_accuracies=train_accuracies,
                   val_accuracies=val_accuracies,
                   fig_name="Pneumonia_ResNet_loss_curves")

binary_evaluation_metrics(predictions=val_predictions,
                          target=val_targets)

binary_confusion_metrics(predictions=val_predictions,
                         target=val_targets,
                         fig_name="Pneumonia_ResNet_confusion_matrix")