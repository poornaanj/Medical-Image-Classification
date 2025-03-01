import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from helper import train_model_with_early_stop, plot_loss_accuracy, evaluation_metrics, confusion_metrics, get_mean_std
from torchmetrics.classification import Accuracy
from torchinfo import summary
from models import CNNModel_C

#setting up device
device = "cuda" if torch.cuda.is_available() else "cpu"

#dataset paths
train_dir = ""
val_dir = ""

#mean and std for transform
mean, std = get_mean_std(train_dir)
mean_std_path = os.path.join("..","Model_weights")
os.makedirs(mean_std_path,exist_ok=True)
mean_std_path = os.path.join(mean_std_path,'covid_mean_std.pth')
torch.save({'mean':mean,'std':std},mean_std_path)

#transform
transform = transforms.Compose([
    transforms.Resize(size=(224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean.tolist(), std=std.tolist())
])

#loading data
train_data = datasets.ImageFolder(root=train_dir,
                            transform=transform)

val_data = datasets.ImageFolder(root=train_dir,
                            transform=transform)


print(f"Classes: {train_data.class_to_idx}")
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


images, labels = next(iter(train_dataLoader))
print(f"Batch shape : {images.shape}")
print(f"Labels shape : {labels.shape}")


#initiating a model
model_0 = CNNModel_C().to(device)

#obtaining a summary of the model
summary(model_0,input_size=[1,3,224,224])

#loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_0.parameters(), lr=0.001)

# accuracy function
acc_fn = Accuracy(task="multiclass", num_classes=4)
acc_fn.to(device)

train_losses, train_accuracies, val_losses, val_accuracies, val_predictions, val_targets = train_model_with_early_stop(model=model_0,
                                                                                         train_dataLoader=train_dataLoader,
                                                                                         val_dataLoader=val_dataLoader,
                                                                                         loss_function=loss_fn,
                                                                                         optimizer=optimizer,
                                                                                         accuracy_function=acc_fn,
                                                                                         epoches=50,
                                                                                         early_stop=10,
                                                                                         device=device,
                                                                                         model_name="Covid_CNN_model")

plot_loss_accuracy(train_losses=train_losses,
                   val_losses=val_losses,
                   train_accuracies=train_accuracies,
                   val_accuracies=val_accuracies,
                   fig_name="Covid_CNN")

evaluation_metrics(predictions=val_predictions,
                   target=val_targets)

confusion_metrics(predictions=val_predictions,
                  target=val_targets,
                  fig_name = "Covid_CNN_confusion_matrix")