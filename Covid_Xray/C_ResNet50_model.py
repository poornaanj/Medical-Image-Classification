import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models
from torch.utils.data import DataLoader
from torchmetrics.classification import Accuracy
from train_test_model import train_model_with_early_stop, plot_loss_accuracy, evaluation_metrics
from torchinfo import summary

#setting up device
device = "cuda" if torch.cuda.is_available() else "cpu"

#dataset paths
train_dir = ""
val_dir = ""
test_dir = ""

#loading the pre-trained ResNet50 model
weights = models.ResNet50_Weights.DEFAULT

transform = weights.transforms()

#loading data
train_data = datasets.ImageFolder(root=train_dir,
                            transform=transform)

val_data = datasets.ImageFolder(root=train_dir,
                            transform=transform)

test_data = datasets.ImageFolder(root=test_dir,
                                 transform=transform)

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

#initiating the model
model = models.resnet50(weights=weights).to(device)

#freezing the model
for params in model.parameters():
  params.requires_grad = False

#updating the classifer
torch.manual_seed(42)
torch.cuda.manual_seed(42)
output_shape = 4
model.fc = nn.Linear(in_features=model.fc.in_features,
                     out_features=output_shape).to(device)

#summary after updating the classifier layer
summary(model,
        input_size=(1, 3, 224, 224),
        col_names=["input_size", "output_size", "num_params", "trainable"])

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(params = model.parameters(),
                             lr=0.001)
acc_fn = Accuracy(task="multiclass", num_classes=2).to(device)

train_losses, train_accuracies, val_losses, val_accuracies, model_weights, val_predictions, val_targets = train_model_with_early_stop(model=model,
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
                   fig_name="covid_resnet50")

evaluation_metrics(predictions=val_predictions,
                   target=val_targets)