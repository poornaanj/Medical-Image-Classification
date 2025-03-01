import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from tqdm import tqdm
import torch
import copy
import matplotlib.pyplot as plt
from torchmetrics.classification import BinaryPrecision, BinaryRecall, BinaryF1Score, BinaryConfusionMatrix, MulticlassPrecision, MulticlassRecall, MulticlassF1Score, MulticlassConfusionMatrix, BinaryAccuracy, MulticlassAccuracy
import seaborn as sns
import os


def train_model_with_early_stop(model, train_dataLoader, val_dataLoader, loss_function, optimizer, accuracy_function, epoches, early_stop, device, model_name):

  train_losses = []
  train_accuracies = []
  val_losses = []
  val_accuracies = [] 

  best_loss = float('inf')
  best_model_weights = None
  early_stop = early_stop

  for epoch in tqdm(range(epoches),total=epoches):

    #training
    model.train()

    total_train_loss = 0
    total_train_accuracy = 0

    for images, labels in train_dataLoader:

      images = images.to(device)
      labels = labels.to(device)

      predictions = model(images)
      loss = loss_function(predictions, labels)
      accuracy = accuracy_function(torch.argmax(predictions, dim=1),labels)

      total_train_loss += loss.item()*images.size(0)
      total_train_accuracy += accuracy.item()*images.size(0)

      optimizer.zero_grad()

      loss.backward()

      optimizer.step()

    avg_train_loss = total_train_loss/len(train_dataLoader.dataset)
    avg_train_accuracy = total_train_accuracy/len(train_dataLoader.dataset)

    train_losses.append(avg_train_loss)
    train_accuracies.append(avg_train_accuracy)

    #validation
    model.eval()

    total_val_loss = 0
    total_val_accuracy = 0

    val_predictions = []
    val_targets = []

    with torch.inference_mode():
      for images, labels in val_dataLoader:

        images = images.to(device)
        labels = labels.to(device)

        predictions = model(images)
        loss = loss_function(predictions, labels)
        predictions = torch.argmax(predictions, dim=1)
        accuracy = accuracy_function(predictions,labels)

        val_predictions.append(predictions.cpu())
        val_targets.append(labels.cpu())

        total_val_loss += loss.item()*images.size(0)
        total_val_accuracy += accuracy.item()*images.size(0)

      avg_val_loss = total_val_loss/len(val_dataLoader.dataset)
      avg_val_accuracy = total_val_accuracy/len(val_dataLoader.dataset)

      val_losses.append(avg_val_loss)
      val_accuracies.append(avg_val_accuracy)

      val_predictions = torch.cat(val_predictions)
      val_targets = torch.cat(val_targets)

      if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        best_model_weights = copy.deepcopy(model.state_dict())
        early_stop = early_stop
      else:
         early_stop -= 1
         if early_stop == 0:
          print("Early stopping")
          break


    print("-"*100)
    print(f"Epoch : {epoch+1}")
    print(f"Train loss : {avg_train_loss:.4f} | Train accuracy {avg_train_accuracy:.4f}")
    print(f"Val loss : {avg_val_loss:.4f} | Val accuracy {avg_val_accuracy:.4f}")

  #saving the model
  base_dir = os.path.join("..", "Model_weights") 
  os.makedirs(base_dir, exist_ok=True)
  model_path = os.path.join(base_dir, f"{model_name}.pth")
  torch.save(best_model_weights, model_path)

  return train_losses, train_accuracies, val_losses, val_accuracies, val_predictions, val_targets


def test_model(model, test_dataloader, device):

    test_predictions = []
    test_targets = []

    model.eval()
    with torch.inference_mode():
        for images, labels in test_dataloader:

            images = images.to(device)
            labels = labels.to(device)

            predictions = torch.argmax(model(images),dim=1)
            test_predictions.append(predictions.cpu())
            test_targets.append(labels.cpu())

        test_predictions = torch.cat(test_predictions)
        test_targets = torch.cat(test_targets)

    return test_predictions, test_targets


def plot_loss_accuracy(train_losses,val_losses, train_accuracies, val_accuracies, fig_name:str):

    plt.figure(figsize=(15, 5))
  
    plt.subplot(1,2,1)
    plt.plot(train_losses,label="Train loss")
    plt.plot(val_losses,label="Validation loss")
    plt.xlabel("No of epoches")
    plt.ylabel("Loss")
    plt.title("Loss")
    plt.grid(axis='x')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(train_accuracies,label="Train accuracy")
    plt.plot(val_accuracies,label="Validation accuracy")
    plt.xlabel("No of epoches")
    plt.ylabel("Accuracy")
    plt.title("Accuracy")
    plt.grid(axis='x')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{fig_name}.png")
    plt.close()

def binary_evaluation_metrics(predictions:torch.tensor, target:torch.tensor):
  
  accuracy_metric = BinaryAccuracy()
  accuracy = accuracy_metric(predictions, target)
  
  recall_metric = BinaryRecall()
  recall = recall_metric(predictions, target)

  precision_metric = BinaryPrecision()
  precision = precision_metric(predictions, target)

  f1_metric = BinaryF1Score()
  f1 = f1_metric(predictions,target)

  print(f"Accuracy : {accuracy:.4f}")
  print(f"Recall : {recall:.4f}")
  print(f"Precision : {precision:.4f}")
  print(f"F1 score : {f1:.4f}")

def evaluation_metrics(predictions:torch.tensor, target:torch.tensor):

  accuracy_metric = MulticlassAccuracy(num_classes=4, average='macro')
  accuracy = accuracy_metric(predictions, target)

  recall_metric = MulticlassRecall(num_classes=4, average='macro')
  recall = recall_metric(predictions, target)

  precision_metric = MulticlassPrecision(num_classes=4, average='macro')
  precision = precision_metric(predictions, target)

  f1_metric = MulticlassF1Score(num_classes=4, average='macro')
  f1 = f1_metric(predictions,target)

  print(f"Accuracy : {accuracy:.4f}")
  print(f"Recall : {recall:.4f}")
  print(f"Precision : {precision:.4f}")
  print(f"F1 score : {f1:.4f}")

def binary_confusion_metrics(predictions:torch.tensor, target:torch.tensor, fig_name:str):

  cm_metric = BinaryConfusionMatrix()
  bcm = cm_metric(predictions,target)
  bcm = bcm.numpy()

  #plot
  plt.figure(figsize=(5, 4))
  sns.heatmap(bcm, annot=True, fmt="d", cmap="Blues", xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
  plt.xlabel("Predicted Label")
  plt.ylabel("True Label")
  plt.title("Confusion Matrix")
  plt.savefig(f"{fig_name}.png")
  plt.close()

def confusion_metrics(predictions:torch.tensor, target:torch.tensor, fig_name:str):

  cm_metric = MulticlassConfusionMatrix(num_classes=4)
  cm = cm_metric(predictions,target)
  cm = cm.numpy()

  #plot
  plt.figure(figsize=(5, 4))
  sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["0", "1","2","3"], yticklabels=["0", "1","2","3"])
  plt.xlabel("Predicted Label")
  plt.ylabel("True Label")
  plt.title("Confusion Matrix")
  plt.savefig(f"{fig_name}.png")
  plt.close()

def get_mean_std(train_dir):
    total_images= 0
    mean = torch.zeros(3)
    std = torch.zeros(3)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    data = datasets.ImageFolder(root=train_dir,transform=transform)
    loader = DataLoader(data, batch_size=32, shuffle=False)


    for images, _ in tqdm(loader,total=len(loader)):

        batch_size, num_channels, height, width = images.shape
        total_images += batch_size

        # Sum mean and std per channel
        mean += images.mean(dim=(0, 2, 3)) * batch_size
        std += images.std(dim=(0, 2, 3)) * batch_size

    mean /= total_images
    std /= total_images

    return mean, std




