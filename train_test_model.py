import tqdm
import torch
import copy
import matplotlib.pyplot as plt

#training loop with early stop

def train_model_with_early_stop(model, train_dataLoader, val_dataLoader, loss_function, optimizer, accuracy_function, epoches, early_stop, device):

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

    with torch.inference_mode():
      for images, labels in val_dataLoader:

        images = images.to(device)
        labels = labels.to(device)

        predictions = model(images)
        loss = loss_function(predictions, labels)
        accuracy = accuracy_function(torch.argmax(predictions, dim=1),labels)

        total_val_loss += loss.item()*images.size(0)
        total_val_accuracy += accuracy.item()*images.size(0)

      avg_val_loss = total_val_loss/len(val_dataLoader.dataset)
      avg_val_accuracy = total_val_accuracy/len(val_dataLoader.dataset)

      val_losses.append(avg_val_loss)
      val_accuracies.append(avg_val_accuracy)

      if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        best_model_weights = copy.deepcopy(model.state_dict())
        early_stop = early_stop
      else:
         early_stop -= 1
         if early_stop == 0:
          break


    print("-"*100)
    print(f"Epoch : {epoch+1}")
    print(f"Train loss : {avg_train_loss:.4f} | Train accuracy {avg_train_accuracy:.4f}")
    print(f"Val loss : {avg_val_loss:.4f} | Val accuracy {avg_val_accuracy:.4f}")


  return train_losses, train_accuracies, val_losses, val_accuracies, best_model_weights

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