from models import CNNModel_P
import torch
import torch.nn as nn
import os
from torchvision import transforms,datasets,models
from torch.utils.data import DataLoader
from helper import test_model, binary_evaluation_metrics, binary_confusion_metrics

print("Model testing for covid dataset")

#setting up device
device = "cuda" if torch.cuda.is_available() else "cpu"

#dataset path
test_dir = ""

### Testing with CNN model

#loading stats 
data_stats = torch.load(f=os.path.join("Model_weights","covid_mean_std.pth"))
mean = data_stats['mean']
std = data_stats['std']

#transform
c_transform = transforms.Compose([
    transforms.Resize(size=(224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean.tolist(), std=std.tolist())
])

#loading data
c_test_data = datasets.ImageFolder(root=test_dir,
                            transform=c_transform)

print(f"Test data classes for CNN: {c_test_data.class_to_idx}")
print(f"Test data size for CNN : {len(c_test_data)}")

c_test_dataLoader = DataLoader(dataset=c_test_data,
                            batch_size=32,
                            shuffle=False,
                            num_workers=4)

#loading the model
cnn_model = CNNModel_P().to(device)
model_path = os.path.join("Model_weights","Covid_CNN_model.pth")
cnn_model.load_state_dict(torch.load(f=model_path))

#making predictions
c_test_predictions, c_test_targets = test_model(model=cnn_model,
                                            test_dataloader=c_test_dataLoader,
                                            device=device)

#resutls
print("Results for covid test dataset with CNN model")

binary_evaluation_metrics(predictions=c_test_predictions,
                          target=c_test_targets)

binary_confusion_metrics(predictions=c_test_predictions,
                         target=c_test_targets,
                         fig_name="Test_covid_CNN_confusion_matrix")


## Testing for Pneumonia dataset with ResNet Model

#loading the pre-trained ResNet50 model
weights = models.ResNet50_Weights.DEFAULT
r_transform = weights.transforms()
resnet_model = models.resnet50(weights=weights).to(device)
resnet_model.fc = nn.Linear(in_features=resnet_model.fc.in_features,
                     out_features=2).to(device)
r_model_path = os.path.join("Model_weights","Covid_ResNet50_model.pth")
resnet_model.load_state_dict(torch.load(f=r_model_path))

#loading data
r_test_data = datasets.ImageFolder(root=test_dir,
                            transform=r_transform)

print(f"Covid test data classes for ResNet: {r_test_data.class_to_idx}")
print(f"Covid test data size for ResNet : {len(r_test_data)}")

r_test_dataLoader = DataLoader(dataset=r_test_data,
                            batch_size=32,
                            shuffle=False,
                            num_workers=4
                            )

#making predictions
r_test_predictions, r_test_targets = test_model(model=resnet_model,
                                            test_dataloader=r_test_dataLoader,
                                            device=device)

#resutls
print("Results for Covid test dataset with ResNet50 model")

binary_evaluation_metrics(predictions=r_test_predictions,
                          target=r_test_targets)

binary_confusion_metrics(predictions=r_test_predictions,
                         target=r_test_targets,
                         fig_name="Test_covid_ResNet_confusion_matrix")

