import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

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




