from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_cifar10_loaders(batch_size, val_batch_size, num_workers=2, data_dir="./data"):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),  # [0,1] -> [-1,1]
    ])

    trainset = datasets.CIFAR10(data_dir, train=True,  download=True, transform=transform)
    testset  = datasets.CIFAR10(data_dir, train=False, download=True, transform=transform)

    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    testloader = DataLoader(
        testset, batch_size=val_batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return trainloader, testloader, trainset, testset
