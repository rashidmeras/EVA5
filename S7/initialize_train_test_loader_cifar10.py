import torch
import torchvision.transforms as transforms
import torchvision

def train_test_loader(batch_size = 4, num_workers=2):

    print("\n Initialize train and test loader with Batch Size:{}".format(batch_size))

    torch.manual_seed(22)

    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2023, 0.1994, 0.2010)

    transform = transforms.Compose(
                                    [transforms.ToTensor(),
                                    transforms.Normalize(mean, std)]
                                  )
    
    train_transform = transforms.Compose([
                            transforms.ColorJitter(brightness=0.10, contrast=0.1, saturation=0.10, hue=0.1),
                            transforms.RandomAffine(degrees=10, translate=(0.1,0.1), scale=(0.9, 1.1)),
                            transforms.RandomRotation((-7.0,7.0), fill=(1,)),                          
                            transforms.ToTensor(),
                            transforms.Normalize(mean, std)
                        ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, test_loader
