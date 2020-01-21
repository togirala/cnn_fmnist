import torchvision.datasets as ds
import torchvision.transforms as transforms
import torch.utils.data as data

res_path = 'models/'



input_training_set = ds.FashionMNIST(
        root='input/', \
        train = True, 
        # transform=None, 
        target_transform=None, 
        download=True,
        transform = transforms.Compose([transforms.ToTensor()])
        )

# data_loader = data.DataLoader(input_training_set, batch_size=4, shuffle=True, num_workers=args.nThreads)
batch_loader = data.DataLoader(input_training_set, batch_size=1000, shuffle=False)





