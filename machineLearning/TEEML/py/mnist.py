import torchvision, torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

train = torchvision.datasets.MNIST(root='./mnist/', train=True, transform=transforms.ToTensor(), download=True)

dataloader = DataLoader(train, batch_size=50, shuffle=True)
for step, (x, y) in enumerate(dataloader):
    b_x = x.shape
    b_y = y.shape
    print('Step: ', step, '| train_data的维度', b_x, '| train_target的维度', b_y)
