import Pytorch.Network as net
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.utils.data as Data
from torchvision import transforms
import torch
from tensorboardX import SummaryWriter

# MNIST = torchvision.datasets.MNIST('/MNIST', download=True)
# MNIST.train_
dataloader = Data.DataLoader(
    torchvision.datasets.MNIST('../MNIST', train=True, download=True, transform=transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))
    ])), batch_size=32, shuffle=True, )
device = torch.device("cuda")
model = net.Renet(28, 1, 0).to(device)
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0001)
writer = SummaryWriter('log')
for i in range(1000):
    # total_loss = 0
    for index, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        pred = model(data)
        pred = F.softmax(pred, 1)
        assert torch.sum(torch.gt(pred, 1.0)).data == 0
        loss = F.nll_loss(pred, target)
        loss.backward()
        optimizer.step()
        # total_loss += loss
        writer.add_scalar('loss', loss, i * 1874 + index)
        print("Epoch : {}, Batch Index : {}\n".format(i, index))
        print("loss : {}\n".format(loss))
        print("Accuracy : ")
        print(float(torch.sum(torch.eq(torch.argmax(pred, 1), target)).data) / 32.0)
