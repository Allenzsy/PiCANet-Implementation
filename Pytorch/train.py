import torch
from torch.utils.data import DataLoader
import torchvision
from Pytorch.Network import Unet
from Pytorch.Dataset import DUTS_dataset
from tensorboardX import SummaryWriter
import datetime
import os


cfg = {'PicaNet': "GGLLL",
       'Size': [28, 28, 28, 56, 112, 224],
       'Channel': [1024, 512, 512, 256, 128, 64],
       'loss_ratio': [0.5, 0.5, 0.5, 0.8, 0.8, 1]}

if __name__ == '__main__':
    vgg = torchvision.models.vgg16(pretrained=True)
    device = torch.device("cuda")
    batch_size = 1
    epoch = 20
    dataset = DUTS_dataset('../DUTS-TR')
    # noise = torch.randn((batch_size, 3, 224, 224)).type(torch.cuda.FloatTensor)
    # target = torch.randn((batch_size, 1, 224, 224)).type(torch.cuda.FloatTensor)

    # print(vgg.features(noise))
    # print(model(noise))
    # print(model.seq)
    # print(vgg.features)
    # print(F.mse_loss(model.seq[:8](noise), vgg.features[:8](noise)))
    model = Unet(cfg).cuda()
    model.encoder.seq.load_state_dict(vgg.features.state_dict())
    opt_en = torch.optim.SGD(model.encoder.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
    opt_dec = torch.optim.SGD(model.decoder.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
    dataloader = DataLoader(dataset, batch_size)
    now = datetime.datetime.now()
    os.makedirs('log/{}'.format(now.strftime('%m%d%H%M')), exist_ok=True)
    writer = SummaryWriter('log/{}'.format(now.strftime('%m%d%H%M')))
    for epo in range(epoch):
        for i, batch in enumerate(dataloader):
            # print(batch['image'].size())
            opt_dec.zero_grad()
            opt_en.zero_grad()
            img = batch['image'].to(device)
            mask = batch['mask'].to(device)
            _, loss = model(img, mask)
            loss.backward()
            opt_dec.step()
            opt_en.step()
            writer.add_scalar('loss', float(loss), global_step= i + epo*len(dataloader))
            if i % 1000 == 0:
                os.makedirs('models/{}'.format(now.strftime('%m%d%H%M')), exist_ok=True)
                torch.save(model, 'models/{}/{}epo_{}step.ckpt'.format(now.strftime('%m%d%H%M'), epo, i))
