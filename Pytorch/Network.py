import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        pass

    def forward(self, *input):
        pass


def make_layers(cfg, in_channels):
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'm':
            layers += [nn.MaxPool2d(kernel_size=2, stride=1)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


# [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        configure = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'm', 512, 512, 512, 'm']
        self.seq = make_layers(configure, 3)
        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=12, dilation=12)  # fc6 in paper
        self.conv7 = nn.Conv2d(1024, 1024, 3, 1, 1)  # fc7 in paper

    def forward(self, *input):
        x = input[0]
        conv1 = self.seq[:4](x)
        conv2 = self.seq[4:9](conv1)
        conv3 = self.seq[9:16](conv2)
        conv4 = self.seq[16:23](conv3)
        conv5 = self.seq[23:30](conv4)
        conv6 = self.conv6(conv5)
        conv7 = self.conv7(conv6)

        return conv1, conv2, conv3, conv4, conv5, conv7


class DecoderCell(nn.Module):
    def __init__(self, in_channel, out_channel, mode):
        super(DecoderCell, self).__init__()
        self.bn_en = nn.BatchNorm2d(in_channel)
        self.conv1 = nn.Conv2d(2 * in_channel, in_channel, kernel_size=3, padding=1)  # not specified in paper
        if mode == 'G':
            self.picanet = PiCANet_G()
        elif mode == 'L':
            self.picanet = PiCANet_L()
        else:
            assert 0
        self.conv2 = nn.Conv2d(2 * in_channel, out_channel, kernel_size=3, padding=1)  # not specified in paper
        self.bn_feature = nn.BatchNorm2d(out_channel)
        self.conv3 = nn.Conv2d(out_channel, 1, kernel_size=1, padding=0)  # not specified in paper

    def forward(self, *input):
        assert len(input) <= 2
        if len(input) == 1:
            En = input[0]
            Dec = input[0]  # not specified in paper
        elif len(input) == 2:
            En = input[0]
            Dec = input[1]

        if Dec.size()[2] * 2 == En.size()[2]:
            Dec = F.upsample(Dec, scale_factor=2, mode='bilinear', align_corners=True)
        elif Dec.size()[2] != En.size()[2]:
            assert 0
        En = self.bn_en(En)
        En = F.relu(En)
        fmap = torch.cat((En, Dec), dim=1)  # F
        fmap_att = self.picanet(fmap)  # F_att
        x = torch.cat((fmap, fmap_att), 1)
        x = self.conv2(x)
        x = self.bn_feature(x)
        Dec_out = F.relu(x)
        _y = self.conv3(Dec_out)
        _y = F.sigmoid(_y)

        return Dec_out, _y


class PiCANet_G(nn.Module):
    def __init__(self, size, in_channel, out_channel):
        super(PiCANet_G, self).__init__()
        self.renet = Renet(size, in_channel, out_channel)
        self.in_channel = in_channel

    def forward(self, *input):
        x = input[0]
        size = x.size()
        kernel = self.renet(x)
        kernel = F.softmax(kernel, 1)
        kernel = torch.reshape(kernel, (size[0] * size[2] * size[3], 1, 1, 10, 10))
        x = torch.unsqueeze(x, 0)
        x = F.conv3d(input=x, weight=kernel, bais=None, stride=1, padding=0, dilation=(1, 3, 3), groups=self.in_channel)
        x = torch.reshape(x, (size[0], size[1], size[2], size[3]))
        return x


class PiCANet_L(nn.Module):
    def __init__(self, in_channel):
        super(PiCANet_L, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 128, kernel_size=7, dilation=2, padding=6)
        self.conv2 = nn.Conv2d(128, 49, kernel_size=1)

    def forward(self, *input):
        x = input[0]
        size = x.size()
        kernel = self.conv1(x)
        kernel = self.conv2(kernel)
        kernel = F.softmax(kernel, 1)
        kernel = torch.reshape(kernel, (size[0], 1, 1, 7, 7, size[2], size[3]))
        fmap = []
        x = F.pad(x, (6, 6, 6, 6))
        x = torch.unsqueeze(0)
        for i in range(size[2]):
            for j in range(size[3]):
                pix = F.conv3d(input=F.pad(x, (6-i, -6-i, 6-j, -6-j)), weight=kernel[:, :, :, :, :, i, j], dilation=(1, 2, 2), groups=size[1])
                fmap.append(pix)
        x = torch.cat(fmap, 3)
        x = torch.reshape(x, (size[0], size[1], size[2], size[3]))
        return x


class Renet(nn.Module):
    def __init__(self, size, in_channel, out_channel):
        super(Renet, self).__init__()
        self.size = size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.vertical = nn.LSTM(input_size=in_channel, hidden_size=256, batch_first=True,
                                bidirectional=True)  # each row
        self.horizontal = nn.LSTM(input_size=512, hidden_size=256, batch_first=True,
                                  bidirectional=True)  # each column
        # self.conv = nn.Conv2d(1, out_channel, 1)
        self.fc = nn.Linear(512 * size * size, 10)

    def forward(self, *input):
        x = input[0]
        temp = []
        size = x.size()  # batch, in_channel, height, width
        x = torch.transpose(x, 1, 3)  # batch, width, height, in_channel
        for i in range(self.size):
            h, _ = self.vertical(x[:, :, i, :])
            temp.append(h)  # batch, width, 512
        x = torch.stack(temp, dim=2)  # batch, width, height, 512
        temp = []
        for i in range(self.size):
            h, _ = self.horizontal(x[:, i, :, :])
            temp.append(h)  # batch, width, 512
        x = torch.stack(temp, dim=3)  # batch, height, 512, width
        x = torch.transpose(x, 1, 2)  # batch, 512, height, width
        x = torch.reshape(x, (-1, 512 * self.size * self.size))
        x = self.fc(x)
        return x


if __name__ == '__main__':
    vgg = torchvision.models.vgg16(pretrained=True)
    model = Encoder()
    model.seq.load_state_dict(vgg.features.state_dict())
    # print(model.state_dict().keys())
    # print(vgg.features.state_dict().keys())
    # print(vgg.features)
    noise = torch.randn((1, 3, 224, 224))
    # print(vgg.features(noise))
    # print(model(noise))
    print(model.seq)
    # print(vgg.features)
    # print(F.mse_loss(model.seq[:8](noise), vgg.features[:8](noise)))
