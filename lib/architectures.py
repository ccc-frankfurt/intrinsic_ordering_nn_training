import torch
import torch.nn as nn
import torch.nn.functional as F


def get_feat_size(block, spatial_size, ncolors=3):
    """
    Function to infer spatial dimensionality in intermediate stages of a model after execution of the specified block.
    Parameters:
        block (torch.nn.Module): Some part of the model, e.g. the encoder to determine dimensionality before flattening.
        spatial_size (int): Quadratic input's spatial dimensionality.
        ncolors (int): Number of dataset input channels/colors.
    """

    x = torch.randn(2, ncolors, spatial_size, spatial_size)
    out = block(x)
    num_feat = out.size(1)
    spatial_dim_x = out.size(2)
    spatial_dim_y = out.size(3)

    return num_feat, spatial_dim_x, spatial_dim_y


class Net(nn.Module):

    """ from https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html"""

    # input size (32,32) color images
    def __init__(self, num_classes, num_channels, args):
        super(Net, self).__init__()
        self.args = args
        self.num_channels = num_channels
        self.conv1 = nn.Conv2d(num_channels, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        # getting the flattened input dimensionality
        convs = nn.Sequential(self.conv1, self.pool, self.conv2, self.pool)
        self.channels, self.spatial_dim_x, self.spatial_dim_y = get_feat_size(convs, args.patch_size, num_channels)

        self.fc1 = nn.Linear(self.channels * self.spatial_dim_x * self.spatial_dim_y, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.channels * self.spatial_dim_x * self.spatial_dim_y)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class LeNet5(nn.Module):
    """ From https://towardsdatascience.com/implementing-yann-lecuns-lenet-5-in-pytorch-5e05a0911320
    Batch-norm has been added to ensure better optimization
    and ReLU instead of Tanh for better comparison with other architectures """

    def __init__(self, num_classes, num_channels, args):
        super(LeNet5, self).__init__()
        self.args = args
        self.num_channels = num_channels

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=num_channels, out_channels=6, kernel_size=5, stride=1),
            nn.BatchNorm2d(6, eps=args.batch_norm),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),

            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.BatchNorm2d(16, eps=args.batch_norm),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),

            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
            nn.BatchNorm2d(120, eps=args.batch_norm),
            nn.ReLU()
        )

        # getting the flattened input dimensionality
        self.channels, self.spatial_dim_x, self.spatial_dim_y = get_feat_size(
            self.feature_extractor, args.patch_size, num_channels)

        self.classifier = nn.Sequential(
            nn.Linear(in_features=self.channels*self.spatial_dim_x*self.spatial_dim_y, out_features=84),
            nn.ReLU(),
            nn.Linear(in_features=84, out_features=num_classes),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        return logits


class VGG(nn.Module):
    """from https://github.com/kuangliu/pytorch-cifar"""

    def __init__(self, num_classes, num_channels, args, vgg_name='VGG16'):
        super(VGG, self).__init__()

        cfg = {
            'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
            'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
        }
        self.args = args
        self.batch_norm = args.batch_norm

        self.num_channels = num_channels
        self.features = self._make_layers(cfg[vgg_name])

        # getting the flattened input dimensionality
        self.channels, self.spatial_dim_x, self.spatial_dim_y = get_feat_size(
            self.features, args.patch_size, num_channels)

        # not needed, because GAP reduces the dimensions to number of classes
        self.classifier = nn.Linear(self.channels*self.spatial_dim_x*self.spatial_dim_y, num_classes)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = self.num_channels
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x, eps=self.batch_norm),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        #layers += [nn.AdaptiveAvgPool2d((1,1))]
        return nn.Sequential(*layers)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, batch_norm=1e-5):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, eps=batch_norm)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, eps=batch_norm)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes, eps=batch_norm)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes, eps=batch_norm)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    """ From https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py

    See also https://medium.com/@14prakash/understanding-and-implementing-architectures-of-resnet-and-resnext-for-state-of-the-art-image-cf51669e1624

    ResNet architecture idea from Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
     """

    def __init__(self, num_classes, num_channels, args, resnet_name='ResNet50'):
        super(ResNet, self).__init__()
        self.args = args
        self.num_channels = num_channels
        self.in_planes = 64

        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, eps=args.batch_norm)

        if resnet_name == 'ResNet50':
            block = Bottleneck
            num_blocks = [3, 4, 6, 3]

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, batch_norm=args.batch_norm)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, batch_norm=args.batch_norm)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, batch_norm=args.batch_norm)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, batch_norm=args.batch_norm)

        self.avg_pool2d = nn.AvgPool2d(4)
        convs = nn.Sequential(self.conv1, self.layer1, self.layer2, self.layer3, self.layer4,
                                   self.avg_pool2d)

        # getting the flattened input dimensionality
        self.channels, self.spatial_dim_x, self.spatial_dim_y = get_feat_size(
            convs, args.patch_size, num_channels)

        self.linear = nn.Linear(self.channels*self.spatial_dim_x*self.spatial_dim_y, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, batch_norm):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, batch_norm))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool2d(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class PreActBlock(nn.Module):
    """ From https://github.com/kuangliu/pytorch-cifar/blob/master/models/preact_resnet.py
    Pre-activation version of the BasicBlock. """
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, batch_norm=1e-5):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes, batch_norm)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, batch_norm)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class WRN(nn.Module):
    """ WRN-n-k denotes a residual network that has a total number of convolutional layers n and a widening factor k
    (for example, network with 40 layers and k = 2 times wider than original would be denoted
    as WRN-40-2). Also, when applicable we append block type, e.g. WRN-40-2-B(3, 3)

    https://github.com/szagoruyko/wide-residual-networks/blob/master/pytorch/resnet.py
    """

    def __init__(self, num_classes, num_channels, args):
        super(WRN, self).__init__()
        self.args = args
        self.num_channels = num_channels
        self.in_planes = 16

        block = PreActBlock
        # WRN28-10
        num_blocks = (28 - 4)//6
        widening_factor = 10

        self.conv1 = nn.Conv2d(num_channels, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 16*widening_factor, num_blocks, stride=1, batch_norm=args.batch_norm)
        self.layer2 = self._make_layer(block, 32*widening_factor, num_blocks, stride=2, batch_norm=args.batch_norm)
        self.layer3 = self._make_layer(block, 64*widening_factor, num_blocks, stride=2, batch_norm=args.batch_norm)
        self.bn4 = nn.BatchNorm2d(64*widening_factor, eps=args.batch_norm)
        self.relu4 = nn.ReLU(inplace=True)
        self.avg_pool2d = nn.AvgPool2d(4)

        convs = nn.Sequential(self.conv1, self.layer1, self.layer2, self.layer3,
                                   self.avg_pool2d)

        # getting the flattened input dimensionality
        self.channels, self.spatial_dim_x, self.spatial_dim_y = get_feat_size(
            convs, args.patch_size, num_channels)

        self.linear = nn.Linear(self.channels*self.spatial_dim_x*self.spatial_dim_y, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, batch_norm):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, batch_norm))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.relu4(self.bn4(out))
        out = self.avg_pool2d(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def calc_gpu_mem_req(model, batch_size, patch_size, num_channels):
    """ This method does not always work, unfortunately. Probably due to CUDNN extra memory requirements:
     https://pytorch.org/docs/stable/notes/randomness.html """

    # class attribute for storing total gpu memory requirement of the model
    # (4 bytes/ 32 bits per floating point no.)
    gpu_mem_req = 32 * batch_size * num_channels * patch_size * patch_size

    image_size = patch_size

    for name, layer in model.named_modules():

            if isinstance(layer, nn.Conv2d):
                # example: Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                in_channel = int(str(layer).split('(')[1].split(',')[0])
                out_channel = int(str(layer).split('(')[1].split(',')[1])

                filter_size = int(str(layer).split('kernel_size=')[1].lstrip('(').rstrip(')').split(',')[0])

                try:
                    stride = int(str(layer).split('stride=')[1].lstrip('(').rstrip(')').split(',')[0])
                except Exception:
                    stride = 1
                try:
                    padding = int(str(layer).split('padding=')[1].lstrip('(').rstrip(')').split(',')[0])
                except Exception:
                    padding = 0

                # resulting img_size after convoltion: (W – F + 2P) / S + 1
                image_size = (image_size - filter_size + 2 * padding) / stride + 1

                # gpu memory requirement for conv layer due to layer parameters (batchnorm parameters have been
                # ignored)
                gpu_mem_req += 32 * in_channel * out_channel * filter_size * filter_size

                # gpu memory requirement for conv layer due to layer feature output
                gpu_mem_req += 32 * batch_size * image_size * image_size * out_channel

            elif isinstance(layer, nn.MaxPool2d) or isinstance(layer, nn.AvgPool2d):

                # example: AvgPool2d(kernel_size=1, stride=1, padding=0)
                in_channel = out_channel
                out_channel = out_channel  # num_channels remains the same in max/avg pooling

                filter_size = int(str(layer).split('kernel_size=')[1].lstrip('(').rstrip(')').split(',')[0])
                try:
                    stride = int(str(layer).split('stride=')[1].lstrip('(').rstrip(')').split(',')[0])
                except Exception:
                    stride = 1
                try:
                    padding = int(str(layer).split('padding=')[1].lstrip('(').rstrip(')').split(',')[0])
                except Exception:
                    padding = 0

                # resulting img_size after convoultion: (W – F + 2P) / S + 1
                image_size = (image_size - filter_size + 2 * padding) / stride + 1

                # gpu memory requirement for conv layer due to layer parameters (batchnorm parameters have been
                # ignored)
                gpu_mem_req += 32 * in_channel * out_channel * filter_size * filter_size

                # gpu memory requirement for conv layer due to layer feature output
                gpu_mem_req += 32 * batch_size * image_size * image_size * out_channel

            elif isinstance(layer, nn.Linear):

                in_feature = int(str(layer).split('in_features=')[1].lstrip('(').rstrip(')').split(',')[0])
                no_feature = int(str(layer).split('out_features=')[1].lstrip('(').rstrip(')').split(',')[0])

                # gpu memory requirement for FC layer due to layer parameters
                # (batchnorm parameters have been ignored)
                gpu_mem_req += 32 * batch_size * no_feature

                # gpu memory requirement for FC layer due to layer feature output
                gpu_mem_req += 32 * in_feature * no_feature

    # converting bits to GB: byte->kilo->mega->giga
    gpu_mem_req /= (8. * 1024 * 1024 * 1024)
    return gpu_mem_req