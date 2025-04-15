import torch
import torch.nn as nn
import torchvision
from scipy.stats import entropy
import numpy as np

class BasicConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    
class ResNet18(nn.Module):
    # Keep in mind that you will need to resize the Image to 224x224
    def __init__(self):
        super().__init()
        self.backbone = torchvision.models.resnet18()
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = torch.nn.Linear(num_ftrs, 10)
    def forward(self, x):
        return self.backbone(x)


class MLP(nn.Module):
    def __init__(self, num_layers=7, size=2048, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(3072, size)
        self.hidden = self.hidden = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        self.out = nn.Linear(size, num_classes)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.fc1(x)
        for layer in self.hidden:
            x = layer(x)
            x = self.relu(x)
        x = self.out(x)
        return x
        
        
import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out
    
class EarlyExitBlock(nn.Module):
    def __init__(self, input_features, num_classes=6):
        super(EarlyExitBlock, self).__init__()
        self.flatten = nn.Flatten()
        self.exit = nn.Linear(input_features, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        return self.exit(x)

class EarlyExitResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=6):
        super(EarlyExitResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)
        self.early_exits = nn.ModuleList([
            EarlyExitBlock(65536, num_classes),
            EarlyExitBlock(32768, num_classes),
            EarlyExitBlock(16384, num_classes)
        ])


    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def early_train(self, x):
        outputs = []
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        outputs.append(self.early_exits[0](out))
        out = self.layer2(out)
        outputs.append(self.early_exits[1](out))
        out = self.layer3(out)
        outputs.append(self.early_exits[2](out))
        out = self.layer4(out)
        out = nn.functional.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        outputs.append(out)
        return outputs
    
    def calculate_entropy(self, logits):
        s_y = nn.functional.softmax(logits, dim=1)
        return np.mean(entropy(s_y.cpu().numpy(), axis=1))
        
    

    def forward(self, x, entropy_tol, exit_early=True):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        if exit_early:
            logits = self.early_exits[0](out)
            curr_entropy = self.calculate_entropy(logits)
            if curr_entropy < entropy_tol:
                x.detach()
                return logits, 0, curr_entropy
        out = self.layer2(out)
        if exit_early:
            logits = self.early_exits[1](out)
            curr_entropy = self.calculate_entropy(logits)
            if curr_entropy < entropy_tol:
                x.detach()
                return logits, 1, curr_entropy
        out = self.layer3(out)
        if exit_early:
            logits = self.early_exits[2](out)
            curr_entropy = self.calculate_entropy(logits)
            if curr_entropy < entropy_tol:
                x.detach()
                return logits, 2, curr_entropy
        out = self.layer4(out)
        out = nn.functional.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        if exit_early:
            curr_entropy = self.calculate_entropy(out)
            return out, 3, curr_entropy
        return out

def EarlyExitResNet18():
    return EarlyExitResNet(BasicBlock, [2, 2, 2, 2])
