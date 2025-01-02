import torch
import torch.nn as nn
import torch.nn.init as I


class VGGNet(nn.Module):
    """Deep neural network modelled after the VGG-16 architecture."""

    def __init__(self):
        super(VGGNet, self).__init__()
        
        # Feature extraction block
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 64, 3, padding=1),
            # nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            # nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Output size: N x 112 x 112 x 64

            # Block 2
            nn.Conv2d(64, 128, 3, padding=1),
            # nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            # nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Output size: N x 56 x 56 x 128

            # Block 3
            nn.Conv2d(128, 256, 3, padding=1),
            # nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            # nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            # nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Output size: N x 28 x 28 x 256

            # Block 4
            nn.Conv2d(256, 512, 3, padding=1),
            # nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            # nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            # nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Output size: N x 14 x 14 x 512

            # Block 5
            nn.Conv2d(512, 512, 3, padding=1),
            # nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            # nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            # nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Output size: N x 7 x 7 x 512
        )

        # MLP block for regression output
        self.mlp = nn.Sequential(
            nn.Linear(7 * 7 * 512, 4096, bias=True),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Dropout1d(0.5, inplace=False),
            nn.Linear(4096, 4096, bias=True),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Dropout1d(0.5, inplace=False),
            nn.Linear(4096, 136, bias=True),
        )  # Output size: N x 136

        # Add weights initialisation
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                I.xavier_normal_(m.weight)
                if m.bias is not None:
                    I.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                I.constant_(m.weight, 1)
                I.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                I.xavier_normal_(m.weight)
                I.constant_(m.bias, 0)

        
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.mlp(x)
        return x


class Net(nn.Module):
    """Single deep neural network for keypoint recognition."""

    def __init__(self):
        super(Net, self).__init__()

        # Feature extraction block
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, 5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )  # Output size: N x 3 x 3 x 256

        # MLP block for regression output
        self.mlp = nn.Sequential(
            nn.Linear(2304, 1024, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2, inplace=False),
            nn.Linear(1024, 136, bias=True),
        )  # Output size: N x 136

        # Add weights initialisation
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                I.xavier_normal_(m.weight)
                if m.bias is not None:
                    I.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                I.xavier_normal_(m.weight)
                I.constant_(m.bias, 0)
            

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.mlp(x)
        return x