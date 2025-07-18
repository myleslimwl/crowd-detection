import torch.nn as nn

class RegNet(nn.Module):
    def init(self):
        super(RegNet, self).init()
        self.net = nn.Sequential(
            nn.Conv2d(3, 20, kernel_size=7, padding=3), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(20, 40, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(40, 20, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv2d(20, 10, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv2d(10, 1, kernel_size=1), nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)