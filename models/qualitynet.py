import torch.nn as nn

class QualityNet(nn.Module):
    def init(self):
        super(QualityNet, self).init()
        self.net = nn.Sequential(
            nn.Conv2d(5, 24, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(24, 12, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(12, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)  # Output: attention weights K