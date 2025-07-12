import torch
from torch import Tensor
from torch import nn


class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            #Conv1 Layer
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4,padding=2),
            nn.ReLU(True),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2), # LRN은 논문에 표기된 값 사용
            nn.MaxPool2d(kernel_size=3, stride=2),

            #Conv2 Layer
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1,padding=2),
            nn.ReLU(True),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),

            #Conv3 Layer
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1,padding=1),
            nn.ReLU(True),

            #Conv4 Layer
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1,padding=1),
            nn.ReLU(True),

            #Conv5 Layer
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1,padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=2) # Conv5 뒤에는 Max Pooling 추가한다고 논문에 기재되어있음
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 1000)
        )

    
    def forward(self,x):
        out = self.features(x)
        out = self.avgpool(out)
        out = torch.flatten(out, start_dim=1)
        out = self.classifier(out)
        return out
