import torch
import torch.nn as nn

Img_Channels = 3

class VGG16(nn.Module):
    def __init__(self,num_classes=10):
        super().__init__()
        # VGG16은 conv 2회 -> 2회 -> 3회 -> 3회 -> 3회로 layer 형성되어있음
        
        self.Conv1 = nn.Sequential(
            nn.Conv2d(in_channels=Img_Channels, out_channels=64,kernel_size =3,stride = 1, padding ='same'),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64,kernel_size =3,stride = 1, padding ='same'),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2) # 각 Conv layer 마지막에 Max Pooling 적용
        )
        
        self.Conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128,kernel_size =3,stride = 1, padding ='same'),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128,kernel_size =3,stride = 1, padding ='same'),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
    
        self.Conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256,kernel_size =3,stride = 1, padding ='same'),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256,kernel_size =3,stride = 1, padding ='same'),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256,kernel_size =3,stride = 1, padding ='same'),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.Conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512,kernel_size =3,stride = 1, padding ='same'),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512,kernel_size =3,stride = 1, padding ='same'),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512,kernel_size =3,stride = 1, padding ='same'),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.Conv5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512,kernel_size =3,stride = 1, padding ='same'),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512,kernel_size =3,stride = 1, padding ='same'),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512,kernel_size =3,stride = 1, padding ='same'),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        # 이미지가 224x224가 아닌 다른 사이즈가 왔을때를 위해서 adaptiveAvgPooling 적용
        self.adaptive_pool = nn.AdaptiveAvgPool2d(output_size=(7,7))
              
        self.classifier =  nn.Sequential(
            nn.Linear(in_features=25088, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=num_classes, bias=True)
        )

    def forward(self,x):
        out = self.Conv1(x)
        out = self.Conv2(out)
        out = self.Conv3(out)
        out = self.Conv4(out)
        out = self.Conv5(out)
        out = self.adaptive_pool(out)
        out = torch.flatten(out,start_dim=1)
        out = self.classifier(out)

        return out
