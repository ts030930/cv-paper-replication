import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
# Conv -> BN -> Relu 연속 적용. 
class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,  
                              kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

class InceptionBlock(nn.Module):
    
    def __init__(self, in_channels, ch_1x1, ch_3x3_reduce, ch_3x3, ch_5x5_reduce, ch_5x5, pool_prj): 
        super().__init__()
        # 첫번째 1x1 Conv branch
        self.branch1 = BasicConv2d(in_channels, ch_1x1, kernel_size=1)
        # 3x3 적용 전 1x1 conv -> 3x3 Conv
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch_3x3_reduce, kernel_size=1), 
            BasicConv2d(ch_3x3_reduce, ch_3x3, kernel_size=3, padding=1))
        # 5x5 적용 전 1x1 Conv -> 3x3 Conv. 
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch_5x5_reduce, kernel_size=1),
            BasicConv2d(ch_5x5_reduce, ch_5x5, kernel_size=5, padding=1))
        # Max Pooling 후 1x1 적용
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
            BasicConv2d(in_channels, pool_prj, kernel_size=1))

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        # feature map을 채널 기준으로 Concat 적용. 
        return torch.cat([branch1, branch2, branch3, branch4], 1)


class AuxClassifier(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((4,4))
        self.conv1 = BasicConv2d(in_channels, 128, kernel_size=1)
        self.fc = nn.Sequential(
            nn.Linear(4*4*128, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.7),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.avgpool(x)
        x = self.conv1(x)
        x = torch.flatten(x, 1)
        return self.fc(x)




class GoogLeNet(nn.Module):
    def __init__(self, num_classes=1000, aux_logits=True, dropout=0.2):
        super().__init__()

        self.aux_logits = aux_logits
        self.training = True
        
        self.conv1 = BasicConv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.conv2 = BasicConv2d(64, 64, kernel_size=1)
        self.conv3 = BasicConv2d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        
        # InceptionBlock 생성자(in_channels, ch1x1, ch3x3_reduce, ch3x3, ch5x5_reduce, ch5x5, pool)
        # 처음 2개의 Inception Block
        self.inception3a = InceptionBlock(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = InceptionBlock(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        
        # InceptionBlock 생성자(in_channels, ch1x1, ch3x3_reduce, ch3x3, ch5x5_reduce, ch5x5, pool)
        # 5개의 Inception Block
        self.inception4a = InceptionBlock(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = InceptionBlock(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = InceptionBlock(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = InceptionBlock(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = InceptionBlock(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        # InceptionBlock 생성자(in_channels, ch1x1, ch3x3_reduce, ch3x3, ch5x5_reduce, ch5x5, pool)
        # 5개의 Inception Block
        self.inception5a = InceptionBlock(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = InceptionBlock(832, 384, 192, 384, 48, 128, 128)

        if aux_logits:
            self.aux1 = AuxClassifier(512, num_classes)
            self.aux2 = AuxClassifier(528, num_classes)
        else:
            self.aux1 = None
            self.aux2 = None

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool2(x)
        
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)

        x = self.inception4a(x)
        # 보조 classifier 출력
        if self.training:
            out1 = self.aux1(x)

        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        # 보조 classifier 출력
        if self.training:
            out2 = self.aux2(x)

        x = self.inception4e(x)
        x = self.maxpool4(x)

        x = self.inception5a(x)
        x = self.inception5b(x)

        x = self.avgpool(x)

        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)                
        if self.training:
            return [x, out1, out2] 
        else:
            return x

def set_train(self):
        self.training = True


googlenet_model = GoogLeNet(num_classes=1000, aux_logits=True, dropout=0.4)
summary(model=googlenet_model, input_size=(1, 3, 224, 224),
        col_names=['input_size', 'output_size', 'num_params'], 
        row_settings=['var_names'])
