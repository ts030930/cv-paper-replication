class ResNet34(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv_block_01 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.stage_01 = self.make_basic_stage(in_channels=64, last_channels=64, stride=1, blocks=3)
        self.stage_02 = self.make_basic_stage(in_channels=64, last_channels=128, stride=2, blocks=4)
        self.stage_03 = self.make_basic_stage(in_channels=128, last_channels=256, stride=2, blocks=6)
        self.stage_04 = self.make_basic_stage(in_channels=256, last_channels=512, stride=2, blocks=3)

        self.adaptive_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc = nn.Linear(512, num_classes)

    def make_basic_stage(self, in_channels, last_channels, stride, blocks):
        layers = []
        downsample = None
        if stride != 1:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=last_channels, 
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_features=last_channels)
            )
        layers.append(BasicBlock(in_channels=in_channels, last_channels=last_channels,
                                     stride=stride, downsample=downsample))
        for _ in range(1, blocks):
            layers.append(BasicBlock(in_channels=last_channels, last_channels=last_channels, 
                                    stride=1, downsample=None))
        
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_block_01(x)
        x = self.stage_01(x)
        x = self.stage_02(x)
        x = self.stage_03(x)
        x = self.stage_04(x)
        x = self.adaptive_pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)

        return x
