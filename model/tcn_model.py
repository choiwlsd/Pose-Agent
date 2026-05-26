import torch 
import torch.nn as nn

class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        # padding: 인과적 패딩(미래 데이터 안봄)
        padding = (kernel_size - 1) * dilation

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size,
                        dilation=dilation, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # 잔차 연결을 위한 1x1 컨볼루션
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()
        
    def forward(self, x):
        # 인과적 패딩으로 인해 뒤에 남은 부분 잘라냄
        out = self.conv(x)[:, :, :x.size(2)]
        return torch.relu(out + self.downsample(x))
    

class TCN(nn.Module):
    def __init__(self, input_size=6, num_classes=2):
        super().__init__()
        self.tcn = nn.Sequential(
            TCNBlock(input_size, 32, kernel_size=3, dilation=1),
            TCNBlock(32, 64, kernel_size=3, dilation=2),
            TCNBlock(64, 64, kernel_size=3, dilation=4)
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # 시퀀스 길이 상관없이 평균 풀링
            nn.Flatten(),             # (batch_size, 64)
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        # x: (batch_size, input_size, seq_len)
        x = x.permute(0, 2, 1)  # (batch_size, seq_len, input_size) → (batch_size, input_size, seq_len)
        x = self.tcn(x)          # (batch_size, 64, seq_len)
        return self.classifier(x) # (batch_size, num_classes)
    
    