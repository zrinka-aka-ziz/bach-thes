import comet_ml
import torch
import torch.nn as nn
import torch.nn.functional as functional
from tqdm import tqdm

from configure import Config
config = Config()

torch.manual_seed(config.seed_value) # cpu  vars
torch.cuda.manual_seed(config.seed_value)
torch.cuda.manual_seed_all(config.seed_value) # gpu vars
torch.backends.cudnn.deterministic = True  #needed
torch.backends.cudnn.benchmark = False

if config.extension=="model5":
    class ConvBnRelu(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size, padding, stride):
            super(ConvBnRelu, self).__init__()
            self.conv = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Dropout(config.drop)
            )

        def forward(self, x):
            x = self.conv(x)
            return x
else:
    class ConvBnRelu(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size, padding, stride):
            super(ConvBnRelu, self).__init__()
            self.conv = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
            )

        def forward(self, x):
            x = self.conv(x)
            return x
		
class StackEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(StackEncoder, self).__init__()
        self.conv1 = ConvBnRelu(in_channels, out_channels, kernel_size=(3, 3), stride=1, padding=1)
        self.conv2 = ConvBnRelu(out_channels, out_channels, kernel_size=(3, 3), stride=1, padding=1)
        self.maxPool = nn.MaxPool2d(kernel_size=(2, 2), stride=2, ceil_mode=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x_trace = x
        x = self.maxPool(x)
        return x, x_trace

if config.extension=="model4":
    class StackDecoder(nn.Module):
        def __init__(self, in_channels, out_channels):
            super(StackDecoder, self).__init__()
            self.upSample = nn.Sequential(
                                    nn.Upsample(scale_factor=2, mode='bilinear'),
                                    nn.Conv2d(in_channels, out_channels, kernel_size=(1,1)),
            )
            self.conv1 = ConvBnRelu(in_channels, out_channels, kernel_size=(3, 3), stride=1, padding=1)
            #Crop + concat step between these 2
            self.conv2 = ConvBnRelu(out_channels, out_channels, kernel_size=(3, 3), stride=1, padding=1)

        def crop_concat(self, upsampled, bypass):
            #Crop y to the (h, w) of x and concat them. Used for the expansive path.
            #Returns the concatenated tensor
        
            c = (bypass.size()[2] - upsampled.size()[2]) // 2
            bypass = functional.pad(bypass, (-c, -c, -c, -c))

            return torch.cat((upsampled, bypass), 1)

        def forward(self, x, down_tensor):
            x = self.upSample(x)
            x = self.crop_concat(x, down_tensor)
            x = self.conv1(x)
            x = self.conv2(x)
            return x		
else:
    class StackDecoder(nn.Module):
        def __init__(self, in_channels, out_channels):
            super(StackDecoder, self).__init__()
            self.upSample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(2, 2), stride=2, padding=0)
            self.conv1 = ConvBnRelu(in_channels, out_channels, kernel_size=(3, 3), stride=1, padding=1)
            #Crop + concat step between these 2
            self.conv2 = ConvBnRelu(out_channels, out_channels, kernel_size=(3, 3), stride=1, padding=1)

        def crop_concat(self, upsampled, bypass):
            #Crop y to the (h, w) of x and concat them. Used for the expansive path.
            #Returns the concatenated tensor
        
            c = (bypass.size()[2] - upsampled.size()[2]) // 2
            bypass = functional.pad(bypass, (-c, -c, -c, -c))

            return torch.cat((upsampled, bypass), 1)

        def forward(self, x, down_tensor):
            x = self.upSample(x)
            x = self.crop_concat(x, down_tensor)
            x = self.conv1(x)
            x = self.conv2(x)
            return x

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.down1 = StackEncoder(config.channels, 8)
        self.down2 = StackEncoder(8, 16)
        self.down3 = StackEncoder(16, 32)
        self.down4 = StackEncoder(32, 64)

        self.center = nn.Sequential(
            ConvBnRelu(64, 128, kernel_size=(3, 3), stride=1, padding=1),
            ConvBnRelu(128, 128, kernel_size=(3, 3), stride=1, padding=1)
        )

        self.up1 = StackDecoder(in_channels=128, out_channels=64)
        self.up2 = StackDecoder(in_channels=64, out_channels=32)
        self.up3 = StackDecoder(in_channels=32, out_channels=16)
        self.up4 = StackDecoder(in_channels=16, out_channels=8)

        # 1x1 convolution at the last layer
        # Different from the paper is the output size here
        self.output_seg_map = nn.Conv2d(8, 1, kernel_size=(1, 1), padding=0, stride=1)

    def forward(self, x):
        x, x_trace1 = self.down1(x)  # Calls the forward() method of each layer
        x, x_trace2 = self.down2(x)
        x, x_trace3 = self.down3(x)
        x, x_trace4 = self.down4(x)

        x = self.center(x)

        x = self.up1(x, x_trace4)
        x = self.up2(x, x_trace3)
        x = self.up3(x, x_trace2)
        x = self.up4(x, x_trace1)

        out = self.output_seg_map(x)
        return torch.sigmoid(out)
