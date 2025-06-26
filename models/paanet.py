import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers import DoubleConv, Down, Up, OutConv


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=1):
        super(ChannelAttention, self).__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)

        self.conv1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(in_channels // reduction)

        self.conv2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(in_channels)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x1 = self.global_avgpool(x) # Shape: (B, C, 1, 1)  

        x1 = self.conv1(x1) # Shape: (B, C//reduction, 1, 1)  
        x1 = self.bn1(x1) # Shape: (B, C//reduction, 1, 1)  

        x1 = self.conv2(x1) # Shape: (B, C, 1, 1)  
        x1 = self.bn2(x1) # Shape: (B, C, 1, 1)  

        attention_map = self.softmax(x1) # Shape: (B, C, 1, 1) 

        return attention_map


class PositionAttention(nn.Module):
    def __init__(self, in_channels, reduction=2):
        super(PositionAttention, self).__init__()
        
        self.conv_U = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        self.conv_V = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)

        self.relu = nn.ReLU()

        self.conv_out = nn.Conv2d(1, in_channels, kernel_size=1)

    def forward(self, x):
        # U_x and V_y are of shape C/2 x H x W
        U_x = self.conv_U(x) # Shape: (B, C//reduction, H, W)  
        V_y = self.conv_V(x) # Shape: (B, C//reduction, H, W) 

        U_x = self.relu(U_x) # Shape: (B, C//reduction, H, W) 
        V_y = self.relu(V_y) # Shape: (B, C//reduction, H, W)

        # after softmax on each spatial position
        softmax_V_y = F.softmax(V_y, dim=2) # Softmax along H dimension, Shape: (B, C//reduction, H, W)  
        softmax_V_y = F.softmax(softmax_V_y, dim=3) # Softmax along W dimension, Shape: (B, C//reduction, H, W)  

        # element wise multiplication
        attention_map = U_x * softmax_V_y # Shape: (B, C//reduction, H, W)  

        # sum over spatial positions / channels 
        attention_map = attention_map.sum(dim=1, keepdim=True) # Shape: (B, 1, H, W)  

        attention_map = self.conv_out(attention_map)  # Shape: (B, C, H, W)

        return attention_map


class DoubleAttentionModule(nn.Module):
    def __init__(self, in_channels, reduction_channel=1, reduction_position=2):  
        super(DoubleAttentionModule, self).__init__()  
        self.channel_attention = ChannelAttention(in_channels, reduction_channel)  
        self.position_attention = PositionAttention(in_channels, reduction_position)  

        self.conv_theta = nn.Conv2d(in_channels, in_channels, kernel_size=1) 
        self.bn_theta = nn.BatchNorm2d(in_channels)
        self.relu_theta = nn.ReLU() 

        self.conv_recalibration = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.bn_recalibration = nn.BatchNorm2d(in_channels)  
        self.relu_recalibration = nn.ReLU()
  
    def forward(self, x):  
        Ap = self.position_attention(x)  # Shape: [B, C, H, W]  
        Ac = self.channel_attention(x)   # Shape: [B, C, 1, 1]  
          
        # Reshape Ac to be broadcastable with Ap  
        Ac = Ac.view(Ac.size(0), Ac.size(1), 1, 1)  
          
        # Multiply position and channel attention maps  
        As = Ap * Ac  # Shape: [B, C, H, W]  

        # Recalibration step  
        theta_x = self.conv_theta(x)  # Shape: [B, C, H, W]  
        theta_x = self.bn_theta(theta_x)  
        theta_x = self.relu_theta(theta_x)  

        As_prime = As * theta_x  # Element-wise multiplication, Shape: [B, C, H, W]

        # Further transformation  
        As_prime = self.conv_recalibration(As_prime)  # Shape: [B, C, H, W]  
        As_prime = self.bn_recalibration(As_prime)  
        As_prime = self.relu_recalibration(As_prime)

        # Element-wise addition with input  
        output = x + As_prime  # Shape: [B, C, H, W]  
          
        return output


class PyramidUpsamplingModule(nn.Module):
    def __init__(self, in_channels, reduction=4):  
        super(PyramidUpsamplingModule, self).__init__()  

        self.conv = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1) 

    def forward(self, x, target_h, target_w):
        # Reduce dimensions with 1x1 convolution  
        x = self.conv(x)  # Shape: (B, out_channels, H, W)  

        # Upsample to the size of the input image  
        x = F.interpolate(x, size=(target_h, target_w), mode='bilinear', align_corners=True)  # Shape: (B, out_channels, target_h, target_w)  

        return x  
              

class PAANet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(PAANet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))

        # double attention modules corresponding to each decoder block
        self.att1 = DoubleAttentionModule(1024 // factor)
        self.att2 = DoubleAttentionModule(512)
        self.att3 = DoubleAttentionModule(256)
        self.att4 = DoubleAttentionModule(128)

        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))

        # pyramid upsampling modules for each attentive feature map  
        self.pum1 = PyramidUpsamplingModule(1024 // factor, 4)  
        self.pum2 = PyramidUpsamplingModule(512, 4)  
        self.pum3 = PyramidUpsamplingModule(256, 4)  
        self.pum4 = PyramidUpsamplingModule(128, 4) 

        self.outc = (OutConv(1024 // factor // 4 + 512 // 4 + 256 // 4 + 128 // 4, n_classes))

        print("PAANet Model Created")

    def forward(self, x):
        input_h, input_w = x.size(2), x.size(3)  
          
        x1 = self.inc(x)        # (N, 64, H, W)  
        x2 = self.down1(x1)     # (N, 128, H/2, W/2)  
        x3 = self.down2(x2)     # (N, 256, H/4, W/4)  
        x4 = self.down3(x3)     # (N, 512, H/8, W/8)  
        x5 = self.down4(x4)     # (N, 1024 // factor, H/16, W/16)  
  
        x5 = self.att1(x5)      # (N, 1024 // factor, H/16, W/16)  
        x5_p = self.pum1(x5, input_h, input_w)    # (N, 1024 // factor // 4, H, W)  
  
        x = self.up1(x5, x4)    # (N, 512, H/8, W/8)  
        x = self.att2(x)        # (N, 512, H/8, W/8)  
        x4_p = self.pum2(x, input_h, input_w)     # (N, 512 // 4, H, W)  
  
        x = self.up2(x, x3)     # (N, 256, H/4, W/4)  
        x = self.att3(x)        # (N, 256, H/4, W/4)  
        x3_p = self.pum3(x, input_h, input_w)     # (N, 256 // 4, H, W)  
  
        x = self.up3(x, x2)     # (N, 128, H/2, W/2)  
        x = self.att4(x)        # (N, 128, H/2, W/2)  
        x2_p = self.pum4(x, input_h, input_w)     # (N, 128 // 4, H, W)   
  
        # Concatenate the pyramid upsampled features  
        x_p = torch.cat([x5_p, x4_p, x3_p, x2_p], dim=1)  # (N, sum(out_channels), H, W) 
    
        logits = self.outc(x_p)  # (N, n_classes, H, W)  
        return logits  


if __name__ == "__main__":
    # Example usage  
    n_channels = 3  
    n_classes = 14  
    model = PAANet(n_channels, n_classes)  
    x = torch.rand(4, n_channels, 270, 480)  # (batch_size, channels, height, width)  
    output = model(x)  
    print(output.shape)  # Expected output shape: (4, n_classes, 256, 256) 