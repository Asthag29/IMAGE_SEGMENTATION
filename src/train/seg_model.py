# %%
import torch
from torch import nn
from dataloaders import TrainSegmentationDataloader , TestSegmentationDataloader

# %%
class FishSegmentation(nn.Module):
    def __init__(self):
        super(FishSegmentation, self).__init__()
        
        # Helper function for conv blocks
        def conv_block(in_channels, out_channels):   #using two conv2d layer increases the receptive field
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )
        
        # Encoder
        self.encoder1 = conv_block(1, 64)
        self.encoder2 = conv_block(64, 128)
        self.encoder3 = conv_block(128, 256)
        
        # Bottleneck
        self.bottleneck = conv_block(256, 512)
        
        # Decoder
        self.decoder1 = conv_block(512, 256)
        self.decoder2 = conv_block(256, 128)
        self.decoder3 = conv_block(128, 64)
        
        # Pooling and upsampling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upconv1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)  #upsampling
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        
        # Final layer
        self.final = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc3))
        
        # Decoder with skip connections
        dec1 = self.decoder1(torch.cat([self.upconv1(bottleneck), enc3], dim=1))
        dec2 = self.decoder2(torch.cat([self.upconv2(dec1), enc2], dim=1))
        dec3 = self.decoder3(torch.cat([self.upconv3(dec2), enc1], dim=1))
        
        return self.final(dec3)




# %%

Network= FishSegmentation()

x = torch.randn(60, 1, 224, 224)  # [batch, channels, height, width]
output = Network(x)
print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
print(f"Output min: {output.min().item():.4f}")
print(f"Output max: {output.max().item():.4f}")

print(f"Model parameters: {sum(p.numel() for p in Network.parameters()):,}")
# %%
