import torch
import torchvision.ops
from torch import nn

class DeformableConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1):

        super(DeformableConv2d, self).__init__()

        self.padding = kernel_size//2
        
        self.offset_conv = nn.Conv2d(in_channels, 
                                     2 * kernel_size * kernel_size,
                                     kernel_size=kernel_size, 
                                     padding=self.padding, 
                                     stride=stride,
                                     bias=True)

        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)
        # self.offset_conv.register_backward_hook(self._set_lr)
        
        self.modulator_conv = nn.Conv2d(in_channels, 
                                     1 * kernel_size * kernel_size,
                                     kernel_size=kernel_size, 
                                     padding=self.padding, 
                                     stride=stride,
                                     bias=True)

        nn.init.constant_(self.modulator_conv.weight, 0.)
        nn.init.constant_(self.modulator_conv.bias, 0.)
        # self.modulator_conv.register_backward_hook(self._set_lr)
        
        self.regular_conv = nn.Conv2d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=self.padding,
                                      bias=True)
    
    
    def _set_lr(self, module, grad_input, grad_output):
        pass
        # grad_input = (grad_input[i] * 0.01 for i in range(len(grad_input)))
        # grad_output = (grad_output[i] * 0.01 for i in range(len(grad_output)))
   
    
    def forward(self, x):
        h, w = x.shape[2:]
        max_length = max(h, w)/2.
        
        offset = self.offset_conv(x).clamp(-max_length, max_length)        
        modulator = torch.sigmoid(self.modulator_conv(x))
        
        x = torchvision.ops.deform_conv2d(input=x, 
                                          offset=offset, 
                                          weight=self.regular_conv.weight, 
                                          bias=self.regular_conv.bias, 
                                          padding=self.padding)
        return x
