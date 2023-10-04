import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from .extractor import SetConv
except ImportError:
    from extractor import SetConv

class MotionEncoder(nn.Module):
    def __init__(self):
        super(MotionEncoder, self).__init__()
        self.conv_flow = nn.Conv1d(3, 64-3, 1)

    def forward(self, flow):
        flo = F.relu(self.conv_flow(flow.transpose(1, 2).contiguous()))
        out = torch.cat([flo, flow.transpose(1, 2).contiguous()], dim=1) 
        # b,64,n
        return out
    
class ConvGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv1d(input_dim+hidden_dim, hidden_dim, 1)
        self.convr = nn.Conv1d(input_dim+hidden_dim, hidden_dim, 1)
        self.convq = nn.Conv1d(input_dim+hidden_dim, hidden_dim, 1)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)

        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        rh_x = torch.cat([r*h, x], dim=1)
        q = torch.tanh(self.convq(rh_x))

        h = (1 - z) * h + z * q
        return h
    
class FlowHead(nn.Module):
    def __init__(self, input_dim):
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, 64, 1)
        self.setconv = SetConv(64, 64)
        self.out_conv = nn.Sequential(
            nn.Conv1d(128, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 3, 1)
        )

    def forward(self, x, graph):
        # x: b,c,n
        out = self.conv1(x)
        out_setconv = self.setconv(x.transpose(1, 2).contiguous(), graph).transpose(1, 2).contiguous()
        out = self.out_conv(torch.cat([out_setconv, out], dim=1))
        return out
    
# class Neural_Prior(torch.nn.Module):
#     def __init__(self, dim_x=3, filter_size=128, act_fn='relu', layer_size=8):
#         super().__init__()
#         self.layer_size = layer_size
        
#         self.nn_layers = torch.nn.ModuleList([])
#         # input layer (default: xyz -> 128)
#         if layer_size >= 1:
#             self.nn_layers.append(torch.nn.Sequential(torch.nn.Linear(dim_x, filter_size)))
#             if act_fn == 'relu':
#                 self.nn_layers.append(torch.nn.ReLU())
#             elif act_fn == 'sigmoid':
#                 self.nn_layers.append(torch.nn.Sigmoid())
#             for _ in range(layer_size-1):
#                 self.nn_layers.append(torch.nn.Sequential(torch.nn.Linear(filter_size, filter_size)))
#                 if act_fn == 'relu':
#                     self.nn_layers.append(torch.nn.ReLU())
#                 elif act_fn == 'sigmoid':
#                     self.nn_layers.append(torch.nn.Sigmoid())
#             self.nn_layers.append(torch.nn.Linear(filter_size, dim_x))
#         else:
#             self.nn_layers.append(torch.nn.Sequential(torch.nn.Linear(dim_x, dim_x)))

#     def forward(self, x):
#         """ points -> features
#             [B, N, 3] -> [B, K]
#         """
#         for layer in self.nn_layers:
#             x = layer(x)
                
#         return x
    

if __name__ == "__main__":
    flow=torch.rand((2,2048,3)).cuda()
    motion_encoder=MotionEncoder().cuda()
    out=motion_encoder(flow)
    print(out.shape)