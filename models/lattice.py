import os
import sys
import numpy as np
import time
import torch

from knn_cuda import KNN

# add path
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_dir not in sys.path:
    sys.path.append(project_dir)


def fetch_knn_points(point_cloud, knn_indices):
        """
        Fetch the K nearest neighbor points from a point cloud using PyTorch's torch.gather.

        Args:
        - point_cloud (torch.Tensor): A tensor representing the point cloud with shape [b, n, 3].
        - knn_indices (torch.Tensor): A tensor representing the K nearest neighbor indices with shape [b, n, k].

        Returns:
        - knn_points (torch.Tensor): A tensor containing the K nearest neighbor points with shape [b, n, k, 3].
        """

        b, n, k = knn_indices.shape
        knn_indices=knn_indices.unsqueeze(-1).expand(-1,-1,-1,3)
        point_cloud=point_cloud.unsqueeze(2).expand(-1,-1,k,-1)
        knn_points=torch.gather(point_cloud,1,knn_indices)

        return knn_points

class Lattice(torch.nn.Module):
    def __init__(self,nb,spacing,batchsize,xmin=0,ymin=0,zmin=0,xmax=2*np.pi,ymax=2*np.pi,zmax=2*np.pi,device="cuda") -> None:
        super(Lattice, self).__init__()
        self.nb = nb
        self.xmin = xmin
        self.ymin = ymin
        self.zmin = zmin
        self.xmax = xmax
        self.ymax = ymax
        self.zmax = zmax
        self.spacing = spacing

        # create a grid 
        x = torch.linspace(self.xmin,self.xmax,self.spacing)
        y = torch.linspace(self.ymin,self.ymax,self.spacing)
        z = torch.linspace(self.zmin,self.zmax,self.spacing)
        xx,yy,zz = torch.meshgrid(x,y,z)
        
        
        self.grid_flows=torch.zeros((batchsize,x.shape[0]*y.shape[0]*z.shape[0],3)).to(device)
        self.grid_coords=torch.concat([xx.unsqueeze(-1),yy.unsqueeze(-1),zz.unsqueeze(-1)],dim=-1).view(-1,3).contiguous().repeat(batchsize,1,1).to(device)
        # self.grid_flows=torch.zeros((batchsize,x.shape[0]*y.shape[0],1)).to(device)
        # self.grid_coords=torch.concat([xx.unsqueeze(-1),yy.unsqueeze(-1)],dim=-1).view(-1,2).contiguous().repeat(batchsize,1,1).to(device)



    def idw_interpolation(self,point_cloud,value,grid_coords,power=2):
        '''
        point_cloud: (b,n,3):(X,Y,Z)
        value: (b,n,3):(Vx,Vy,Vz)
        grid_coords: (b,m,3):(X,Y,Z)

        return: (b,m,3):(Vx,Vy,Vz)
        '''
        knn = KNN(k=self.nb, transpose_mode=True)
        dist, idx = knn(point_cloud.contiguous(), grid_coords.contiguous())
        knn_values = fetch_knn_points(value.contiguous(), idx) # (b, n, k, 3)
        weights = torch.pow(dist + 1e-8, -power)
        weights = weights / torch.sum(weights, dim=-1, keepdim=True)# (b, n, k)
        
        interpolated_values = torch.sum(weights.unsqueeze(-1) * knn_values, dim=2) # (b, n, 3)
        return interpolated_values


    def splat(self,flow,coords):
        interpolated_values=self.idw_interpolation(coords,flow,self.grid_coords)
        self.grid_flows=interpolated_values

    def gradient(self):
        # compute gradient of grid_flows
        # self.grid_flows: (b,m,3)
        # self.grid_coords: (b,m,3)
        # using torch.gradient

        grid_flows=self.grid_flows.view(-1,self.spacing,self.spacing,self.spacing,3).contiguous()
        dFx_dx=torch.gradient(grid_flows[:,:,:,:,0],spacing=2*np.pi/self.spacing,dim=1)[0]
        dFy_dy=torch.gradient(grid_flows[:,:,:,:,1],spacing=2*np.pi/self.spacing,dim=2)[0]
        dFz_dz=torch.gradient(grid_flows[:,:,:,:,2],spacing=2*np.pi/self.spacing,dim=3)[0]

        

        return dFx_dx,dFy_dy,dFz_dz
    
    def divergence_per_point(self):
        dFx_dx,dFy_dy,dFz_dz=self.gradient()
        return (dFx_dx+dFy_dy+dFz_dz).view(-1,self.spacing*self.spacing*self.spacing).contiguous()

    def divergence(self):
        div_pp=self.divergence_per_point()
        div_pp_norm=torch.abs(div_pp)
        return torch.mean(div_pp_norm,dim=-1).mean(dim=-1)
    

    def forward(self,flow,coords):
        self.splat(flow=flow,coords=coords)
        return self.divergence()
        

    # def visualize(self):
    #     import matplotlib.pyplot as plt
    #     from mpl_toolkits.mplot3d import Axes3D
    #     fig = plt.figure()
    #     ax = Axes3D(fig)
    #     print(self.grid_coords.shape,self.grid_flows.shape)
    #     ax.scatter(self.grid_coords[0,:,0].cpu().numpy(),
    #                     self.grid_coords[0,:,1].cpu().numpy(),
    #                     self.grid_flows[0,:,0].cpu().numpy(),
    #                    )
    #     # ax.scatter(self.grid_coords[0,:,0].cpu().numpy(),
    #     #            self.grid_coords[0,:,1].cpu().numpy(),
    #     #            self.grid_coords[0,:,2].cpu().numpy(),
    #     #            c=self.grid_flows[0,:,0].cpu().numpy(),
    #     #            )
    #     plt.show()

if __name__=="__main__":
    l=Lattice(nb=8,spacing=10,batchsize=2)
    coords=(torch.rand((2,2048,3))*2*np.pi).cuda()
    coords.requires_grad=True
    flow=coords**2-30
    
    print(l(flow,coords))

    # use autograd to compute divergence
    from torch.autograd import grad
    def gradient(inputs, outputs, create_graph=True, retain_graph=True):
        d_points = torch.ones_like(outputs, requires_grad=False, device=outputs.device)
        points_grad = grad(
            outputs=outputs,
            inputs=inputs,
            grad_outputs=d_points,
            create_graph=create_graph,
            retain_graph=retain_graph,
            only_inputs=True)[0]#[:, -3:]
        return points_grad
    dx = gradient(coords, flow[:,:,0])
    dy = gradient(coords, flow[:,:,1])
    dz = gradient(coords, flow[:,:,2])
    div = dx[:,:,0] + dy[:,:,1] + dz[:,:,2]
    print(div.mean(dim=-1))
    