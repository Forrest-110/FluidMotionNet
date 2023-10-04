import torch
from torch.autograd import grad
from knn_cuda import KNN
import torchsort

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



def find_nearest_K_neighbors(pointcloud,k):
    transpose_mode=True
    knn=KNN(k=k,transpose_mode=transpose_mode)
    ref=pointcloud.cuda()
    query=pointcloud.cuda()
    dist,indx=knn(ref,query)
    print(dist)
    # b,n,c=pointcloud.shape
    # # compute the distance matrix
    # pointcloud=pointcloud.unsqueeze(2).expand(-1,-1,n,-1)
    # dist=torch.norm(pointcloud-pointcloud.transpose(1,2),dim=-1)
    # # dist: [b,n,n]
    # # sort the distance matrix
    # dist=dist.reshape(b*n,n)
    # indx=torchsort.soft_rank(dist,regularization_strength=1e-1).long()
    # indx=indx.reshape(b,n,n)
    # indx=indx[:,:,1:k+1]
    return indx

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

def divergence_loss_direct_Version(src_pc_input,est_flow,nb_neigh=2,loss_norm=1,clamp=50):
    assert src_pc_input.shape==est_flow.shape

    nearest_neighbor_idx=find_nearest_K_neighbors(src_pc_input,k=nb_neigh)
    print(nearest_neighbor_idx)
    # nearest_neighbor_idx: [b,n,k]
    # get the nearest neighbor point, src_pc_input: [b,n,3]
    nearest_point=fetch_knn_points(src_pc_input,nearest_neighbor_idx)
    # nearest_point: [b,n,k,3]
    nearest_flow=fetch_knn_points(est_flow,nearest_neighbor_idx)
    
    src_pc_input=src_pc_input.unsqueeze(2).expand(-1,-1,nb_neigh,-1)
    est_flow=est_flow.unsqueeze(2).expand(-1,-1,nb_neigh,-1)
    dx=(src_pc_input-nearest_point)[:,:,:,0]
    dy=(src_pc_input-nearest_point)[:,:,:,1]
    dz=(src_pc_input-nearest_point)[:,:,:,2]
    dFx=(est_flow-nearest_flow)[:,:,:,0]
    dFy=(est_flow-nearest_flow)[:,:,:,1]
    dFz=(est_flow-nearest_flow)[:,:,:,2]
    div=dFx*(dx)+dFy*(dy)+dFz*(dz)
    div[div.isnan()]=0.0
    div_per_point=div.mean(dim=-1)
    print(div_per_point)
    if loss_norm==1:
        div_term=torch.clamp(torch.abs(div_per_point), max=clamp)
    elif loss_norm==2:
        div_term=torch.clamp(div_per_point**2, max=clamp)
    else:
        raise NotImplementedError
    div_loss=torch.mean(div_term)
    return div_loss


points = torch.rand((2,10,3),requires_grad=True).cuda()
vector_field = points **2 - points

dx = gradient(points, vector_field[:,:,0])
dy = gradient(points, vector_field[:,:,1])
dz = gradient(points, vector_field[:,:,2])
div = dx[:,:,0] + dy[:,:,1] + dz[:,:,2]
print(div)

loss=(divergence_loss_direct_Version(points,vector_field,nb_neigh=4))
loss.backward()
print(vector_field.grad)