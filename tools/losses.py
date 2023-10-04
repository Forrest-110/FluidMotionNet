""" Loss functions. """
import os
import sys
import torch
import numpy as np

# import torchsort
from knn_cuda import KNN
from models.lattice import Lattice
# add path
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_dir not in sys.path:
    sys.path.append(project_dir)


try:
    from auxiliary.ChamferDistancePytorch.chamfer3D import dist_chamfer_3D
    chamfer_dist_3d_cu = dist_chamfer_3D.chamfer_3DDist()
except ImportError:
    print("Could not load compiled 3D CUDA chamfer distance")

from tools.utils import iterate_in_chunks


def compute_loss_unsupervised(recon_flow, corr_conf, target_pc_recon, graph, batch, args):
    """
    Compute unsupervised training loss.

    Parameters
    ----------
    recon_flow: torch.Tensor
        Flow from reconstruction of the target point cloud by the source point cloud.
    corr_conf: torch.Tensor
        Correspondence confidence.
    target_pc_recon: torch.Tensor
        Cross reconstructed target point cloud.
    graph: scoop.models.graph.Graph
        Nearest neighbor graph for the source point cloud.
    batch: scoop.datasets.generic.Batch
        Contains ground truth flow and mask.
    args: dictionary.
        Arguments for loss terms.

    Returns
    -------
    loss : torch.Tensor
        Training loss for current batch.

    """
    mask = None

    if args.use_corr_conf:
        point_weight = corr_conf
    else:
        point_weight = None

    target_pc_input = batch["sequence"][1]
    src_pc_input = batch["sequence"][0]
    target_recon_loss = chamfer_loss(target_pc_recon, target_pc_input, point_weight, args.backward_dist_weight, mask)

    loss = target_recon_loss

    if args.use_corr_conf and args.corr_conf_loss_weight > 0:
        if mask is not None:
            corr_conf_masked = corr_conf[mask > 0]
        else:
            corr_conf_masked = corr_conf

        corr_conf_loss = 1 - torch.mean(corr_conf_masked)
        loss = loss + (args.corr_conf_loss_weight * corr_conf_loss)
    else:
        corr_conf_loss = 0

    if args.use_smooth_flow and args.smooth_flow_loss_weight > 0:
        smooth_flow_loss, _ = smooth_loss(recon_flow, graph, args.nb_neigh_smooth_flow, loss_norm=1, mask=mask)
        loss = loss + (args.smooth_flow_loss_weight * smooth_flow_loss)
    else:
        smooth_flow_loss = 0

    if args.use_div_flow and args.div_flow_loss_weight > 0:
        # src_pc_input.requires_grad=True
        criterion=Lattice(nb=args.div_neighbor,spacing=args.lattice_steps,batchsize=recon_flow.shape[0])
        div_flow_loss = criterion(recon_flow,src_pc_input)
        loss = loss + (args.div_flow_loss_weight * div_flow_loss)
    else:
        div_flow_loss = 0

    return loss, target_recon_loss, corr_conf_loss, smooth_flow_loss, div_flow_loss


def chamfer_dist_3d_pt(pc1, pc2, backward_dist_weight=0.0, chunk_size=2048):
    """
    Compute Chamfer Distance between two point clouds.
    Input:
        pc1: (b, n, 3) torch.Tensor, first point cloud xyz coordinates.
        pc2: (b, m, 3) torch.Tensor, second point cloud xyz coordinates.
        backward_dist_weight: float, weight for backward distance
        chunk_size: int, chunk size for distance computation.

    Output:
        dist1: (b, n) torch.Tensor float32, for each point in pc1, the distance to the closest point in pc2.
        dist2: (b, m) torch.Tensor float32, for each point in pc2, the distance to the closest point in pc1.
        idx1: (b, n) torch.Tensor int32, for each point in pc1, the index of the closest point in pc2 (values are in the range [0, ..., m-1]).
        idx2: (b, m) torch.Tensor int32, for each point in pc2, the index of the closest point in pc1 (values are in the range [0, ..., n-1]).
    """

    b = pc1.shape[0]
    n = pc1.shape[1]
    m = pc2.shape[1]
    device = pc1.device

    dist1 = torch.zeros([b, n], dtype=torch.float32, device=device)
    idx1 = torch.zeros([b, n], dtype=torch.int32, device=device)

    rng1 = np.arange(n)
    for chunk in iterate_in_chunks(rng1, chunk_size):
        pc1_curr = torch.unsqueeze(pc1[:, chunk], dim=2).repeat(1, 1, m, 1)
        pc2_curr = torch.unsqueeze(pc2, dim=1).repeat(1, len(chunk), 1, 1)
        diff = pc1_curr - pc2_curr  # shape (b, cs, m, 3)
        dist = torch.sum(diff ** 2, dim=-1)  # shape (b, cs, m)

        min1 = torch.min(dist, dim=2)
        dist1_curr = min1.values
        idx1_curr = min1.indices.type(torch.IntTensor)
        idx1_curr = idx1_curr.to(dist.device)

        dist1[:, chunk] = dist1_curr
        idx1[:, chunk] = idx1_curr

    if backward_dist_weight == 0.0:
        dist2 = None
        idx2 = None
    else:
        dist2 = torch.zeros([b, m], dtype=torch.float32, device=device)
        idx2 = torch.zeros([b, m], dtype=torch.int32, device=device)

        rng2 = np.arange(m)
        for chunk in iterate_in_chunks(rng2, chunk_size):
            pc1_curr = torch.unsqueeze(pc1, dim=2).repeat(1, 1, len(chunk), 1)
            pc2_curr = torch.unsqueeze(pc2[:, chunk], dim=1).repeat(1, n, 1, 1)
            diff = pc1_curr - pc2_curr  # shape (b, n, cs, 3)
            dist = torch.sum(diff ** 2, dim=-1)  # shape (b, n, cs)

            min2 = torch.min(dist, dim=1)
            dist2_curr = min2.values
            idx2_curr = min2.indices.type(torch.IntTensor)
            idx2_curr = idx2_curr.to(dist.device)

            dist2[:, chunk] = dist2_curr
            idx2[:, chunk] = idx2_curr

    return dist1, dist2, idx1, idx2


def chamfer_loss(pc1, pc2, point_weight=None, backward_dist_weight=1.0, mask=None, use_chamfer_cuda=True):
    if not pc1.is_cuda:
        pc1 = pc1.cuda()

    if not pc2.is_cuda:
        pc2 = pc2.cuda()

    if use_chamfer_cuda:
        dist1, dist2, idx1, idx2 = chamfer_dist_3d_cu(pc1, pc2)
    else:
        dist1, dist2, idx1, idx2 = chamfer_dist_3d_pt(pc1, pc2, backward_dist_weight)

    if point_weight is not None:
        dist1_weighted = dist1 * point_weight
    else:
        dist1_weighted = dist1

    if mask is not None:
        dist1_masked = dist1_weighted[mask > 0]
        dist1_mean = torch.mean(dist1_masked)
    else:
        dist1_mean = torch.mean(dist1_weighted)

    if backward_dist_weight == 1.0:
        loss = dist1_mean + torch.mean(dist2)
    elif backward_dist_weight == 0.0:
        loss = dist1_mean
    else:
        loss = dist1_mean + backward_dist_weight * torch.mean(dist2)

    return loss


def smooth_loss(est_flow, graph, nb_neigh, loss_norm=1, mask=None):
    b, n, c = est_flow.shape
    est_flow_neigh = est_flow.reshape(b * n, c)
    est_flow_neigh = est_flow_neigh[graph.edges]
    est_flow_neigh = est_flow_neigh.view(b, n, graph.k_neighbors, c)
    est_flow_neigh = est_flow_neigh[:, :, 1:(nb_neigh + 1)]
    flow_diff = (est_flow.unsqueeze(2) - est_flow_neigh).norm(p=loss_norm, dim=-1)

    if mask is not None:
        mask_neigh = mask.reshape(b * n)
        mask_neigh = mask_neigh[graph.edges]
        mask_neigh = mask_neigh.view(b, n, graph.k_neighbors)
        mask_neigh = mask_neigh[:, :, 1:(nb_neigh + 1)]
        mask_neigh_sum = mask_neigh.sum(dim=-1)

        flow_diff_masked = flow_diff * mask_neigh
        flow_diff_masked_sum = flow_diff_masked.sum(dim=-1)
        smooth_flow_per_point = flow_diff_masked_sum / (mask_neigh_sum + 1e-8)
        smooth_flow_per_point = smooth_flow_per_point[mask > 0]
    else:
        smooth_flow_per_point = flow_diff.mean(dim=-1)

    smooth_flow_loss = smooth_flow_per_point.mean()

    return smooth_flow_loss, smooth_flow_per_point
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
def divergence_loss(src_pc_input,est_flow,loss_norm=1,clamp=50):
    assert src_pc_input.shape==est_flow.shape
    dx=gradient(src_pc_input,est_flow[:,:,0])
    dy=gradient(src_pc_input,est_flow[:,:,1])
    dz=gradient(src_pc_input,est_flow[:,:,2])
    div = dx[:,:,0] + dy[:,:,1] + dz[:,:,2]
    div[div.isnan()]=0.0

    if loss_norm==1:
        div_term=torch.clamp(torch.abs(div), max=clamp)
    elif loss_norm==2:
        div_term=torch.clamp(div**2, max=clamp)
    else:
        raise NotImplementedError

    div_loss=torch.mean(div_term)

    return div_loss

def find_nearest_K_neighbors(pointcloud,k):
    transpose_mode=True
    knn=KNN(k=k+1,transpose_mode=transpose_mode)
    ref=pointcloud.cuda()
    query=pointcloud.cuda()
    dist,indx=knn(ref,query)
    indx=indx[:,:,1:k+1]
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

def divergence_loss_direct_Version(src_pc_input,est_flow,nb_neigh=2,loss_norm=2,clamp=50):
    assert src_pc_input.shape==est_flow.shape

    nearest_neighbor_idx=find_nearest_K_neighbors(src_pc_input,k=nb_neigh)
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
    # div=dFx/(dx)+dFy/(dy)+dFz/(dz)
    div=torch.div(dFx,dx+1e-12)+torch.div(dFy,dy+1e-12)+torch.div(dFz,dz+1e-12)
    div[div.isnan()]=0.0
    div_per_point=div.mean(dim=-1)
    if loss_norm==1:
        div_term=torch.clamp(torch.abs(div_per_point), max=clamp)
    elif loss_norm==2:
        div_term=torch.clamp(div_per_point**2, max=clamp)
    else:
        raise NotImplementedError
    div_loss=torch.mean(div_term)
    return div_loss

# def find_nearest_neighbors(pointcloud):
#     """
#     Find the nearest neighbor for each point in the point cloud.

#     Args:
#         pointcloud (torch.Tensor): A tensor of shape [b, n, 3] representing the point cloud.

#     Returns:
#         nearest_neighbors (torch.Tensor): A tensor of shape [b, n] containing the indices of the nearest neighbors for each point.
#     """
#     b, n, _ = pointcloud.size()

#     # Reshape the point cloud to [b*n, 3]
#     pointcloud_flat = pointcloud.view(b * n, 3)

#     # Compute the pairwise Euclidean distances between points in the point cloud
#     pairwise_distances = torch.cdist(pointcloud_flat, pointcloud_flat)

#     # Set the diagonal elements to a large value to avoid selecting the point itself
#     diag_mask = torch.eye(b * n).bool()
#     pairwise_distances[diag_mask] = float('inf')

#     # Find the index of the nearest neighbor for each point
#     nearest_neighbors = torch.argmin(pairwise_distances, dim=1)

#     # Reshape the nearest neighbor indices to [b, n]
#     nearest_neighbors = nearest_neighbors.view(b, n)

#     return nearest_neighbors

# def divergence_loss_direct_Version(src_pc_input,est_flow,graph, nb_neigh,loss_norm=1,clamp=50):
#     assert src_pc_input.shape==est_flow.shape

#     nearest_neighbor_idx=find_nearest_neighbors(src_pc_input)
#     nearest_point=torch.gather(src_pc_input,1,nearest_neighbor_idx.unsqueeze(-1).repeat(1,1,3))
#     nearest_flow=torch.gather(est_flow,1,nearest_neighbor_idx.unsqueeze(-1).repeat(1,1,3))
#     dx=(src_pc_input-nearest_point)[:,:,0]
#     dy=(src_pc_input-nearest_point)[:,:,1]
#     dz=(src_pc_input-nearest_point)[:,:,2]
#     dFx=(est_flow-nearest_flow)[:,:,0]
#     dFy=(est_flow-nearest_flow)[:,:,1]
#     dFz=(est_flow-nearest_flow)[:,:,2]
#     div=dFx/(2*dx)+dFy/(2*dy)+dFz/(2*dz)
#     div[div.isnan()]=0.0
#     div_per_point=div.mean(dim=-1)
#     if loss_norm==1:
#         div_term=torch.clamp(torch.abs(div_per_point), max=clamp)
#     elif loss_norm==2:
#         div_term=torch.clamp(div_per_point**2, max=clamp)
#     else:
#         raise NotImplementedError
#     div_loss=torch.mean(div_term)
#     return div_loss

#     # b,n,c=src_pc_input.shape
#     # src_pc_input_neigh = src_pc_input.reshape(b * n, c)
#     # src_pc_input_neigh = src_pc_input_neigh[graph.edges]
#     # src_pc_input_neigh = src_pc_input_neigh.view(b, n, graph.k_neighbors, c)
#     # src_pc_input_neigh = src_pc_input_neigh[:, :, 1:(nb_neigh + 1)]

#     # b,n,c=est_flow.shape
#     # est_flow_neigh = est_flow.reshape(b * n, c)
#     # est_flow_neigh = est_flow_neigh[graph.edges]
#     # est_flow_neigh = est_flow_neigh.view(b, n, graph.k_neighbors, c)
#     # est_flow_neigh = est_flow_neigh[:, :, 1:(nb_neigh + 1)]

#     # assert not est_flow.isnan().any()
#     # assert not est_flow_neigh.isnan().any()

#     # dFx=(est_flow.unsqueeze(2)-est_flow_neigh)[:,:,:,0]
#     # dFy=(est_flow.unsqueeze(2)-est_flow_neigh)[:,:,:,1]
#     # dFz=(est_flow.unsqueeze(2)-est_flow_neigh)[:,:,:,2]

#     # dx=(src_pc_input.unsqueeze(2)-src_pc_input_neigh)[:,:,:,0]
#     # dy=(src_pc_input.unsqueeze(2)-src_pc_input_neigh)[:,:,:,1]
#     # dz=(src_pc_input.unsqueeze(2)-src_pc_input_neigh)[:,:,:,2]

#     # # if abs(dx)<0.01, then dx=sgn(dx)*0.01
#     # min_dist_threshold=1e-8
#     # max_dist_threshold=50
#     # # min_F_threshold=1e-8
#     # # max_F_threshold=50
#     # # assert not (dx.abs()<dist_threshold).any()
#     # # assert not (dy.abs()<dist_threshold).any()
#     # # assert not (dz.abs()<dist_threshold).any()

#     # # assert not (dFx.abs()<dist_threshold).any()
#     # # assert not (dFy.abs()<dist_threshold).any()
#     # # assert not (dFz.abs()<dist_threshold).any()

#     # # dx[dx.abs()<min_dist_threshold]=min_dist_threshold*dx[dx.abs()<min_dist_threshold]/dx[dx.abs()<min_dist_threshold].abs()
#     # # dy[dy.abs()<min_dist_threshold]=min_dist_threshold*dy[dy.abs()<min_dist_threshold]/dy[dy.abs()<min_dist_threshold].abs()
#     # # dz[dz.abs()<min_dist_threshold]=min_dist_threshold*dz[dz.abs()<min_dist_threshold]/dz[dz.abs()<min_dist_threshold].abs()
#     # # dx[dx.abs()>max_dist_threshold]=max_dist_threshold*dx[dx.abs()>max_dist_threshold]/dx[dx.abs()>max_dist_threshold].abs()
#     # # dy[dy.abs()>max_dist_threshold]=max_dist_threshold*dy[dy.abs()>max_dist_threshold]/dy[dy.abs()>max_dist_threshold].abs()
#     # # dz[dz.abs()>max_dist_threshold]=max_dist_threshold*dz[dz.abs()>max_dist_threshold]/dz[dz.abs()>max_dist_threshold].abs()

#     # # dFx[dFx.abs()<min_dist_threshold]=min_dist_threshold*dFx[dFx.abs()<min_dist_threshold]/dFx[dFx.abs()<min_dist_threshold].abs()
#     # # dFy[dFy.abs()<min_dist_threshold]=min_dist_threshold*dFy[dFy.abs()<min_dist_threshold]/dFy[dFy.abs()<min_dist_threshold].abs()
#     # # dFz[dFz.abs()<min_dist_threshold]=min_dist_threshold*dFz[dFz.abs()<min_dist_threshold]/dFz[dFz.abs()<min_dist_threshold].abs()
#     # # dFx[dFx.abs()>max_dist_threshold]=max_dist_threshold*dFx[dFx.abs()>max_dist_threshold]/dFx[dFx.abs()>max_dist_threshold].abs()
#     # # dFy[dFy.abs()>max_dist_threshold]=max_dist_threshold*dFy[dFy.abs()>max_dist_threshold]/dFy[dFy.abs()>max_dist_threshold].abs()
#     # # dFz[dFz.abs()>max_dist_threshold]=max_dist_threshold*dFz[dFz.abs()>max_dist_threshold]/dFz[dFz.abs()>max_dist_threshold].abs()
    
    
#     # assert not (dx.isnan()).any()
#     # assert not (dy.isnan()).any()
#     # assert not (dz.isnan()).any()
#     # assert not (dFx.isnan()).any()
#     # assert not (dFy.isnan()).any()
#     # assert not (dFz.isnan()).any()
#     # div = dFx/(2*dx) + dFy/(2*dy) + dFz/(2*dz)
#     # # if torch.isnan(div).any():
#     # #     return torch.tensor(0.0,device=div.device)
#     # div[div.isnan()]=0.001
#     # div_per_point=div.mean(dim=-1)

#     # if loss_norm==1:
#     #     div_term=torch.clamp(torch.abs(div_per_point), max=clamp)
#     # elif loss_norm==2:
#     #     div_term=torch.clamp(div_per_point**2, max=clamp)
#     # else:
#     #     raise NotImplementedError
    
#     # div_loss=torch.mean(div_term)

#     # return div_loss


    




if __name__ == "__main__":
    # test divergence loss
    src_pc_input=torch.rand((2,100,3),requires_grad=True)
    est_flow=src_pc_input**2-src_pc_input
    div=divergence_loss(src_pc_input,est_flow)
    print(div)
