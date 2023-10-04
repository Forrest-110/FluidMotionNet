import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

class SetConv(torch.nn.Module):
    def __init__(self, nb_feat_in, nb_feat_out):
        """
        Module that performs DGCNN-like convolution on point clouds.
        Parameters
        ----------
        nb_feat_in : int
            Number of input channels.
        nb_feat_out : int
            Number of ouput channels.
        Returns
        -------
        None.
        """

        super(SetConv, self).__init__()

        if nb_feat_in % 2 != 0:
            mid_feature = nb_feat_out // 2
        else:
            mid_feature = (nb_feat_out + nb_feat_in) // 2

        self.fc1 = torch.nn.Conv2d(nb_feat_in + 3, mid_feature, 1, bias=False)
        self.gn1 = torch.nn.GroupNorm(8, mid_feature, affine=True)

        self.fc2 = torch.nn.Conv1d(mid_feature, nb_feat_out, 1, bias=False)
        self.gn2 = torch.nn.GroupNorm(8, nb_feat_out, affine=True)

        self.fc3 = torch.nn.Conv1d(nb_feat_out, nb_feat_out, 1, bias=False)
        self.gn3 = torch.nn.GroupNorm(8, nb_feat_out, affine=True)

        self.pool = lambda x: torch.max(x, 2)[0]
        self.lrelu = torch.nn.LeakyReLU(negative_slope=0.1)

    def forward(self, signal, graph):
        """
        Performs PointNet++-like convolution
        Parameters
        ----------
        signal : torch.Tensor
            Input features of size B x N x nb_feat_in.
        graph : flot.models.graph.Graph
            Graph build on the input point cloud on with the input features
            live. The graph contains the list of nearest neighbors (NN) for
            each point and all edge features (relative point coordinates with
            NN).

        Returns
        -------
        torch.Tensor
            Ouput features of size B x N x nb_feat_out.
        """

        # Input features dimension
        b, n, c = signal.shape
        n_out = graph.size[0] // b

        assert n_out == n

        # Concatenate input features with edge features
        signal = signal.reshape(b * n, c)
        edge_feature = signal[graph.edges].reshape(-1, graph.k_neighbors, c) - signal.view(b * n, 1, c)
        signal = torch.cat((edge_feature.view(-1, c), graph.edge_feats), -1)
        signal = signal.view(b, n_out, graph.k_neighbors, c + 3)
        signal = signal.transpose(1, -1)

        # Pointnet++-like convolution
        for func in [
            self.fc1,
            self.gn1,
            self.lrelu,
            self.pool,
            self.fc2,
            self.gn2,
            self.lrelu,
            self.fc3,
            self.gn3,
            self.lrelu,
        ]:
            signal = func(signal)

        return signal.transpose(1, -1)


class GeoSetConv(torch.nn.Module):
    def __init__(self, nb_feat_in, nb_feat_out):
        """
        Module that performs DGCNN-like convolution on point clouds.
        Parameters
        ----------
        nb_feat_in : int
            Number of input channels.
        nb_feat_out : int
            Number of ouput channels.
        Returns
        -------
        None.
        """

        super(GeoSetConv, self).__init__()

        if nb_feat_in % 2 != 0:
            mid_feature = nb_feat_out // 2
        else:
            mid_feature = (nb_feat_out + nb_feat_in) // 2

        self.fc1 = torch.nn.Conv2d(nb_feat_in + 6, mid_feature, 1, bias=False)
        self.gn1 = torch.nn.GroupNorm(8, mid_feature, affine=True)

        self.fc2 = torch.nn.Conv1d(mid_feature, nb_feat_out, 1, bias=False)
        self.gn2 = torch.nn.GroupNorm(8, nb_feat_out, affine=True)

        self.fc3 = torch.nn.Conv1d(nb_feat_out, nb_feat_out, 1, bias=False)
        self.gn3 = torch.nn.GroupNorm(8, nb_feat_out, affine=True)

        self.pool = lambda x: torch.max(x, 2)[0]
        self.lrelu = torch.nn.LeakyReLU(negative_slope=0.1)

    def forward(self, signal, graph):
        """
        Performs PointNet++-like convolution
        Parameters
        ----------
        signal : torch.Tensor
            Input features of size B x N x nb_feat_in.
        graph : flot.models.graph.Graph
            Graph build on the input point cloud on with the input features
            live. The graph contains the list of nearest neighbors (NN) for
            each point and all edge features (relative point coordinates with
            NN).

        Returns
        -------
        torch.Tensor
            Ouput features of size B x N x nb_feat_out.
        """
        
        # Input features dimension
        b, n, c = signal.shape
        n_out = graph.size[0] // b

        assert n_out == n

        # Concatenate input features with edge features
        signal = signal.reshape(b * n, c)
        edge_feature = signal[graph.edges].reshape(-1, graph.k_neighbors, c) - signal.view(b * n, 1, c)
        signal = torch.cat((edge_feature.view(-1, c), graph.edge_feats), -1)
        signal = signal.view(b, n_out, graph.k_neighbors, c + 6)
        signal = signal.transpose(1, -1)
        
        # Pointnet++-like convolution
        for func in [
            self.fc1,
            self.gn1,
            self.lrelu,
            self.pool,
            self.fc2,
            self.gn2,
            self.lrelu,
            self.fc3,
            self.gn3,
            self.lrelu,
        ]:
            signal = func(signal)
        
        return signal.transpose(1, -1)
    
class Graph:
    def __init__(self, edges, edge_feats, k_neighbors, size):
        """
        Directed nearest neighbor graph constructed on a point cloud.

        Parameters
        ----------
        edges : torch.Tensor
            Contains list with nearest neighbor indices.
        edge_feats : torch.Tensor
            Contains edge features: relative point coordinates.
        k_neighbors : int
            Number of nearest neighbors.
        size : tuple(int, int)
            Number of points.

        """

        self.edges = edges
        self.size = tuple(size)
        self.edge_feats = edge_feats
        self.k_neighbors = k_neighbors

    @staticmethod
    def construct_graph(pcloud, nb_neighbors):
        """
        Construct a directed nearest neighbor graph on the input point cloud.

        Parameters
        ----------
        pcloud : torch.Tensor
            Input point cloud. Size B x N x 3.
        nb_neighbors : int
            Number of nearest neighbors per point.

        Returns
        -------
        graph : flot.models.graph.Graph
            Graph build on input point cloud containing the list of nearest 
            neighbors (NN) for each point and all edge features (relative 
            coordinates with NN).
            
        """

        # Size
        nb_points = pcloud.shape[1]
        size_batch = pcloud.shape[0]

        # Distance between points
        distance_matrix = torch.sum(pcloud ** 2, -1, keepdim=True)
        distance_matrix = distance_matrix + distance_matrix.transpose(1, 2)
        distance_matrix = distance_matrix - 2 * torch.bmm(
            pcloud, pcloud.transpose(1, 2)
        )

        # Find nearest neighbors
        neighbors = torch.argsort(distance_matrix, -1)[..., :nb_neighbors]
        effective_nb_neighbors = neighbors.shape[-1]
        neighbors = neighbors.reshape(size_batch, -1)

        # Edge origin
        idx = torch.arange(nb_points, device=distance_matrix.device).long()
        idx = torch.repeat_interleave(idx, effective_nb_neighbors)

        # Edge features
        edge_feats = []
        for ind_batch in range(size_batch):
            edge_feats.append(
                pcloud[ind_batch, neighbors[ind_batch]] - pcloud[ind_batch, idx]
            )
        edge_feats = torch.cat(edge_feats, 0)

        # Handle batch dimension to get indices of nearest neighbors
        for ind_batch in range(1, size_batch):
            neighbors[ind_batch] = neighbors[ind_batch] + ind_batch * nb_points
        neighbors = neighbors.view(-1)

        # Create graph
        graph = Graph(
            neighbors,
            edge_feats,
            effective_nb_neighbors,
            [size_batch * nb_points, size_batch * nb_points],
        )

        return graph


class ParGraph:
    def __init__(self, edges, edge_feats, k_neighbors, size):
        """
        Directed nearest neighbor graph constructed on a point cloud.

        Parameters
        ----------
        edges : torch.Tensor
            Contains list with nearest neighbor indices.
        edge_feats : torch.Tensor
            Contains edge features: relative point coordinates.
        k_neighbors : int
            Number of nearest neighbors.
        size : tuple(int, int)
            Number of points.

        """

        self.edges = edges
        self.size = tuple(size)
        self.edge_feats = edge_feats
        self.k_neighbors = k_neighbors

    @staticmethod
    def construct_graph(pcloud, nb_neighbors):
        """
        Construct a directed nearest neighbor graph on the input point cloud.

        Parameters
        ----------
        pcloud : torch.Tensor
            Input point cloud. Size B x N x 3.
        nb_neighbors : int
            Number of nearest neighbors per point.

        Returns
        -------
        graph : flot.models.graph.Graph
            Graph build on input point cloud containing the list of nearest 
            neighbors (NN) for each point and all edge features (relative 
            coordinates with NN).
            
        """

        # Size
        nb_points = pcloud.shape[1]
        size_batch = pcloud.shape[0]

        # Distance between points
        distance_matrix = torch.sum(pcloud ** 2, -1, keepdim=True)
        distance_matrix = distance_matrix + distance_matrix.transpose(1, 2)
        distance_matrix = distance_matrix - 2 * torch.bmm(
            pcloud, pcloud.transpose(1, 2)
        )

        # Find nearest neighbors
        neighbors = torch.argsort(distance_matrix, -1)[..., :nb_neighbors]
        effective_nb_neighbors = neighbors.shape[-1]
        neighbors = neighbors.reshape(size_batch, -1)

        # Edge origin
        idx = torch.arange(nb_points, device=distance_matrix.device).long()
        idx = torch.repeat_interleave(idx, effective_nb_neighbors)

        # Edge features
        edge_feats = []
        for ind_batch in range(size_batch):
            edge_feats.append(
                pcloud[ind_batch, neighbors[ind_batch]] - pcloud[ind_batch, idx]
            )
        edge_feats = torch.cat(edge_feats, 0)

        feat_r = torch.norm(edge_feats, p=2, dim=-1, keepdim=True) # B,N*nb,1
        feat_theta = torch.acos(edge_feats[:,-1].unsqueeze(-1) / (feat_r+0.0001))
        feat_phi = torch.atan(edge_feats[:,1].unsqueeze(-1) / (edge_feats[:,0].unsqueeze(-1)+0.0001))
        edge_feats_en = torch.cat((edge_feats, feat_r, feat_theta, feat_phi), dim=-1)

        # Handle batch dimension to get indices of nearest neighbors
        for ind_batch in range(1, size_batch):
            neighbors[ind_batch] = neighbors[ind_batch] + ind_batch * nb_points
        neighbors = neighbors.view(-1)

        # Create graph
        graph = ParGraph(
            neighbors,
            edge_feats_en,
            effective_nb_neighbors,
            [size_batch * nb_points, size_batch * nb_points],
        )

        return graph

def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   
    return idx


def get_graph_feature(x, k=20, idx=None, dim9=False):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if dim9 == False:
            idx = knn(x, k=k)   
        else:
            idx = knn(x[:, 6:], k=k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()  
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
    return feature      


class GeoDGCNN_flow2(nn.Module):
    def __init__(self, k, emb_dims, dropout):
        super(GeoDGCNN_flow2, self).__init__()
        # self.args = args
        self.k = k
        self.emb_dims = emb_dims
        self.dropout = dropout
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(96)
        self.bn4 = nn.BatchNorm2d(96)
        self.bn5 = nn.BatchNorm1d(self.emb_dims)
        self.bn6 = nn.BatchNorm1d(512)
        self.bn7 = nn.BatchNorm1d(256)
        
        self.conv1 = nn.Sequential(nn.Conv2d(32*2, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 96, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(96, 96, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(352, self.emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv1d(self.emb_dims+352, 512, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv1d(512, 256, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=self.dropout)
        self.conv8 = nn.Conv1d(256, 128, kernel_size=1, bias=False)

        self.feat_conv1 = GeoSetConv(3, 32)
        self.feat_conv2 = GeoSetConv(32, 64)
        self.feat_conv3 = GeoSetConv(64, 96)
        

    def forward(self, x):
        
        
        geo_graph = ParGraph.construct_graph(x, self.k)
        g1 = self.feat_conv1(x, geo_graph)     # B x nb_feat_out x N
        g2 = self.feat_conv2(g1, geo_graph)
        g3 = self.feat_conv3(g2, geo_graph)
        g1 = g1.transpose(1, 2).contiguous() 
        g2 = g2.transpose(1, 2).contiguous() 
        g3 = g3.transpose(1, 2).contiguous() 

        

        x = get_graph_feature(g1, k=self.k)     
        x = self.conv1(x)                       
        x = self.conv2(x)                       
        x2 = x.max(dim=-1, keepdim=False)[0]    

        x = get_graph_feature(x2, k=self.k)     
        x = self.conv3(x)                       
        x = self.conv4(x)                       
        x3 = x.max(dim=-1, keepdim=False)[0]    

        mid = torch.cat((g1, x2, x3, g2, g3), dim=1)      

        x = self.conv5(mid)                       

        x = torch.cat((x, mid), dim=1)   

        x = self.conv6(x)                       
        x = self.conv7(x)                       
        x = self.dp1(x)
        x = self.conv8(x)    

                         
        
        return x.transpose(1, 2).contiguous(),geo_graph