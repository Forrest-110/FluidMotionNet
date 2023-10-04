import os
import sys
import glob
# import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
try:
    from .generic import SceneFlowDataset
except ImportError:
    from generic import SceneFlowDataset


class FluidFlowDataset(SceneFlowDataset):
    def __init__(self, root_dir, nb_points, all_points, mode, nb_examples=-1):
        super(FluidFlowDataset,self).__init__(nb_points,all_points)
        self.mode = mode
        self.nb_examples = nb_examples
        self.root_dir = root_dir
        self.filenames = self.get_file_list()
        self.filename_curr = ""

        self.cache = {}
        self.cache_size = 30000

    def __len__(self):

        return len(self.filenames)
    
    def get_file_list(self):
        """
        Find and filter out paths to all examples in the dataset. 
        
        """

        #
        if self.mode == "train" :
            pattern = "TRAIN*.npz"
        elif self.mode == "test" or self.mode == "val":
            pattern = "TEST*.npz"
        else:
            raise ValueError("Mode " + str(self.mode) + "unknown.")
        filenames = glob.glob(os.path.join(self.root_dir, pattern))
        filenames = [d for d in filenames if 'TRAIN_C_0140_left_0006-0' not in d]

        # # Train / val / test split
        # if self.mode == "train" or self.mode == "val":
        #     ind_val = set(np.linspace(0, len(filenames) - 1, 2000).astype("int"))
        #     ind_all = set(np.arange(len(filenames)).astype("int"))
        #     ind_train = ind_all - ind_val
        #     assert (
        #         len(ind_train.intersection(ind_val)) == 0
        #     ), "Train / Val not split properly"
        #     filenames = np.sort(filenames)
        #     if self.mode == "train":
        #         filenames = filenames[list(ind_train)]
        #     elif self.mode == "val":
        #         filenames = filenames[list(ind_val)]

        filenames = np.sort(filenames)

        if 0 < self.nb_examples < len(filenames):
            idx_perm = np.random.permutation(len(filenames))
            idx_sel = idx_perm[:self.nb_examples]
            filenames = filenames[idx_sel]

        print(self.mode, ': ',len(filenames))

        return filenames
    

    def load_sequence(self, idx):
        """
        Load a sequence of point clouds.

        Parameters
        ----------
        idx : int
            Index of the sequence to load.

        Returns
        -------
        sequence : list(np.array, np.array)
            List [pc1, pc2] of point clouds between which to estimate scene 
            flow. pc1 has size n x 3 and pc2 has size m x 3.
            
        ground_truth : list(np.array, np.array)
            List [mask, flow]. mask has size n x 1 and pc1 has size n x 3. 
            flow is the ground truth scene flow between pc1 and pc2. mask is 
            binary with zeros indicating where the flow is not valid/occluded.

        """

        if idx in self.cache:
            sequence,ground_truth=self.cache[idx]
        else:
            self.filename_curr = self.filenames[idx]

            # Load data
        
            with np.load(self.filename_curr) as data:
                sequence = [data["pos1"].astype('float32'), data["pos2"].astype('float32')]
                ground_truth = [np.ones_like(data["pos1"][:, 0:1]), data["flow"].astype('float32')]
            if len(self.cache) < self.cache_size:
                # self.cache[index] = (pos1, pos2, color1, color2, flow)
                self.cache[idx] = (sequence,ground_truth)
        
        return sequence, ground_truth
    


class FluidFlowDataset_GOTversion(Dataset):
    def __init__(self, root_dir, nb_points, all_points, mode, nb_examples=-1):
        self.mode = mode
        self.nb_examples = nb_examples
        self.root_dir = root_dir
        self.filenames = self.get_file_list()
        self.filename_curr = ""
        self.nb_points = nb_points
        self.all_points = all_points

        self.cache = {}
        self.cache_size = 30000

    def __len__(self):

        return len(self.filenames)
    
    def __getitem__(self, index):
        if index in self.cache:
            # pos1, pos2, color1, color2, flow = self.cache[index]
            pos1, pos2, flow = self.cache[index]
        else:
            # print(self.datapath[index])
            fn = self.filenames[index]
            with open(fn, 'rb') as fp:
                data = np.load(fp)
                pos1 = data['pos1'].astype('float32')
                pos2 = data['pos2'].astype('float32')
                flow = data['flow'].astype('float32')
            
            

            if len(self.cache) < self.cache_size:
                # self.cache[index] = (pos1, pos2, color1, color2, flow)
                self.cache[index] = (pos1, pos2, flow)
        n1 = pos1.shape[0]
        n2 = pos2.shape[0]
        orig_size = [np.array([n1], dtype=np.int32), np.array([n2], dtype=np.int32)]
        if self.mode == 'train':
                
                sample_idx1 = np.random.choice(n1, self.nb_points, replace=False)
                
                sample_idx2 = np.random.choice(n2, self.nb_points, replace=False)

                pos1 = pos1[sample_idx1, :]
                pos2 = pos2[sample_idx2, :]
                # color1 = color1[sample_idx1, :]
                # color2 = color2[sample_idx2, :]
                flow = flow[sample_idx1, :]
                #mask1 = mask1[sample_idx1]
        else:
                pos1 = pos1[:self.nb_points, :]
                pos2 = pos2[:self.nb_points, :]
                # color1 = color1[:self.npoints, :]
                # color2 = color2[:self.npoints, :]
                flow = flow[:self.nb_points, :]
                #mask1 = mask1[:self.npoints]

        pos1_center = np.mean(pos1, 0)
        pos1 -= pos1_center
        pos2 -= pos1_center
        sequence = [pos1, pos2]
        ground_truth = [np.ones_like(pos1[:, 0:1]), flow]
        sequence, ground_truth, orig_size = self.to_torch(sequence, ground_truth, orig_size)
        data = {"sequence": sequence, "ground_truth": ground_truth, "orig_size": orig_size}

        return data
        
    
    def get_file_list(self):
        """
        Find and filter out paths to all examples in the dataset. 
        
        """

        #
        if self.mode == "train" :
            pattern = "TRAIN*.npz"
        elif self.mode == "test" or self.mode == "val":
            pattern = "TEST*.npz"
        else:
            raise ValueError("Mode " + str(self.mode) + "unknown.")
        filenames = glob.glob(os.path.join(self.root_dir, pattern))
        filenames = [d for d in filenames if 'TRAIN_C_0140_left_0006-0' not in d]

        

        filenames = np.sort(filenames)

        if 0 < self.nb_examples < len(filenames):
            idx_perm = np.random.permutation(len(filenames))
            idx_sel = idx_perm[:self.nb_examples]
            filenames = filenames[idx_sel]

        print(self.mode, ': ',len(filenames))

        return filenames
    
    def to_torch(self, sequence, ground_truth, orig_size):
        """
        Convert numpy array and torch.Tensor.

        Parameters
        ----------
        sequence : list(np.array, np.array)
            List [pc1, pc2] of point clouds between which to estimate scene 
            flow. pc1 has size n x 3 and pc2 has size m x 3.
            
        ground_truth : list(np.array, np.array)
            List [mask, flow]. mask has size n x 1 and pc1 has size n x 3. 
            flow is the ground truth scene flow between pc1 and pc2. mask is 
            binary with zeros indicating where the flow is not valid/occluded.

        orig_size : list(np.array, np.array)
            List [n1, n2]. Original size of the point clouds.

        Returns
        -------
        sequence : list(torch.Tensor, torch.Tensor)
            List [pc1, pc2] of point clouds between which to estimate scene 
            flow. pc1 has size 1 x n x 3 and pc2 has size 1 x m x 3.
            
        ground_truth : list(torch.Tensor, torch.Tensor)
            List [mask, flow]. mask has size 1 x n x 1 and pc1 has size 
            1 x n x 3. flow is the ground truth scene flow between pc1 and pc2.
            mask is binary with zeros indicating where the flow is not 
            valid/occluded.

        """

        sequence = [torch.unsqueeze(torch.from_numpy(s), 0).float() for s in sequence]
        ground_truth = [torch.unsqueeze(torch.from_numpy(gt), 0).float() for gt in ground_truth]
        orig_size = [torch.unsqueeze(torch.from_numpy(os), 0) for os in orig_size]

        return sequence, ground_truth, orig_size


    
if __name__ =='__main__':
    folder = 'FluidFlow3D-norm' # 'data_sample'
    dataset_path = os.path.join('/data/Sceneflow/FluidFlow3D-family', folder)
    dataset=FluidFlowDataset(root_dir=dataset_path,nb_points=4096,all_points=False,mode='test')
    print(dataset.filenames[0])
    data=dataset[0]
    sequence=data['sequence']
    ground_truth=data['ground_truth']
    orig_size=data['orig_size']
    print(ground_truth[1].shape)


