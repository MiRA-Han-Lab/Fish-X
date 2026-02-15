import h5py
import numpy as np
import warnings
import os
# import open3d as o3d
import data_utils.augmentation as augmentation
from torch.utils.data import Dataset
import torch
from torch.utils.data import DataLoader

warnings.filterwarnings('ignore')


def pc_normalize(pc):
    # print('-----------------pc',pc.shape)
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    # print('-----------------pc_cen', pc.shape)
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    # print('-----------------pc_nor', pc.shape)
    return pc


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


def mesh2pc(load_path, point_num=2048, vis=False):
    obj_mesh = o3d.io.read_triangle_mesh(load_path)
    vertices = np.array(obj_mesh.vertices)
    vertices = pc_normalize(vertices)
    faces = np.array(obj_mesh.triangles)
    ply = o3d.geometry.TriangleMesh()
    ply.vertices = o3d.utility.Vector3dVector(vertices)
    ply.triangles = o3d.utility.Vector3iVector(faces)
    ply.compute_triangle_normals()

    pcd = o3d.geometry.TriangleMesh.sample_points_uniformly(ply, number_of_points=point_num)
    points = np.asarray(pcd.points)

    if vis:
        pcd.paint_uniform_color([0, 0, 0])
        o3d.visualization.draw_geometries([pcd])
    
    return points


class ModelNetDataLoader(Dataset):
    def __init__(self, root, npoint=2048, uniform=False, istrain=True):

        self.npoints = npoint
        self.root = root
        self.data = os.listdir(root)

        self.uniform = uniform
        self.istrain = istrain

    def __len__(self):
        return len(self.data)

    def _list(self):
        return self.data

    def _get_item(self, index):
        # point_loc = self.data[index]
        # point_set = mesh2pc(os.path.join(self.root, point_loc), point_num=self.npoints)
        point_loc = self.data[index]
        point_set = np.load(os.path.join(self.root, point_loc))



        if self.istrain:
            
            point_set2 = mesh2pc(os.path.join(self.root, point_loc), point_num=self.npoints)

            point_view1 = augmentation.transform(point_set)
            point_view2 = augmentation.transform(point_set2)

            point_view1 = farthest_point_sample(point_view1, self.npoints)
            point_view2 = farthest_point_sample(point_view2, self.npoints)

            point_view1[:, 0:3] = pc_normalize(point_view1[:, 0:3])
            point_view2[:, 0:3] = pc_normalize(point_view2[:, 0:3])

            point_view1 = point_view1[:, 0:3]
            point_view2 = point_view2[:, 0:3]
            
            point_view1 = torch.tensor(point_view1, dtype=torch.float32)
            point_view2 = torch.tensor(point_view2, dtype=torch.float32)

            return (point_view1, point_view2)
        else:

            point_set = farthest_point_sample(point_set, self.npoints)

            point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        
            # point_set = point_set[:, 0:3]
            point_set = torch.tensor(point_set, dtype=torch.float32)

            return point_set

    def __getitem__(self, index):
        return self._get_item(index)

    
class Provider(object):
    def __init__(self, cfg, root, n_point=2048):
        self.data = ModelNetDataLoader(root, npoint=n_point, uniform=True, istrain=True)
        self.batch_size = cfg.TRAIN.batch_size
        self.num_workers = cfg.TRAIN.num_workers
        self.is_cuda = cfg.TRAIN.if_cuda
        self.data_iter = None
        self.iteration = 0
        self.epoch = 1

    def __len__(self):
        return self.data.num_per_epoch

    def build(self):
        self.data_iter = iter(
            DataLoader(dataset=self.data, batch_size=self.batch_size, num_workers=self.num_workers,
                       shuffle=True, drop_last=False, pin_memory=True))

    def next(self):
        if self.data_iter is None:
            self.build()
        try:
            batch = next(self.data_iter)
            self.iteration += 1
            if self.is_cuda:
                batch[0] = batch[0].cuda()
                batch[1] = batch[1].cuda()
            return batch
        except StopIteration:
            self.epoch += 1
            self.build()
            self.iteration += 1
            batch = next(self.data_iter)
            if self.is_cuda:
                batch[0] = batch[0].cuda()
                batch[1] = batch[1].cuda()
            return batch


def visualizePointCloud(points, savePath):
    """
    Input:
        points: pointcloud data, [N, D]
    """
    x = [k[0] for k in points]
    y = [k[1] for k in points]
    z = [k[2] for k in points]
    fig = plt.figure(dpi=500)
    ax = fig.add_subplot(111, projection='3d')
    plt.title('Points cloud')
    ax.scatter(x, y, z, c='b', marker='.', s=10, linewidth=0, alpha=1, cmap='spectral')
    plt.savefig(savePath)


if __name__ == '__main__':

    import torch
    from matplotlib import pyplot as plt
    import augmentation as augmentation

    data = ModelNetDataLoader('united.h5', uniform=True, istrain=True)

    DataLoader = torch.utils.data.DataLoader(data, batch_size=12, shuffle=False)

    for (point_view1, point_view2) in DataLoader:
        point = point_view1.numpy()
        for i in range(len(point)):
            visualizePointCloud(point[i], 'test' + str(i) + '.png')
        print(point.shape)
