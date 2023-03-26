import os
import glob
import h5py
import random
import numpy as np
from torch.utils.data import Dataset

id2cat = ['airplane', 'bag', 'cap', 'car', 'chair', 'earphone', 'guitar', 'knife', 'lamp', 'laptop', 
        'motorbike', 'mug', 'pistol', 'rocket', 'skateboard', 'table']
cat2part = {'airplane': ['body','wing','tail','engine or frame'], 'bag': ['handle','body'], 'cap': ['panels or crown','visor or peak'], 
            'car': ['roof','hood','wheel or tire','body'],
            'chair': ['back','seat pad','leg','armrest'], 'earphone': ['earcup','headband','data wire'], 
            'guitar': ['head or tuners','neck','body'], 
            'knife': ['blade', 'handle'], 'lamp': ['leg or wire','lampshade'], 
            'laptop': ['keyboard','screen or monitor'], 
            'motorbike': ['gas tank','seat','wheel','handles or handlebars','light','engine or frame'], 'mug': ['handle', 'cup'], 
            'pistol': ['barrel', 'handle', 'trigger and guard'], 
            'rocket': ['body','fin','nose cone'], 'skateboard': ['wheel','deck','belt for foot'], 'table': ['desktop','leg or support','drawer']}
id2part2cat = [['body', 'airplane'], ['wing', 'airplane'], ['tail', 'airplane'], ['engine or frame', 'airplane'], ['handle', 'bag'], ['body', 'bag'], 
            ['panels or crown', 'cap'], ['visor or peak', 'cap'],
            ['roof', 'car'], ['hood', 'car'], ['wheel or tire',  'car'], ['body', 'car'],
            ['backrest or back', 'chair'], ['seat', 'chair'], ['leg or support', 'chair'], ['armrest', 'chair'], 
            ['earcup', 'earphone'], ['headband', 'earphone'], ['data wire',  'earphone'], 
            ['head or tuners', 'guitar'], ['neck', 'guitar'], ['body', 'guitar'], ['blade', 'knife'], ['handle', 'knife'], 
            ['support or tube of wire', 'lamp'], ['lampshade', 'lamp'], ['canopy', 'lamp'], ['support or tube of wire', 'lamp'], 
            ['keyboard', 'laptop'], ['screen or monitor', 'laptop'], ['gas tank', 'motorbike'], ['seat', 'motorbike'], ['wheel', 'motorbike'], 
            ['handles or handlebars', 'motorbike'], ['light', 'motorbike'], ['engine or frame', 'motorbike'], ['handle', 'mug'], ['cup or body', 'mug'], 
            ['barrel', 'pistol'], ['handle', 'pistol'], ['trigger and guard', 'pistol'], ['body', 'rocket'], ['fin', 'rocket'], ['nose cone', 'rocket'], 
            ['wheel', 'skateboard'], ['deck',  'skateboard'], ['belt for foot', 'skateboard'], 
            ['desktop', 'table'], ['leg or support', 'table'], ['drawer''table']]


def download_shapenetpart(data_path):
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    if not os.path.exists(os.path.join(data_path, 'shapenet_part_seg_hdf5_data')):
        os.mkdir(os.path.join(data_path, 'shapenet_part_seg_hdf5_data'))
        www = 'https://shapenet.cs.stanford.edu/media/shapenet_part_seg_hdf5_data.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s --no-check-certificate; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], os.path.join(data_path, 'shapenet_part_seg_hdf5_data')))
        os.system('rm %s' % (zipfile))
        

def load_data_partseg(data_path, partition):
    download_shapenetpart(data_path)
    all_data = []
    all_label = []
    all_seg = []

    if partition == 'trainval':
        file = glob.glob(os.path.join(data_path, 'shapenet_part_seg_hdf5_data', 'hdf5_data', '*train*.h5')) \
               + glob.glob(os.path.join(data_path, 'shapenet_part_seg_hdf5_data', 'hdf5_data', '*val*.h5'))
    elif partition == 'train':
        file = glob.glob(os.path.join(data_path, 'shapenet_part_seg_hdf5_data', 'hdf5_data', '*train*.h5'))
    elif partition == 'val':
        file = glob.glob(os.path.join(data_path, 'shapenet_part_seg_hdf5_data', 'hdf5_data', '*val*.h5'))
    else:
        file = glob.glob(os.path.join(data_path, 'shapenet_part_seg_hdf5_data', 'hdf5_data', '*test*.h5'))
    for h5_name in file:
        f = h5py.File(h5_name, 'r+')
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        seg = f['pid'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
        all_seg.append(seg)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    all_seg = np.concatenate(all_seg, axis=0)

    if partition == 'test':
        return all_data, all_label, all_seg
    else:
        kshot = 16
        category_num = {}
        for i in range(16):
            category_num[i] = []
        for j in range(all_label.shape[0]):
            category_num[int(all_label[j,0])].append(j)

        all_data1, all_label1, all_seg1 = [], [], []
        
        for i in range(16):
            list = range(0, len(category_num[i]))
            nums = random.sample(list, kshot)
            for n in nums:
                all_data1.append(all_data[category_num[i][n],:,:][None, :,:])
                all_label1.append(all_label[category_num[i][n]][:, None])
                all_seg1.append(all_seg[category_num[i][n]][None,:])
        
        all_data1 = np.concatenate(all_data1, axis=0)
        all_label1 = np.concatenate(all_label1, axis=0)
        all_seg1 = np.concatenate(all_seg1, axis=0)
        return all_data1, all_label1, all_seg1


class ShapeNetPart(Dataset):
    def __init__(self, data_path='data/', num_points=2048, partition='train', class_choice=None):
        self.data, self.label, self.seg = load_data_partseg(data_path, partition)
        self.cat2id = {'airplane': 0, 'bag': 1, 'cap': 2, 'car': 3, 'chair': 4, 
                       'earphone': 5, 'guitar': 6, 'knife': 7, 'lamp': 8, 'laptop': 9, 
                       'motorbike': 10, 'mug': 11, 'pistol': 12, 'rocket': 13, 'skateboard': 14, 'table': 15}
        self.seg_num = [4, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3]
        self.index_start = [0, 4, 6, 8, 12, 16, 19, 22, 24, 28, 30, 36, 38, 41, 44, 47]
        self.num_points = num_points
        self.partition = partition        
        self.class_choice = class_choice

        if self.class_choice != None:
            id_choice = self.cat2id[self.class_choice]
            indices = (self.label == id_choice).squeeze()
            self.data = self.data[indices]
            self.label = self.label[indices]
            self.seg = self.seg[indices]
            self.seg_num_all = self.seg_num[id_choice]
            self.seg_start_index = self.index_start[id_choice]
        else:
            self.seg_num_all = 50
            self.seg_start_index = 0

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        seg = self.seg[item][:self.num_points]
        return pointcloud, label, seg
    
    def __len__(self):
        return self.data.shape[0]

