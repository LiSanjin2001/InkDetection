import os
import cv2
import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader

from utils.other_utils import setup_seed


# ######################################################################## 1.制作数据集
def get_data(mode, folder_path, size):
    """获取数据"""
    print(os.listdir(folder_path))
    # 获取掩码
    print('Get mask...')
    mask_path = os.path.join(folder_path, 'mask.png')
    mask = cv2.imread(mask_path, 0)
    # 计算填充值
    pad0 = (size - mask.shape[0] % size)
    pad1 = (size - mask.shape[1] % size)
    # 获取train的label
    if mode == 'train':
        # 获取label
        print('Get label...')
        label_path = os.path.join(folder_path, 'inklabels.png')
        label = cv2.imread(label_path, 0)
        # 归一化
        label = label / 255
        # 填充
        label = np.pad(label, ((0, pad0), (0, pad1)), 'constant', constant_values=0)
    else:
        label = None
    # 获取image
    image_data_list = []
    data_path = os.path.join(folder_path, 'surface_volume')
    for idx, fit_name in enumerate(sorted(os.listdir(data_path))):
        fit_path = os.path.join(data_path, fit_name)
        print('{}/{}: {}'.format(idx + 1, len(os.listdir(data_path)), fit_path))
        # 读取
        image_data = cv2.imread(fit_path, 0)
        # 填充
        image_data = np.pad(image_data, ((0, pad0), (0, pad1)), 'constant', constant_values=0)
        # 保存
        image_data_list.append(image_data)
    # 堆叠
    print('Stacking...')
    image_stack = np.stack(image_data_list, axis=2)  # (H, W, C)
    return image_stack, label


def save_npy(mode, data_path, save_path, size, stride):
    """保存npy文件"""
    for folder_num in sorted(os.listdir(data_path)):
        folder_path = os.path.join(data_path, folder_num)
        print('Deal with : {}'.format(folder_path))
        image_data, label_data = get_data(mode, folder_path, size)
        # 切割
        print('Cutting...')
        x_list = list(range(0, image_data.shape[0] - size + 1, stride))
        y_list = list(range(0, image_data.shape[1] - size + 1, stride))
        npy_num = len(x_list) * len(y_list)
        cnt = 0
        for x_start in x_list:
            for y_start in y_list:
                cnt += 1
                # 子图象坐标
                x_end = x_start + size
                y_end = y_start + size
                # 子图象保存名（补零）
                flie_name = '{}_{}_{}'.format(folder_num,
                                                  str(x_start).zfill(5), str(y_start).zfill(5))
                # 提取子标签
                sub_label = None
                if mode == 'train':
                    sub_label = label_data[x_start:x_end, y_start:y_end]
                    # 判断子标签是否全为0
                    if np.sum(sub_label) == 0:
                        print('[{}/{}:{}] No ink in this sub image!'.format(
                            cnt, npy_num, flie_name))
                        continue
                # 提取子图象
                sub_image = image_data[x_start:x_end, y_start:y_end, :]
                # 保存
                if mode == 'train':
                    np.savez(os.path.join(save_path, flie_name),
                             image=sub_image, label=sub_label, allow_pickle=True)
                    print('[{}/{}:{}] Save success!'.format(cnt, npy_num, flie_name))
                else:
                    np.savez(os.path.join(save_path, flie_name),
                             image=sub_image, allow_pickle=True)
                    print('[{}/{}:{}] Save success!'.format(cnt, npy_num, flie_name))


def make_train_test(args):
    """制作训练集和测试集"""
    print('Making datasets...')
    root_path = args.data_path
    size = args.size
    stride = args.stride
    datasets_path = os.path.join(args.datasets_path, str(size) + '_' + str(stride))
    # 创建数据集文件夹
    os.makedirs(datasets_path)
    # 训练集
    print('Making train datasets...')
    train_data_path = os.path.join(root_path, 'train')
    train_save_path = os.path.join(datasets_path, 'train')
    os.makedirs(train_save_path)
    print('Deal with : {}'.format(train_save_path))
    save_npy(mode='train', data_path=train_data_path,
             save_path=train_save_path, size=size, stride=stride)

    # 测试集
    test_data_path = os.path.join(root_path, 'test')
    test_path = os.path.join(datasets_path, 'test')
    os.makedirs(test_path)
    print('Deal with : {}'.format(test_path))
    save_npy(mode='test', data_path=test_data_path,
             save_path=test_path, size=size, stride=size)


def make_dataset(args):
    """制作数据集"""
    # 是否制作数据集
    if args.make_datasets:
        # 数据集地址
        datasets_path = os.path.join(args.datasets_path,
                                     str(args.size) + '_' + str(args.stride))
        # 是否重新制作数据集
        if args.remake:
            # 数据集是否存在
            if os.path.exists(datasets_path):
                # 删除数据集
                os.system('rm -rf {}'.format(datasets_path))
            make_train_test(args)
        else:
            # 数据集是否存在
            if os.path.exists(datasets_path):
                print('Datasets {} already exists!'.format(datasets_path))
            else:
                make_train_test(args)

# ######################################################################## 2.制作迭代器
class InkDataset(Dataset):
    def __init__(self, data_list, arg):
        super(InkDataset, self).__init__()
        self.data_list = data_list
        self.args = arg

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        os.environ['CUDA_VISIBLE_DEVICES'] = self.args.gpu_ids
        data_path = self.data_list[index]
        file = np.load(data_path)
        data = file['image']
        data = torch.from_numpy(data).float().to(self.args.device, dtype=torch.float32)
        data = data.permute(2, 0, 1)
        # data = self.transform(data)
        if 'label' in file.keys():
            label = file['label']
            label = torch.from_numpy(label).long().to(self.args.device, dtype=torch.float32)
            # mask = self.transform(mask)
            return data, label
        else:
            label = data_path
            return data, label


def load_train_dataset(arg):
    """加载训练集和验证集"""
    os.environ['CUDA_VISIBLE_DEVICES'] = arg.gpu_ids
    setup_seed(arg.seed)
    data_path = os.path.join(arg.datasets_path, str(arg.size) + '_' + str(arg.stride), 'train')
    val_list = sorted(random.sample(os.listdir(data_path), arg.val_num))
    val_path_list = list(os.path.join(data_path, val) for val in val_list)
    val_dataset = InkDataset(val_path_list, arg)
    val_loaders = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=arg.num_workers)

    train_list = sorted(list(set(os.listdir(data_path)) - set(val_list)))
    train_path_list = list(os.path.join(data_path, train) for train in train_list)
    train_dataset = InkDataset(train_path_list, arg)
    train_loaders = DataLoader(train_dataset, batch_size=arg.batch_size, shuffle=True, num_workers=arg.num_workers)

    return train_loaders, val_loaders


def load_test_dataset(arg):
    """加载测试集"""
    os.environ['CUDA_VISIBLE_DEVICES'] = arg.gpu_ids
    setup_seed(arg.seed)
    data_path = os.path.join(arg.datasets_path, str(arg.size) + '_' + str(arg.stride), 'test')
    test_list = sorted(os.listdir(data_path))
    test_list = list(os.path.join(data_path, test) for test in test_list)
    test_dataset = InkDataset(test_list, arg)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=arg.num_workers)

    return test_loader
