import os, sys
from PIL import Image
import numpy as np
from torch.utils import data
from datasets.data_utils import CDDataAugmentation
"""
CD data set with pixel-level labels
├─image
├─image_post
├─label
└─list
"""
IMG_FOLDER_NAME = "A"
IMG_POST_FOLDER_NAME = 'B'
LIST_FOLDER_NAME = 'list'
ANNOT_FOLDER_NAME = "label"

IGNORE = 255

label_suffix='.png' # jpg for gan dataset, others : png

def load_img_name_list(dataset_path):
    img_name_list = np.loadtxt(dataset_path, dtype=str)
    # only take the first col(pass the comment behind the image name)
    if img_name_list.ndim == 2:
        return img_name_list[:, 0]
    return img_name_list

def load_image_label_list_from_npy(npy_path, img_name_list):
    cls_labels_dict = np.load(npy_path, allow_pickle=True).item()
    return [cls_labels_dict[img_name] for img_name in img_name_list]

def get_img_post_path(root_dir,img_name):
    return os.path.join(root_dir, IMG_POST_FOLDER_NAME, img_name)

def get_img_path(root_dir, img_name):
    return os.path.join(root_dir, IMG_FOLDER_NAME, img_name)

def get_label_path(root_dir, img_name):
    return os.path.join(root_dir, ANNOT_FOLDER_NAME, img_name.replace('.jpg', label_suffix))

'''
    split = train
    split_val = val
    VOCdataloder(Object Detection)
'''
# split come from get_loader(test) or get_loaders(train,val)
class ImageDataset(data.Dataset):
    def __init__(self, root_dir, split='train', img_size=256, is_train=True,to_tensor=True):
        super(ImageDataset, self).__init__()
        # 'path to the root of LEVIR-CD dataset'
        self.root_dir = root_dir
        # 256
        self.img_size = img_size
        # train | test | val
        self.split = split  
        # root_dir + list + train|test|val + txt
        # 'path to the root of LEVIR-CD dataset'/list/train(test,val).txt
        self.list_path = os.path.join(self.root_dir, LIST_FOLDER_NAME, self.split +'.txt')
        # train | test | val name list(not image itself)
        self.img_name_list = load_img_name_list(self.list_path)
        # image num
        self.A_size = len(self.img_name_list)  
        # transform numpy to tensor
        self.to_tensor = to_tensor
        '''
            with_random_hflip=True,
            with_random_vflip=True,
            with_scale_random_crop=True,
            with_random_blur=True,
        '''
        if is_train:
            self.augm = CDDataAugmentation(
                img_size=self.img_size,
                with_random_hflip=True,
                with_random_vflip=True,
                with_scale_random_crop=True,
                with_random_blur=True,
            )
        else:
            self.augm = CDDataAugmentation(
                img_size=self.img_size
            )
    
    # DataLoader get data by __getitem__ function
    def __getitem__(self, index): 
        # get image from train|test|val
        # return os.path.join(root_dir, IMG_POST_FOLDER_NAME, img_name)
        # 'path to the root of LEVIR-CD dataset'/A(B)/img_name.png(jpg)
        name = self.img_name_list[index]
        A_path = get_img_path(self.root_dir, self.img_name_list[index % self.A_size])
        B_path = get_img_post_path(self.root_dir, self.img_name_list[index % self.A_size])
        # PIL object to numpy array(asarray op on array type directly which good for large data)
        img = np.asarray(Image.open(A_path).convert('RGB'))
        img_B = np.asarray(Image.open(B_path).convert('RGB'))
        [img, img_B], _ = self.augm.transform([img, img_B],[], to_tensor=self.to_tensor)
        return {'A': img, 'B': img_B, 'name': name}

    def __len__(self): 
        """Return the total number of images in the dataset."""
        return self.A_size

class CDDataset(ImageDataset):
    def __init__(self, root_dir, img_size, split='train', is_train=True, label_transform=None,
                 to_tensor=True):
        super(CDDataset, self).__init__(root_dir, img_size=img_size, split=split, is_train=is_train,
                                        to_tensor=to_tensor)
        self.label_transform = label_transform

    def __getitem__(self, index):
        name = self.img_name_list[index]
        # A pre B post L change
        A_path = get_img_path(self.root_dir, self.img_name_list[index % self.A_size])
        B_path = get_img_post_path(self.root_dir, self.img_name_list[index % self.A_size])
        # print("A_path :", A_path)
        # print("B_path :", B_path)
        img = np.asarray(Image.open(A_path).convert('RGB'))
        img_B = np.asarray(Image.open(B_path).convert('RGB'))
        L_path = get_label_path(self.root_dir, self.img_name_list[index % self.A_size])
        
        # dtype=np.uint8 0-255 (keep the origial values)

        # convert: w1x1 + w2x2 + w3x3 (w1 + w2 + w3 = 1)
        # (3,256,256) -> (256,256) | 255,255,255 -> 255 | 0,0,0 -> 0
        # LEVIR [0] or [0,255] RGB is the same as  gray


        # png and tif is Lossyless Compression jpg is Lossy Compression


        # print("L_path :", L_path)
        label = np.array(Image.open(L_path), dtype=np.uint8)
        # print("label shape RGB", label.shape, np.unique(label))
        
        # Get the label's R channel of RGB
        # label = np.array(Image.open(L_path), dtype=np.uint8)[:,:,0] 
        # print("label shape RGB [:,:,0]", label.shape, np.unique(label))

        label = np.array(Image.open(L_path).convert('L'), dtype=np.uint8)
        # print("label shape gray", label.shape, np.unique(label))

        #with open('LEVIR.txt', 'a') as file: 
        with open('DSIFN.txt', 'a') as file: 
            # 写入原始图像的尺寸和唯一值  
            file.write(f"L_path : {L_path}\n")  
            label = np.array(Image.open(L_path), dtype=np.uint8)  
            file.write(f"label shape RGB {label.shape} {np.unique(label)}\n")  
            
            # 写入取第一个通道后的尺寸和唯一值  
            # label = np.array(Image.open(L_path), dtype=np.uint8)[:,:,0]  
            # file.write(f"label shape RGB [:,:,0] {label.shape} {np.unique(label)}\n")  
            
            # 转换为灰度图后写入尺寸和唯一值  
            label = np.array(Image.open(L_path).convert('L'), dtype=np.uint8)  
            file.write(f"label shape gray {label.shape} {np.unique(label)}\n") 

        #sys.exit(0)

        # print("label shape before auge",label.shape)
        # change : 1 unchange : 0

        # self.label_transform == 'norm' is set in data_config.py
        if self.label_transform == 'norm':
            label = label // 255

        # print("label shape before auge", label.shape)

        # to_tensor expand channel dim in label
        [img, img_B], [label] = self.augm.transform([img, img_B], [label], to_tensor=self.to_tensor)

        # print("A, B, L_shape in DataLoader", img.shape, img_B.shape, label.shape, np.unique(label))

        return {'name': name, 'A': img, 'B': img_B, 'L': label}

