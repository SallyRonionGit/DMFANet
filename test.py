from util.metric_tool import ConfuseMatrixMeter
import torch
from option import Options
from data.cd_dataset import DataLoader
from model.create_model import create_model
from tqdm import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt  
import sys
from torchvision.io.image import read_image
from torchcam.methods import GradCAM
from torchvision.transforms.functional import normalize, resize, to_pil_image
from torchcam.utils import overlay_mask

def de_norm(tensor_data):
    mean = torch.tensor([0.485, 0.456, 0.406], device=tensor_data.device).view(1, -1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=tensor_data.device).view(1, -1, 1, 1)
    return tensor_data * std + mean

def make_numpy_single_img(tensor_data):
    tensor_data = tensor_data.detach()
    vis = np.array(tensor_data.cpu()).transpose((1, 2, 0))
    print("utils", vis.shape)
    if vis.shape[2] == 1:
        vis = vis.squeeze()
        vis = np.stack([vis, vis, vis], axis=-1)
    vis = np.clip(vis, a_min=0.0, a_max=1.0)
    return vis

def _visualize_constrast(pred, gt):
        gt = gt.to(pred.device)
        print(pred.device, gt.device)
        temp = torch.zeros_like(pred, dtype = torch.int, device=pred.device)
        temp[(pred == 0) & (gt == 0)] = 0  
        temp[(pred == 1) & (gt == 1)] = 1
        temp[(pred == 1) & (gt == 0)] = 2  
        temp[(pred == 0) & (gt == 1)] = 3 

        b, _, h, w = temp.shape
        constrast_vis = torch.zeros((b, 3, h, w), dtype=torch.float32)  

        for i in range(b):  
            for j in range(h):  
                for k in range(w):  
                    pixel_value = temp[i, 0, j, k].item()  
                    if pixel_value == 0:  
                        constrast_vis[i, :, j, k] = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)  # black real unchange
                    elif pixel_value == 1:  
                        constrast_vis[i, :, j, k] = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)  # white real change
                    elif pixel_value == 2:  
                        constrast_vis[i, :, j, k] = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)  # red false detect  
                    elif pixel_value == 3:  
                        constrast_vis[i, :, j, k] = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)  # green miss detect
        print("constrast_vis", constrast_vis.shape)
        return constrast_vis

def _visualize_single_img(t1, t2, label, pred, batch_size, vis_dir, batch_id):
        print(os.path.dirname(os.path.abspath(__file__)))
        os.makedirs(vis_dir, exist_ok=True)
        # de norm 
        vis_t1 = de_norm(t1)
        vis_t2 = de_norm(t2)
        pred = pred.unsqueeze(1)
        label = label.unsqueeze(1)

        print("t1", vis_t1.shape)
        print("t2", vis_t1.shape)
        print("pred", pred.shape, torch.unique(pred))
        print("label", label.shape, torch.unique(label))

        for i in range(batch_size):
            print("batch_size", batch_size)
            
            vis_input = make_numpy_single_img(vis_t1[i])
            vis_input2 = make_numpy_single_img(vis_t2[i])
            vis_pred = make_numpy_single_img(pred[i])
            vis_gt = make_numpy_single_img(label[i])
            vis_constrast = make_numpy_single_img(_visualize_constrast(pred, label)[i])

            file_t1_name = os.path.join(vis_dir, f'eval_{batch_id}_t1_{i}.jpg')
            file_t2_name = os.path.join(vis_dir, f'eval_{batch_id}_t2_{i}.jpg')
            file_pred_name = os.path.join(vis_dir, f'eval_{batch_id}_pred_{i}.jpg')
            file_gt_name = os.path.join(vis_dir, f'eval_{batch_id}_gt_{i}.jpg')
            file_constrast_name = os.path.join(vis_dir, f'eval_{batch_id}_constrast_{i}.jpg')

            print(file_t1_name, file_t2_name, file_pred_name, file_gt_name, file_constrast_name)
            
            plt.imsave(file_t1_name, vis_input)
            plt.imsave(file_t2_name, vis_input2)
            plt.imsave(file_pred_name, vis_pred)
            plt.imsave(file_gt_name, vis_gt)
            # sys.exit(0)
            plt.imsave(file_constrast_name, vis_constrast)

def _heat_map():
        save_dir = './heatmap/LEVIR'
        list_path = './datasets/LEVIR/test/test.txt'
        image_A_path = './datasets/LEVIR/test/A'
        image_B_path = './datasets/LEVIR/test/B'
        image_label_path = './datasets/LEVIR/test/label'
        
        with open(list_path, 'r') as f:
            image_names = [line.strip() for line in f.readlines()]

        opt = Options().parse()
        opt.phase = 'test'
        test_loader = DataLoader(opt)
        test_data = test_loader.load_data()

        test_size = len(test_loader)
        print("#testing images = %d" % test_size)

        opt.load_pretrain = True
        model = create_model(opt)
        model.eval()
        # sys.exit(0)

        tbar = tqdm(test_data, ncols=80)
        image_id = 0
        for i, _data in enumerate(tbar):
            print(model.detector.p4_to_p3.selfattn.h)
            target_layer = [model.detector.p4_to_p3]
            cam = GradCAM(model, target_layer)
            
            val_pred = model.inference_cam(_data['img1'].cuda(), _data['img2'].cuda())[0]
            # print(val_pred[0].shape)
            # sys.exit(0)
            activation_map = cam(class_idx=1, scores=val_pred)
            activation_map = activation_map[0]
            print("act", activation_map.shape, torch.unique(activation_map))
            
            # update metric
            # val_target = _data['cd_label'].detach()
            # val_pred = torch.argmax(val_pred.detach(), dim=1)

            for j in range(test_size):
                image_name = image_names[image_id]
                img_A_file = os.path.join(image_A_path, image_name)
                img_B_file = os.path.join(image_B_path, image_name)
                img_label_file = os.path.join(image_label_path, image_name)

                # Load the images
                img_A = read_image(img_A_file)
                img_B = read_image(img_B_file)
                img_label = read_image(img_label_file)
                
                img_label = img_label.repeat(3, 1, 1)  # Convert to 3-channel image for visualization
                print("A,B,label", img_A.shape, img_B.shape, img_label.shape)

                result_A = overlay_mask(to_pil_image(img_A), to_pil_image(activation_map[j].squeeze(0), mode='F'), alpha=0.5, colormap='jet')
                result_B = overlay_mask(to_pil_image(img_B), to_pil_image(activation_map[j].squeeze(0), mode='F'), alpha=0.5, colormap='jet')
                result_label = overlay_mask(to_pil_image(img_label), to_pil_image(activation_map[j].squeeze(0), mode='F'), alpha=0.5, colormap='jet')

                # Save the result
                save_A_path = os.path.join(save_dir, f'A_{i}_{image_id}_{image_name}')
                save_B_path = os.path.join(save_dir, f'B_{i}_{image_id}_{image_name}')
                save_label_path = os.path.join(save_dir, f'L_{i}_{image_id}_{image_name}')
                result_A.save(save_A_path)
                result_B.save(save_B_path)
                result_label.save(save_label_path)
                sys.exit(0)
                image_id = image_id + 1


if __name__ == '__main__':
    # _heat_map()
    opt = Options().parse()
    opt.phase = 'test'
    test_loader = DataLoader(opt)
    test_data = test_loader.load_data()
    test_size = len(test_loader)
    print("#testing images = %d" % test_size)

    opt.load_pretrain = True
    model = create_model(opt)

    tbar = tqdm(test_data, ncols=80)
    total_iters = test_size
    running_metric = ConfuseMatrixMeter(n_class=2)
    running_metric.clear()

    model.eval()
    with torch.no_grad():
        for i, _data in enumerate(tbar):
            val_pred = model.inference(_data['img1'].cuda(), _data['img2'].cuda())
            # update metric
            val_target = _data['cd_label'].detach()
            val_pred = torch.argmax(val_pred.detach(), dim=1)
            _ = running_metric.update_cm(pr=val_pred.cpu().numpy(), gt=val_target.cpu().numpy())
            # _visualize_single_img(_data['img1'], _data['img2'], val_target, val_pred, 16, './vis/' + opt.name, i)

        val_scores = running_metric.get_scores()
        message = '(phase: %s) ' % (opt.phase)
        for k, v in val_scores.items():
            message += '%s: %.3f ' % (k, v * 100)
        print(message)

