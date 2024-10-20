import os
import numpy as np
import matplotlib.pyplot as plt  
from models.networks import *
from misc.metric_tool import ConfuseMatrixMeter
from misc.logger_tool import Logger
from utils import de_norm, make_numpy_single_img
import utils
from torchvision.io.image import read_image
from torchcam.methods import GradCAM
from torchvision.transforms.functional import normalize, resize, to_pil_image
from torchcam.utils import overlay_mask

class CDEvaluator():
    def __init__(self, args, dataloader):
        self.dataloader = dataloader
        self.n_class = args.n_class
        self.batch_size = args.batch_size
        self.net_G = define_G(args=args, gpu_ids=args.gpu_ids)
        self.device = torch.device("cuda:%s" % args.gpu_ids[0] if torch.cuda.is_available() and len(args.gpu_ids)>0 else "cpu")
        self.running_metric = ConfuseMatrixMeter(n_class=self.n_class)
        logger_path = os.path.join(args.checkpoint_dir, 'log_test.txt')
        self.logger = Logger(logger_path)
        self.logger.write_dict_str(args.__dict__)
        self.epoch_acc = 0
        self.best_val_acc = 0.0
        self.best_epoch_id = 0
        self.G_pred = None
        self.pred_vis = None
        self.batch = None
        self.is_training = False
        self.batch_id = 0
        self.epoch_id = 0
        self.checkpoint_dir = args.checkpoint_dir
        self.vis_dir = args.vis_dir
        self.net_name = args.net_G

        if os.path.exists(self.checkpoint_dir) is False:
            os.mkdir(self.checkpoint_dir)
        if os.path.exists(self.vis_dir) is False:
            os.mkdir(self.vis_dir)

    def _load_checkpoint(self, checkpoint_name='best_ckpt.pt'):
        if os.path.exists(os.path.join(self.checkpoint_dir, checkpoint_name)):
            self.logger.write('loading last checkpoint...\n')
            checkpoint = torch.load(os.path.join(self.checkpoint_dir, checkpoint_name), map_location=self.device)
            model_G_state_dict = checkpoint['model_G_state_dict']
            self.net_G.load_state_dict(checkpoint['model_G_state_dict'], strict=False)
            self.net_G.to(self.device)
            self.best_val_acc = checkpoint['best_val_acc']
            self.best_epoch_id = checkpoint['best_epoch_id']
            self.logger.write('Eval Historical_best_acc = %.4f (at epoch %d)\n' %
                  (self.best_val_acc, self.best_epoch_id))
            self.logger.write('\n')

        else:
            raise FileNotFoundError('no such checkpoint %s' % checkpoint_name)

    def _visualize_constrast(self):
        gt = self.batch['L'].to(self.device).long()
        if self.net_name == "USSFC":
            pred = torch.where(self.G_pred > 0.5, torch.ones_like(self.G_pred), torch.zeros_like(self.G_pred)).long()
        else:    
            pred = torch.argmax(self.G_pred, dim=1, keepdim=True)

        temp = torch.zeros_like(pred, dtype = torch.int, device=self.device)
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


    def _visualize_single_img(self):
        vis_t1 = de_norm(self.batch['A'])
        vis_t2 = de_norm(self.batch['B'])

        print("t1", vis_t1.shape)
        print("t2", vis_t1.shape)
        print("pred", self._visualize_pred().shape, torch.unique(self._visualize_pred()))
        print("label", self.batch['L'].shape, torch.unique(self.batch['L']))

        for i in range(self.batch_size):
            print("self.batch_size", self.batch_size)
            vis_input = make_numpy_single_img(vis_t1[i])
            vis_input2 = make_numpy_single_img(vis_t2[i])

            vis_pred = make_numpy_single_img(self._visualize_pred()[i])
            vis_gt = make_numpy_single_img(self.batch['L'][i])
            vis_constrast = make_numpy_single_img(self._visualize_constrast()[i])

            file_t1_name = os.path.join(self.vis_dir, f'eval_{self.batch_id}_t1_{i}.jpg')
            file_t2_name = os.path.join(self.vis_dir, f'eval_{self.batch_id}_t2_{i}.jpg')
            file_pred_name = os.path.join(self.vis_dir, f'eval_{self.batch_id}_pred_{i}.jpg')
            file_gt_name = os.path.join(self.vis_dir, f'eval_{self.batch_id}_gt_{i}.jpg')
            file_constrast_name = os.path.join(self.vis_dir, f'eval_{self.batch_id}_constrast_{i}.jpg')

            plt.imsave(file_t1_name, vis_input)
            plt.imsave(file_t2_name, vis_input2)
            plt.imsave(file_pred_name, vis_pred)
            plt.imsave(file_gt_name, vis_gt)
            plt.imsave(file_constrast_name, vis_constrast)


    def _visualize_pred(self):
        if self.net_name == "USSFC":
            pred = torch.where(self.G_pred > 0.5, torch.ones_like(self.G_pred), torch.zeros_like(self.G_pred)).long()
        else:    
            pred = torch.argmax(self.G_pred, dim=1, keepdim=True)
        pred_vis = pred * 255
        return pred_vis

    def _update_metric(self):
        target = self.batch['L'].to(self.device).detach()
        G_pred = self.G_pred.detach()
        if self.net_name == "USSFC":
            G_pred = torch.where(G_pred > 0.5, torch.ones_like(G_pred), torch.zeros_like(G_pred)).long()
        else:    
            G_pred = torch.argmax(G_pred, dim=1)
        current_score = self.running_metric.update_cm(pr=G_pred.cpu().numpy(), gt=target.cpu().numpy())
        return current_score

    def _collect_running_batch_states(self):
        running_acc = self._update_metric()
        m = len(self.dataloader)

        if np.mod(self.batch_id, 100) == 1:
            message = 'Is_training: %s. [%d,%d],  running_mf1: %.5f\n' %\
                      (self.is_training, self.batch_id, m, running_acc)
            self.logger.write(message)
        # self._visualize_single_img()

    def _collect_epoch_states(self):
        scores_dict = self.running_metric.get_scores()
        np.save(os.path.join(self.checkpoint_dir, 'scores_dict.npy'), scores_dict)
        self.epoch_acc = scores_dict['mf1']
        with open(os.path.join(self.checkpoint_dir, '%s.txt' % (self.epoch_acc)),
                  mode='a') as file:
            pass
        message = ''
        for k, v in scores_dict.items():
            message += '%s: %.5f ' % (k, v)
        self.logger.write('%s\n' % message)  
        self.logger.write('\n')

    def _clear_cache(self):
        self.running_metric.clear()

    def _heat_map(self, checkpoint_name='best_ckpt.pt', save_dir='./heatmaps/Concat_GMDG_SYSU'):
        self._load_checkpoint(checkpoint_name)
        self.net_G.eval()

        os.makedirs(save_dir, exist_ok=True)
        list_path = 'SYSU/list/test.txt'
        image_A_path = 'SYSU/A'
        image_B_path = 'SYSU/B'
        image_label_path = 'SYSU/label'
        
        with open(list_path, 'r') as f:
            image_names = [line.strip() for line in f.readlines()]

        image_id = 0
        for self.batch_id, batch in enumerate(self.dataloader, 0):
            self.batch = batch
            img_in1 = batch['A'].to(self.device)
            img_in2 = batch['B'].to(self.device)

            target_layer = [self.net_G.BiFPN.Classifier.conv3]
            cam = GradCAM(self.net_G, target_layer)

            scores = self.net_G(img_in1, img_in2)[1]

            activation_map = cam(class_idx=1, scores=scores)
            activation_map = activation_map[0]
            print("act", activation_map.shape, torch.unique(activation_map))

            current_batch_size = len(batch['A'])
            for i in range(current_batch_size):
                image_name = image_names[image_id]
                img_A_file = os.path.join(image_A_path, image_name)
                img_B_file = os.path.join(image_B_path, image_name)
                img_label_file = os.path.join(image_label_path, image_name)

                # Load the images
                img_A = read_image(img_A_file)
                img_B = read_image(img_B_file)
                img_label = read_image(img_label_file)
                if image_label_path == 'WHU/label' or image_label_path == 'SYSU/label':
                    img_label = img_label.repeat(3, 1, 1)  # Convert to 3-channel image for visualization
                print("A,B,label", img_A.shape, img_B.shape, img_label.shape)

                result_A = overlay_mask(to_pil_image(img_A), to_pil_image(activation_map[i].squeeze(0), mode='F'), alpha=0.5, colormap='jet')
                result_B = overlay_mask(to_pil_image(img_B), to_pil_image(activation_map[i].squeeze(0), mode='F'), alpha=0.5, colormap='jet')
                result_label = overlay_mask(to_pil_image(img_label), to_pil_image(activation_map[i].squeeze(0), mode='F'), alpha=0.5, colormap='jet')

                # Save the result
                save_A_path = os.path.join(save_dir, f'A_{self.batch_id}_{image_id}_{image_name}')
                save_B_path = os.path.join(save_dir, f'B_{self.batch_id}_{image_id}_{image_name}')
                save_label_path = os.path.join(save_dir, f'L_{self.batch_id}_{image_id}_{image_name}')
                result_A.save(save_A_path)
                result_B.save(save_B_path)
                result_label.save(save_label_path)

                image_id = image_id + 1


    def _forward_pass(self, batch):
        self.batch = batch
        img_in1 = batch['A'].to(self.device)
        img_in2 = batch['B'].to(self.device)
        if self.net_name == 'DMINet':
            self.G_pred_1, self.G_pred_2, self.G_middle_1, self.G_middle_2 = self.net_G(img_in1, img_in2) 
            self.G_pred = self.G_pred_1 + self.G_pred_2
        elif self.net_name == "IFNet":
            self.G_pred_0, self.G_pred_1, self.G_pred_2, self.G_pred_3, self.G_pred_4  = self.net_G(img_in1, img_in2)
            self.G_pred = self.G_pred_0
        elif self.net_name == "GMDG":
            self.G_pred_1, self.G_pred_2  = self.net_G(img_in1, img_in2)
            self.G_pred = self.G_pred_2
        else:
            self.G_pred = self.net_G(img_in1, img_in2)

    def eval_models(self,checkpoint_name='best_ckpt.pt'):
        self._load_checkpoint(checkpoint_name)
        self.logger.write('Begin evaluation...\n')
        self._clear_cache()
        self.is_training = False
        self.net_G.eval()
        
        # Iterate over data.
        for self.batch_id, batch in enumerate(self.dataloader, 0):
            with torch.no_grad():
                self._forward_pass(batch)
            self._collect_running_batch_states()
        self._collect_epoch_states()

