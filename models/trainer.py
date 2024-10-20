import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import os
import sys
import utils
from models.networks import *
import torch
import torch.optim as optim
from misc.metric_tool import ConfuseMatrixMeter
from models.losses import cross_entropy, BCD_loss, SNUNet_Loss, SEIFNet_Loss, TFI_GR_Loss, IFNet_Loss, GMDG_Loss, HFANet_Loss
import models.losses as losses
from misc.logger_tool import Logger, Timer
from utils import de_norm
from thop import profile

class CDTrainer():
    def __init__(self, args, dataloaders):

        self.dataloaders = dataloaders
        self.n_class = args.n_class
        self.net_G = define_G(args=args, gpu_ids=args.gpu_ids)
        self.net_name = args.net_G
        self.lr_policy = args.lr_policy
        self.device = torch.device("cuda:%s" % args.gpu_ids[0] if torch.cuda.is_available() and len(args.gpu_ids)>0
                                   else "cpu")
        print("device", self.device)
        self.lr = args.lr
        if args.optimizer == "sgd":
                self.optimizer_G = optim.SGD(self.net_G.parameters(), lr=self.lr,
                                             momentum=0.9,
                                             weight_decay=5e-4)
        elif args.optimizer == "adam":
            # 1e-4
            self.optimizer_G = optim.Adam(self.net_G.parameters(), lr=self.lr, 
                                          betas=(0.9, 0.99), eps=1e-08, weight_decay=5e-4)
        elif args.optimizer == "adamw":
            self.optimizer_G = optim.AdamW(self.net_G.parameters(), lr=self.lr,
                                           betas=(0.9, 0.999), weight_decay=0.01)

        self.running_metric = ConfuseMatrixMeter(n_class=2)
        self.writer = SummaryWriter(log_dir=args.log_path_dir)
        logger_path = os.path.join(args.checkpoint_dir, 'log.txt')
        self.logger = Logger(logger_path)
        self.logger.write_dict_str(args.__dict__)
        self.timer = Timer()
        self.batch_size = args.batch_size
        
        self.epoch_acc = 0
        self.best_val_acc = 0.0
        self.best_epoch_id = 0
        self.epoch_to_start = 0
        self.max_num_epochs = args.max_epochs
        self.max_num_steps = args.max_steps
        self.global_step = 0

        self.steps_per_epoch = len(dataloaders['train'])
        self.poly_max_epoch = int(np.ceil(self.max_num_steps / self.steps_per_epoch))
        self.cur_step = 0

        self.total_steps = (self.max_num_epochs - self.epoch_to_start)*self.steps_per_epoch

        self.G_pred = None
        self.pred_vis = None
        self.batch = None
        self.G_loss = None
        
        self.loss_DS = args.loss_DS

        self.is_training = False
        self.batch_id = 0
        self.epoch_id = 0
        
        self.lr_factor = 1.
        self.checkpoint_dir = args.checkpoint_dir
        self.vis_dir = args.vis_dir

        if args.loss == 'ce':
            self._pxl_loss = cross_entropy
        elif args.loss == 'bce':
            self._pxl_loss = losses.binary_ce
        elif args.loss == 'ce + dice':
            self._pxl_loss = SEIFNet_Loss
            # self._pxl_loss = BCD_loss    
        elif args.loss == 'SNUNet':
            self._pxl_loss = SNUNet_Loss
        elif args.loss == 'TFI-GR':
            self._pxl_loss = TFI_GR_Loss
        elif args.loss == 'SEIFNet':
            self._pxl_loss = SEIFNet_Loss
        elif args.loss == 'IFNet':
            self._pxl_loss = IFNet_Loss
        elif args.loss == 'HFANet':
            self._pxl_loss = HFANet_Loss
        elif args.loss == 'GMDG':
            self._pxl_loss = GMDG_Loss
            # self._pxl_loss = cross_entropy
        elif args.loss == 'USSFC':
            self._pxl_loss = nn.BCELoss()
        else:
            raise NotImplemented(args.loss)
        
        self.exp_lr_scheduler_G = get_scheduler(self.optimizer_G, args)
        self.VAL_ACC = np.array([], np.float32)
        if os.path.exists(os.path.join(self.checkpoint_dir, 'val_acc.npy')):
            self.VAL_ACC = np.load(os.path.join(self.checkpoint_dir, 'val_acc.npy'))
  
        self.TRAIN_ACC = np.array([], np.float32)
        if os.path.exists(os.path.join(self.checkpoint_dir, 'train_acc.npy')):
            self.TRAIN_ACC = np.load(os.path.join(self.checkpoint_dir, 'train_acc.npy'))

        if os.path.exists(self.checkpoint_dir) is False:
            os.mkdir(self.checkpoint_dir)

        if os.path.exists(self.vis_dir) is False:
            os.mkdir(self.vis_dir)

    def _load_checkpoint(self, ckpt_name='last_ckpt.pt'):
        if os.path.exists(os.path.join(self.checkpoint_dir, ckpt_name)):
            self.logger.write('loading last checkpoint...\n')
            checkpoint = torch.load(os.path.join(self.checkpoint_dir, ckpt_name),
                                    map_location=self.device)
            self.net_G.load_state_dict(checkpoint['model_G_state_dict'])
            self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
            if self.lr_policy != 'poly':
                self.exp_lr_scheduler_G.load_state_dict(
                    checkpoint['exp_lr_scheduler_G_state_dict'])
            
            self.net_G.to(self.device)

            self.epoch_to_start = checkpoint['epoch_id'] + 1
            self.best_val_acc = checkpoint['best_val_acc']
            self.best_epoch_id = checkpoint['best_epoch_id']
            self.logger.write('Epoch_to_start = %d, Historical_best_acc = %.4f (at epoch %d)\n' %
                  (self.epoch_to_start, self.best_val_acc, self.best_epoch_id))
            self.logger.write('\n')
        else:
            print('training from scratch...')
            
    '''
    In Python, a function name or variable name that starts with an underscore (_) usually indicates that it is for "internal use" only, 
    meaning it is primarily designed for use within a module or a class, rather than being part of an API for external calling.
    '''
    def _timer_update(self):
        self.global_step = (self.epoch_id-self.epoch_to_start) * self.steps_per_epoch + self.batch_id
        self.timer.update_progress((self.global_step + 1) / self.total_steps)
        est = self.timer.estimated_remaining()
        imps = (self.global_step + 1) * self.batch_size / self.timer.get_stage_elapsed()
        return imps, est
    
    def _visualize_pred(self):
        if self.net_name == "USSFC":
            pred = torch.where(self.G_pred > 0.5, torch.ones_like(self.G_pred), torch.zeros_like(self.G_pred)).long()
        else:    
            pred = torch.argmax(self.G_pred, dim=1, keepdim=True)
        pred_vis = pred * 255
        return pred_vis
    
    def _save_checkpoint(self, ckpt_name):
        if self.lr_policy == 'poly':
            torch.save({
            'epoch_id': self.epoch_id,
            'best_val_acc': self.best_val_acc,
            'best_epoch_id': self.best_epoch_id,
            'model_G_state_dict': self.net_G.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict()
            }, os.path.join(self.checkpoint_dir, ckpt_name))
        else:
            torch.save({
                'epoch_id': self.epoch_id,
                'best_val_acc': self.best_val_acc,
                'best_epoch_id': self.best_epoch_id,
                'model_G_state_dict': self.net_G.state_dict(),
                'optimizer_G_state_dict': self.optimizer_G.state_dict(),
                'exp_lr_scheduler_G_state_dict': self.exp_lr_scheduler_G.state_dict(),
            }, os.path.join(self.checkpoint_dir, ckpt_name))

    def _update_lr_schedulers(self):
        self.exp_lr_scheduler_G.step()

    def _update_poly_lr(self, step, lr_factor=1):
        cur_step = step
        max_step = self.steps_per_epoch * self.poly_max_epoch
        print("self.epoch_id, cur_step, max_step", self.epoch_id, cur_step, max_step)
        lr = self.lr * (1 - cur_step * 1.0 / max_step) ** 0.9
        if self.epoch_id == 0 and step < 200:
            lr = self.lr * 0.9 * (step + 1) / 200 + 0.1 * self.lr  
        print(self.lr * 0.9 * (step + 1) / 200 + 0.1 * self.lr)
        lr *= lr_factor
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        return lr

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
        m = len(self.dataloaders['train'])
        if self.is_training is False:
            m = len(self.dataloaders['val'])

        imps, est = self._timer_update()
        if np.mod(self.batch_id, 100) == 1:
            if self.lr_policy == 'poly':
                message = 'Is_training: %s. [%d,%d][%d,%d], imps: %.2f, est: %.2fh, G_loss: %.5f, running_mf1: %.5f\n' %\
                        (self.is_training, self.epoch_id, self.poly_max_epoch-1, self.batch_id, m,
                        imps*self.batch_size, est,
                        self.G_loss.item(), running_acc)
                self.logger.write(message)
            else:
                message = 'Is_training: %s. [%d,%d][%d,%d], imps: %.2f, est: %.2fh, G_loss: %.5f, running_mf1: %.5f\n' %\
                        (self.is_training, self.epoch_id, self.max_num_epochs-1, self.batch_id, m,
                        imps*self.batch_size, est,
                        self.G_loss.item(), running_acc)
                self.logger.write(message)

        
        if np.mod(self.batch_id, 500) == 1:
            vis_input = utils.make_numpy_grid(de_norm(self.batch['A']))
            vis_input2 = utils.make_numpy_grid(de_norm(self.batch['B']))
            vis_pred = utils.make_numpy_grid(self._visualize_pred())

            vis_gt = utils.make_numpy_grid(self.batch['L'])
            vis = np.concatenate([vis_input, vis_input2, vis_pred, vis_gt], axis=0)
            vis = np.clip(vis, a_min=0.0, a_max=1.0)
            file_name = os.path.join(
                self.vis_dir, 'istrain_'+str(self.is_training)+'_'+
                              str(self.epoch_id)+'_'+str(self.batch_id)+'.jpg')
            plt.imsave(file_name, vis)
        

    def _collect_epoch_states(self):
        scores = self.running_metric.get_scores()
        self.epoch_acc = scores['mf1']
        self.logger.write('Is_training: %s. Epoch %d / %d, epoch_mF1= %.5f\n' %
              (self.is_training, self.epoch_id, self.max_num_epochs-1, self.epoch_acc))
        message = ''
        for k, v in scores.items():
            message += '%s: %.5f ' % (k, v)
        self.logger.write(message+'\n')
        self.logger.write('\n')

    def _update_checkpoints(self):
        self._save_checkpoint(ckpt_name='last_ckpt.pt')
        self.logger.write('Lastest model updated. Epoch_acc=%.4f, Historical_best_acc=%.4f (at epoch %d)\n'
              % (self.epoch_acc, self.best_val_acc, self.best_epoch_id))
        self.logger.write('\n')

        if self.epoch_acc > self.best_val_acc:
            self.best_val_acc = self.epoch_acc
            self.best_epoch_id = self.epoch_id
            self._save_checkpoint(ckpt_name='best_ckpt.pt')
            self.logger.write('*' * 10 + 'Best model updated!\n')
            self.logger.write('\n')

    def _update_training_acc_curve(self):
        self.TRAIN_ACC = np.append(self.TRAIN_ACC, [self.epoch_acc])
        np.save(os.path.join(self.checkpoint_dir, 'train_acc.npy'), self.TRAIN_ACC)

    def _update_val_acc_curve(self):
        self.VAL_ACC = np.append(self.VAL_ACC, [self.epoch_acc])
        np.save(os.path.join(self.checkpoint_dir, 'val_acc.npy'), self.VAL_ACC)

    def _clear_cache(self):
        self.running_metric.clear()

    def _forward_pass(self, batch):
        self.batch = batch
        img_in1 = batch['A'].to(self.device)
        img_in2 = batch['B'].to(self.device)

        if self.loss_DS == True:
            if self.net_name == "IFNet":
                self.G_pred_0, self.G_pred_1, self.G_pred_2, self.G_pred_3, self.G_pred_4  = self.net_G(img_in1, img_in2)
                self.G_pred = self.G_pred_0
            elif self.net_name == 'DMINet':
                self.G_pred_1, self.G_pred_2, self.G_middle_1, self.G_middle_2 = self.net_G(img_in1, img_in2) 
                self.G_pred = self.G_pred_1 + self.G_pred_2
            elif self.net_name == 'A2Net':
                self.G_pred_1, self.G_pred_2, self.G_pred_3, self.G_pred_4  = self.net_G(img_in1, img_in2)   
                self.G_pred = self.G_pred_1
            elif self.net_name == 'GMDG':
                self.G_pred_1, self.G_pred_2  = self.net_G(img_in1, img_in2)   
                self.G_pred = self.G_pred_2
        else:
            self.G_pred = self.net_G(img_in1, img_in2)
            
    def _backward_G(self):
        gt = self.batch['L'].to(self.device).float()
        if self.loss_DS == True:
            if self.net_name == "IFNet":
                self.G_loss_1 = self._pxl_loss(self.G_pred, gt)
                gt = gt.float()
                gt_1 = F.interpolate(gt, scale_factor=1/2, mode='bilinear')
                gt_2 = F.interpolate(gt, scale_factor=1 / 4, mode='bilinear')
                gt_3 = F.interpolate(gt, scale_factor=1 / 8, mode='bilinear')
                gt_4 = F.interpolate(gt, scale_factor=1 / 16, mode='bilinear')
                self.G_loss2 = self._pxl_loss(self.G_pred_1, gt_1) + self._pxl_loss(self.G_pred_2,gt_2) + self._pxl_loss(self.G_pred_3, gt_3)+ self._pxl_loss(self.G_pred_4, gt_4)
                self.G_loss = self.G_loss_1 + self.G_loss2
            elif self.net_name == "DMINet":
                self.G_loss =  self._pxl_loss(self.G_pred_1, gt) + self._pxl_loss(self.G_pred_2, gt) + 0.5*(self._pxl_loss(self.G_middle_1, gt)+self._pxl_loss(self.G_middle_2, gt))
            elif self.net_name == "A2Net":
                self.G_loss = self._pxl_loss(self.G_pred, gt) + self._pxl_loss(self.G_pred_2, gt) + self._pxl_loss(self.G_pred_3, gt) + self._pxl_loss(self.G_pred_4, gt)
            elif self.net_name == "GMDG":
                self.G_loss = self._pxl_loss(self.G_pred, gt) + self._pxl_loss(self.G_pred_1, gt) 
        else:
            self.G_loss = self._pxl_loss(self.G_pred, gt)
        self.G_loss.backward()    
    
    def model_profile_mac_params(self):
        pre_img = torch.randn(1, 3, 256, 256).to(self.device)
        post_img = torch.randn(1, 3, 256, 256).to(self.device)
        mac, prarms = profile(self.net_G, (pre_img,post_img))
        print(f"FLOPs: {mac*2 / 1e9:.12f} G")    
        print(f"Number of parameters: {prarms / 1e6:.12f} M")

    def train_models(self):
        self._load_checkpoint()
        if self.lr_policy == 'poly':
            for self.epoch_id in range(self.epoch_to_start, self.poly_max_epoch):
                print("self.poly_max_epoch", self.poly_max_epoch)
                self._clear_cache()
                self.is_training = True
                self.net_G.train()  
                self.logger.write('lr: %0.10f\n' % self.optimizer_G.param_groups[0]['lr'])
                epoch_train_loss = 0.0
                for self.batch_id, batch in enumerate(self.dataloaders['train'], 0):
                    lr = self._update_poly_lr(self.batch_id + self.cur_step, self.lr_factor)
                    print("lr", lr)
                    folder_path = './results'  
                    file_name = 'lr.txt'  
                     
                    if not os.path.exists(folder_path):  
                        os.makedirs(folder_path)  
                    
                    with open(os.path.join(folder_path, file_name), 'a') as f:    
                            f.write(str(lr) + "  " + str(self.batch_id) + '\n')
                    
                    self._forward_pass(batch)
                    self.optimizer_G.zero_grad()
                    self._backward_G()
                    self.optimizer_G.step()
                    batch_train_loss = self.G_loss.to("cpu").detach().numpy()
                    epoch_train_loss = epoch_train_loss + batch_train_loss
                    self._collect_running_batch_states()
                    self._timer_update()

                epoch_train_loss = epoch_train_loss / len(self.dataloaders['train'])
                self.writer.add_scalar("Train Loss/epoch", epoch_train_loss, self.epoch_id)

                self._collect_epoch_states()
                self._update_training_acc_curve()
                self.cur_step = self.cur_step + self.steps_per_epoch
                self.logger.write('Begin evaluation...\n')
                self._clear_cache()
                self.is_training = False
                self.net_G.eval()

                epoch_eval_loss = 0.0
                for self.batch_id, batch in enumerate(self.dataloaders['val'], 0):
                    with torch.no_grad():
                        self._forward_pass(batch)
                        gt = self.batch['L'].to(self.device).long()
                        self.G_loss = self._pxl_loss(self.G_pred, gt)
                        epoch_eval_loss = epoch_eval_loss + self.G_loss.to("cpu").detach().numpy() 
                    self._collect_running_batch_states()

                epoch_eval_loss = epoch_eval_loss / len(self.dataloaders['val'])
                self.writer.add_scalar("Eval Loss/epoch", epoch_eval_loss, self.epoch_id)
                self.writer.close()

                self._collect_epoch_states()
                self._update_val_acc_curve() 
                self._update_checkpoints()
        else:
            for self.epoch_id in range(self.epoch_to_start, self.max_num_epochs):
                self._clear_cache()
                self.is_training = True
                self.net_G.train()  
                self.logger.write('lr: %0.7f\n' % self.optimizer_G.param_groups[0]['lr'])

                epoch_train_loss = 0.0
                for self.batch_id, batch in enumerate(self.dataloaders['train'], 0):
                    self._forward_pass(batch)
                    self.optimizer_G.zero_grad()
                    self._backward_G()
                    self.optimizer_G.step()

                    batch_train_loss = self.G_loss.to("cpu").detach().numpy()
                    epoch_train_loss = epoch_train_loss + batch_train_loss

                    self._collect_running_batch_states()
                    self._timer_update()

                epoch_train_loss = epoch_train_loss / len(self.dataloaders['train'])
                self.writer.add_scalar("Train Loss/epoch", epoch_train_loss, self.epoch_id)

                self._collect_epoch_states()
                self._update_training_acc_curve()
                self._update_lr_schedulers()

                self.logger.write('Begin evaluation...\n')
                self._clear_cache()
                self.is_training = False
                self.net_G.eval()

                epoch_eval_loss = 0.0
                for self.batch_id, batch in enumerate(self.dataloaders['val'], 0):
                    with torch.no_grad():
                        self._forward_pass(batch)
                        gt = self.batch['L'].to(self.device).float()
                        self.G_loss = self._pxl_loss(self.G_pred, gt)
                        epoch_eval_loss = epoch_eval_loss + self.G_loss.to("cpu").detach().numpy() 
                    self._collect_running_batch_states()

                epoch_eval_loss = epoch_eval_loss / len(self.dataloaders['val'])
                self.writer.add_scalar("Eval Loss/epoch", epoch_eval_loss, self.epoch_id)
                self.writer.close()

                self._collect_epoch_states()

                self._update_val_acc_curve() 
                self._update_checkpoints()


