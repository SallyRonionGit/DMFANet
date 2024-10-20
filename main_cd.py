from argparse import ArgumentParser
import torch
from models.trainer import *
import torch.multiprocessing
from utils import str2bool
torch.multiprocessing.set_sharing_strategy('file_system')
print(torch.cuda.is_available())

"""
the main function for training the CD networks
"""
def train(args):
    dataloaders = utils.get_loaders(args)
    model = CDTrainer(args=args, dataloaders=dataloaders)
    model.train_models()

def test(args):
    from models.evaluator import CDEvaluator
    dataloader = utils.get_loader(args.data_name, img_size=args.img_size,
                                  batch_size=args.batch_size, is_train=False,
                                  split='test')
    model = CDEvaluator(args=args, dataloader=dataloader)
    model.eval_models()

if __name__ == '__main__':
    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--project_name', default='test', type=str)
    parser.add_argument('--log_path_root', default='tensorboard', type=str)
    parser.add_argument('--checkpoint_root', default='checkpoints', type=str)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--dataset', default='CDDataset', type=str)
    parser.add_argument('--data_name', default='SYSU', type=str)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--split', default="train", type=str)
    parser.add_argument('--split_val', default="val", type=str)
    parser.add_argument('--img_size', default=256, type=int)
    parser.add_argument('--n_class', default=2, type=int)
    parser.add_argument('--net_G', default='GMDG', type=str,
                        help='FC_EF | SNUNet | '
                             'BIT | ChangeFormer'
                             'DMINet| GMDG'
                             'base_transformer_pos_s4_dd8')
    parser.add_argument('--loss', default='GMDG', type=str)
    parser.add_argument('--loss_DS', default=True, type=str2bool, help='Deep Supervision')
    parser.add_argument('--optimizer', default='sgd', type=str, help='adamw, adam, sgd')
    parser.add_argument('--lr', default=0.01, type=float) 
    parser.add_argument('--max_epochs', default=200, type=int)
    parser.add_argument('--max_steps', type=int, default=40000, help='Max number of batch_id')
    parser.add_argument('--lr_policy', default='linear', type=str,
                        help='linear | step | poly')
    parser.add_argument('--lr_decay_iters', default=200, type=int)

    args = parser.parse_args()
    utils.get_device(args)
    print(args.gpu_ids)

    # tensorboard log dir
    args.log_path_dir = os.path.join(args.log_path_root, args.project_name)
    os.makedirs(args.log_path_dir, exist_ok=True)

    # change detection visualization
    args.vis_dir = os.path.join('vis_train', args.project_name)
    os.makedirs(args.vis_dir, exist_ok=True)
    args.checkpoint_dir = os.path.join(args.checkpoint_root, args.project_name)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    train(args)
    test(args)


