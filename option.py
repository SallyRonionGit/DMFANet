import argparse
import torch

class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser()

    def init(self):
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0. use -1 for CPU')
        self.parser.add_argument('--name', type=str, default='WHU')
        self.parser.add_argument('--dataroot', type=str, default='./datasets')
        self.parser.add_argument('--dataset', type=str, default='WHU')
        self.parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='models are saved here')

        self.parser.add_argument('--result_dir', type=str, default='./results', help='results are saved here')
        self.parser.add_argument('--load_pretrain', type=bool, default=False)

        self.parser.add_argument('--phase', type=str, default='train')
        self.parser.add_argument('--input_size', type=int, default=256)
        self.parser.add_argument('--backbone', type=str, default='mobilenetv2')
        self.parser.add_argument('--fpn', type=str, default='fpn')
        self.parser.add_argument('--fpn_channels', type=int, default=128)
        self.parser.add_argument('--deform_groups', type=int, default=4)
        self.parser.add_argument('--gamma_mode', type=str, default='SE')
        self.parser.add_argument('--beta_mode', type=str, default='contextgatedconv')
        self.parser.add_argument('--num_heads', type=int, default=1)
        self.parser.add_argument('--num_points', type=int, default=8)
        self.parser.add_argument('--kernel_layers', type=int, default=1)
        self.parser.add_argument('--init_type', type=str, default='kaiming_normal')
        self.parser.add_argument('--alpha', type=float, default=0.25)
        self.parser.add_argument('--gamma', type=int, default=4, help='gamma for Focal loss')
        self.parser.add_argument('--dropout_rate', type=float, default=0.1)

        self.parser.add_argument('--batch_size', type=int, default=16)
        self.parser.add_argument('--num_epochs', type=int, default=200)
        self.parser.add_argument('--warmup_epochs', type=int, default=20)
        self.parser.add_argument('--num_workers', type=int, default=4, help='#threads for loading data')
        self.parser.add_argument('--lr', type=float, default=5e-4)#1e-5
        self.parser.add_argument('--weight_decay', type=float, default=5e-4)

    def parse(self):
        self.init()
        self.opt = self.parser.parse_args()

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        return self.opt
