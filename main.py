import torch
import warnings
import argparse
from progressbar import *
from torchinfo import summary
from torchview import draw_graph

from models import InkDetection
from utils.log_utils import Logger
from utils.val_utils import val_model
from utils.test_utils import test_model
from utils.other_utils import setup_seed
from utils.train_utils import train_model
from utils.data_utils import make_dataset, load_train_dataset, load_test_dataset

# 命令行参数
parser = argparse.ArgumentParser(description='main')
# 数据参数
parser.add_argument('--make_datasets', type=bool, default=False)
parser.add_argument('--remake', type=bool, default=True)
parser.add_argument('--data_path', type=str, default='/kaggle/vesuvius-challenge/')
parser.add_argument('--size', type=int, default=512)
parser.add_argument('--stride', type=int, default=256)
parser.add_argument('--datasets_path', type=str, default='/kaggle/Inputs/data_ink/')
parser.add_argument('--num_workers', type=int, default=0)
# 模型参数
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--device', type=str, default=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
parser.add_argument('--start_deep', type=int, default=20)
parser.add_argument('--deep_dim', type=int, default=16)
parser.add_argument('--loss_weight', default=[0.5, 0.5], help='loss weight')
# SE Block
parser.add_argument('--use_se', default=False, help='use se block or not')
# CNN3D
parser.add_argument('--use_cnn3d', default=False, help='use cnn3d block or not')
# Attn UNet
parser.add_argument('--use_unet', default=True, help='use unet block or not')
parser.add_argument('--unet_channels', default=[16, 32, 64, 128, 256], help='unet channels')
parser.add_argument('--use_attn', default=True, help='use attention block')
# SegFormer
parser.add_argument('--my_segformer', default=False, help='use segformer block or not')
parser.add_argument('--library_segformer', default=False, help='use segformer block or not')
# 训练参数
parser.add_argument('--gpu_ids', type=str, default='0')
parser.add_argument('--seed', type=int, default=3407)
parser.add_argument('--val_num', type=int, default=93)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--val_epoch', type=int, default=1)
parser.add_argument('--loss_th', type=float, default=0.6)
parser.add_argument('--threshold', type=float, default=0.5)
# 其他
parser.add_argument('--save_path', type=str, default='/kaggle/Outputs/InkDetection/')
parser.add_argument('--subfolder', type=str, default='unet+attn')
args = parser.parse_args()

# GPU
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
# 保存路径
args.save_path = os.path.join(args.save_path, args.subfolder) + "/"
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)
# 日志
sys.stdout = Logger("logs.log", stream=sys.stdout, path=args.save_path)
sys.stderr = Logger("logs.log", stream=sys.stderr, path=args.save_path)
# 警告
warnings.filterwarnings("ignore")
# 随机种子
setup_seed(args.seed)

def main():
    """ 主函数 """
    # 打印参数
    print("################################################################## [INFO] args:")
    for k, v in args.__dict__.items():
        print(k.ljust(20), ':', v)
    # 数据集
    make_dataset(args)
    # 数据迭代器
    train_loader, val_loader = load_train_dataset(args)
    test_loader = load_test_dataset(args)
    # 模型
    net = InkDetection(args)
    print("################################################################## [INFO] model:")
    summary(net,
            input_size=(args.batch_size, args.deep_dim, args.size, args.size),
            device=args.device,
            depth=5)
    draw_graph(net,
               input_size=(args.batch_size, args.deep_dim, args.size, args.size),
               device=args.device, save_graph=True, filename='InkDetection',
               expand_nested=True, depth=5,
               directory=args.save_path)
    # 优化器
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr,
                                  betas=(0.9, 0.999), weight_decay=1e-2)
    # 学习律调整器
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
    # 进度条
    progress = ProgressBar(widgets=['\t\t\t\tProgress: ', Percentage(), ' ', Bar('#'), ' ', Timer(),
                                    ' ', ETA(), ' ', FileTransferSpeed()])
    print("################################################################## [INFO] training:")
    train_loss = []
    val_list = []
    val_loss = []
    for epoch in progress(range(args.epochs)):
        print("[INFO] Epoch {}/{}:".format(epoch + 1, args.epochs))
        train_loss = train_model(train_loader, net, optimizer, epoch, args, train_loss)
        # scheduler.step()
        if (epoch + 1) % args.val_epoch == 0 or epoch == args.epochs - 1 or epoch == 0:
            val_list, val_loss = val_model(val_loader, net, epoch, args, val_list, val_loss)
    print("################################################################## [INFO] testing:")
    test_model(test_loader, net, args, train_loss, val_loss, val_list)

if __name__ == '__main__':
    main()
