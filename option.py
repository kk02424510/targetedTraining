'''train'''
import argparse
import numpy as np

model = 'single_full_vgg16'
#Hybird_v1 single_full_fix Hybird_v2_vgg single_full_vgg16
dataset = 'Kodak'
# BSDS500 DIV2K LIVE classic5 Kodak
loss = '1*MSE+0.5*R_loss' 
# 1*MSE 1*SSIM 1*MSE+2000*SSIM 1*MSE+0.1*R_loss
codec = 'JPEG'
# JPEG JPEG2000 WebP BPG

# var = np.linspace(1,0.1,10).tolist()
patchSize = 224
margin = 4
ch = 6
if codec == 'BPG':
    # imageQFs = list(range(51,22,-3))
    imageQFs = [51,42,33,24]
else:
    # imageQFs = [5, 7, 10] + list(range(15,36,10)) + list(range(50,100,15))
    # imageQFs = [80]
    imageQFs = [15,50,95]

vgg_l=4
modelQF  = 6
conv_g = 1
per_para = 0.1

n_clusters = 4
pd_flag = False

# Training settings
parser = argparse.ArgumentParser(description="PyTorch")

# parser.add_argument('--R_weight', type=str, default=var,
#                     help='R_weight')

# grouping specifications
parser.add_argument('--n_clusters', type=str, default=n_clusters,
                    help='clusting K group')
parser.add_argument('--pd_flag', action="store_true",default=pd_flag,
                    help='padding or not')

# Compression framework specifications
parser.add_argument('--patchSize', type=int, default=patchSize)
parser.add_argument('--margin', type=int, default=margin)
parser.add_argument('--ch', type=int, default=ch)

parser.add_argument('--codec', type=str, default=codec)
parser.add_argument('--imageQFs', type=list, default=imageQFs)
parser.add_argument('--modelQF', type=int, default=modelQF)
parser.add_argument('--conv_g', type=int, default=conv_g)
parser.add_argument('--vgg_l', type=int, default=vgg_l)

# Hardware specifications
parser.add_argument('--n_threads', type=int, default=0,
                    help='number of threads for data loading')
parser.add_argument("--no_cuda", action="store_true",
                    help="disable cuda?")
parser.add_argument('--n_GPUs', type=int, default=1,
                    help='number of GPUs')
parser.add_argument('--seed', type=int, default=50,
                    help='random seed')

# Data specifications
parser.add_argument('--dataset', type=str, default='{}'.format(dataset),
                    help='dataset name')
parser.add_argument('--data_path', type=str, default='D:/kysu/dataset/{}/raw/'.format(dataset),
                    help='dataset directory')

# Model specifications
parser.add_argument('--model', default='{}'.format(model), help='model name')
# parser.add_argument('--pre_train', type=str, default='./model/single_noDSC_v6_ch{}_1.pt'.format(ch),
#                     help='pre-trained model directory')
parser.add_argument('--pre_train', type=str, default='.',
                    help='pre-trained model directory')
# Training specifications
parser.add_argument('--test_every', type=int, default=30,
                    help='do test per every N batches')
parser.add_argument('--epochs', type=int, default=40,
                    help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=1,
                    help='input batch size for training')
parser.add_argument('--clip', type=float, default=0.6,
                    help="Clipping Gradients. Default=0")
parser.add_argument('--itercontrol', action='store_true',
                    help="fixed iter times or not")
                    
# Optimization specifications
parser.add_argument('--lr', type=float, default=0.1,
                    help='learning rate')
parser.add_argument('--lr_decay', type=int, default=40,
                    help='learning rate decay per N epochs')
parser.add_argument('--decay_type', type=str, default='step_15',
                    help='learning rate decay type, ex: step or step_20_40')
parser.add_argument('--gamma', type=float, default=0.1,
                    help='learning rate decay factor for step decay')
parser.add_argument('--optimizer', default='ADAM',
                    choices=('SGD', 'ADAM', 'RMSprop'),
                    help='optimizer to use (SGD | ADAM | RMSprop)')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--beta1', type=float, default=0.9,
                    help='ADAM beta1')
parser.add_argument('--beta2', type=float, default=0.999,
                    help='ADAM beta2')
parser.add_argument('--weight_decay', type=float, default=1e-3,
                    help='weight decay')
parser.add_argument('--epsilon', type=float, default=1e-8,
                    help='ADAM epsilon for numerical stability')

# Loss specifications
parser.add_argument('--loss', type=str, default='{}'.format(loss),
                    help='loss function configuration')

# Log specifications
parser.add_argument('--save', type=str, default='{}_ch{}_{}_{}_{}_patch{}_pd{}_vgg{}all_F2'.format(model, ch, dataset, loss.replace('*',''), codec,patchSize,margin,vgg_l,n_clusters),
                    help='folder name to save')
parser.add_argument('--print_model', action='store_true',
                    help='print model')
parser.add_argument('--print_every', type=int, default=30,
                    help='how many batches to wait before logging training status')
parser.add_argument('--save_results', action='store_false',
                    help='save output results')
parser.add_argument('--save_pt', action='store_true',
                    help='save model parameters')

# unuse specifications
parser.add_argument('--multi', action='store_true',
                    help='train multi channel')
parser.add_argument('--adjustable_clip', action='store_true',
                    help="")
parser.add_argument('--rgb_range', type=int, default=255,
                    help='maximum value of RGB')
parser.add_argument('--colortype', type=str, default='yuv',
                    help='RGB, yuv')


opt = parser.parse_args()
