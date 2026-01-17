import torch
import argparse
import torch.nn as nn
import torch.utils.data as Data
import torch.backends.cudnn as cudnn
from scipy.io import loadmat
import matplotlib.pyplot as plt
from MsAFEFN import Model
from scipy.io import savemat
from sklearn.metrics import confusion_matrix
import scipy.io as io
import numpy as np
import time
import os
import random
from utility import *
torch.autograd.set_detect_anomaly(True)
import visdom

#---------------------------To ensure your code runs smoothly, please refer to the notes. Thank you.
#---------------------------rngsd = You need to enter the seed.
#-------------------------- Replace the code that loads the data and parameters

parser = argparse.ArgumentParser("HSI")
parser.add_argument('--Dataset', choices=['data1, data2, data3, data4'], default='data1', help='dataset to use')
parser.add_argument('--Flag_test', choices=['train', 'test'], default='train', help='testing mark')
parser.add_argument('--Mode', choices=['ClassificationModel'], default='ClassificationModel', help='mode choice')
parser.add_argument('--Gpu_id', default='0', help='gpu id')
parser.add_argument('--Seed', type=int, default=, help='number of seed')
parser.add_argument('--Batch_size', type=int, default=, help='number of batch size')
parser.add_argument('--Test_freq', type=int, default=, help='number of evaluation')
parser.add_argument('--Patches', type=int, default=, help='number of patches')
parser.add_argument('--Epoches', type=int, default=, help='epoch number')
parser.add_argument('--Learning_rate', type=float, default=, help='learning rate')
parser.add_argument('--Gamma', type=float, default=, help='gamma')
parser.add_argument('--Weight_decay', type=float, default=, help='weight_decay')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.Gpu_id)
# -------------------------------------------------------------------------------
def print_args(args):
    for k, v in zip(args.keys(), args.values()):
        print("{0}: {1}".format(k, v))
# -------------------------------------------------------------------------------
print("=================================== Parameters ===================================")
print_args(vars(args))
# -------------------------------------------------------------------------------
# Parameter Setting
args.Seed = rngsd
np.random.seed(args.Seed)
random.seed(args.Seed)
torch.manual_seed(args.Seed)
torch.cuda.manual_seed(args.Seed)
cudnn.deterministic = True
cudnn.benchmark = False
# normalize data by band norm
norm_type = 'bandwise'  # 'pixelwise', 'bandwise'
# prepare data
if args.Dataset == 'data1':
    folder_data = ''
    data_HS = loadmat(folder_data + '')
    data_DSM1 = loadmat(folder_data + '')
    label_TR = loadmat(folder_data + '')
    label_TE = loadmat(folder_data + '')
    gt = loadmat(folder_data + '')['']
    label_TR = label_TR['']
    label_TE = label_TE['']
    input_HS = mynorm(data_HS[''], norm_type)
    input_DSM1 = np.expand_dims(data_DSM1[''], axis=-1)
    height, width, band1 = input_HS.shape
    _, _, band2 = input_DSM1.shape
    input_MultiModal = np.concatenate((input_HS, input_DSM1), axis=2)
    band_MultiModal = [band1, band2]
else:
    raise ValueError("Unknown dataset")
num_classes = np.max(label_TR)

folder_log = '' + str(args.Patches) + '/'

if not os.path.exists(folder_log):
    os.makedirs(folder_log)
# -------------------------------------------------------------------------------
# obtain train positions

select_type = 'normal'
total_pos_TR, number_TR = select_points(label_TR, num_classes, select_type)
# obtain test positions
select_type = 'normal'
total_pos_TE, number_TE = select_points(label_TE, num_classes, select_type)
## test
mirror_image = mirror_hsi(height, width, np.sum(band_MultiModal), input_MultiModal, patch=args.Patches)
mirror_image2 = mirror_hsi(height, width, band1, input_HS, patch=args.Patches)

# obtain train data from train positions
x_TR_patch, y_TR = prepare_data(mirror_image, label_TR, np.sum(band_MultiModal), total_pos_TR, patch=args.Patches)
# obtain test data from test positions
x_TE_patch, y_TE = prepare_data(mirror_image, label_TE, np.sum(band_MultiModal), total_pos_TE, patch=args.Patches)
# -------------------------------------------------------------------------------
# load data
x_TR = torch.from_numpy(x_TR_patch.transpose(0, 3, 1, 2)).type(torch.FloatTensor)  # [#TR, band, patch*patch]
y_TR = torch.from_numpy(y_TR).type(torch.LongTensor)
Label_TR = Data.TensorDataset(x_TR, y_TR)
x_TE = torch.from_numpy(x_TE_patch.transpose(0, 3, 1, 2)).type(torch.FloatTensor)
y_TE = torch.from_numpy(y_TE).type(torch.LongTensor)
Label_TE = Data.TensorDataset(x_TE, y_TE)
Label_TR_loader = Data.DataLoader(Label_TR, batch_size=args.Batch_size, shuffle=True)
Label_TE_loader = Data.DataLoader(Label_TE, batch_size=args.Batch_size, shuffle=True)
# -------------------------------------------------------------------------------
# create model
model = ClassificationModel(
    Classes=num_classes,
    NC=band1,
    NCLidar=band2,
    patch_size=args.Patches
)
model = model.cuda()

# criterion
criterion = nn.CrossEntropyLoss().cuda()
# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=args.Learning_rate, weight_decay=args.Weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=, gamma=args.Gamma)  # You need to go to Settings.

# -------------------------------------------------------------------------------
if args.Flag_test == 'test':

    print("=================================== Testing ===================================")

    PATH = r'.pt'
    model.load_state_dict(torch.load(PATH))
    model.eval()
    pre_total = []
    NCLidar = band2
    NC = band1
    # You need to go to Setting
    part =
    m = height
    n = width
    l = band1 + band2
    patch_size = args.Patches

    pred_all = np.empty((m * n, 1), dtype='float32')
    # print(f"Pad width: {pad_width}")
    number = m * n // part
    print(number)
    for i in range(number):
        D = np.empty((part, l, patch_size, patch_size), dtype='float32')
        count = 0
        for j in range(i * part, (i + 1) * part):
            row = j // n
            col = j - row * n

            patch = mirror_image[row:(row + patch_size), col:(col + patch_size), :]
            # print(f"Patch size: {patch_size }, Patch shape: {patch.shape}")
            # print(f"Patch shape: {patch.shape}")
            patch = np.reshape(patch, (patch_size * patch_size, l))
            patch = np.transpose(patch)
            patch = np.reshape(patch, (l, patch_size, patch_size))
            D[count, :, :, :] = patch
            count += 1
        print(i)
        temp = torch.from_numpy(D)
        temp = temp.cuda()
        temp2 = model(temp[:, 0:NC, :, :], temp[:, NC:, :, :])
        temp3 = torch.max(temp2, 1)[1].squeeze()
        pred_all[i * part:(i + 1) * part, 0] = temp3.cpu()
        del temp, temp2, temp3, D
    if (i + 1) * part < m * n:
        D = np.empty((m * n - (i + 1) * part, l, patch_size, patch_size), dtype='float32')
        count = 0
        for j in range((i + 1) * part, m * n):
            row = j // n
            col = j - row * n

            patch = mirror_image[row:(row + patch_size), col:(col + patch_size), :]

            patch = np.reshape(patch, (patch_size * patch_size, l))
            patch = np.transpose(patch)
            patch = np.reshape(patch, (l, patch_size, patch_size))
            D[count, :, :, :] = patch
            count += 1
        temp = torch.from_numpy(D)
        temp = temp.cuda()
        temp2 = model(temp[:, 0:NC, :, :], temp[:, NC:, :, :])
        temp3 = torch.max(temp2, 1)[1].squeeze()
        pred_all[(i + 1) * part:m * n, 0] = temp3.cpu()
        del temp, temp2, temp3, D
    # You can replace it with the required format, resolution, and storage location for output settings.
    plt.show()
    print(">>> Inference finished!")
    print("=================================== Results ===================================")
    np.set_printoptions(precision=2, suppress=True)
elif args.Flag_test == 'train':
    best_checkpoint = {"OA_TE": 0.50}
    print("=================================== Training ===================================")
    tic = time.time()
    for epoch in range(args.Epoches):
        model.train().cuda()
        train_acc, train_obj, tar_t, pre_t = train_epoch_MM(model, Label_TR_loader, criterion, optimizer, band1)
        scheduler.step()
        OA_TR, AA_TR, Kappa_TR, CA_TR = output_metric(tar_t, pre_t)
        if (epoch % args.Test_freq == 0) | (epoch == args.Epoches - 1):
            print("Epoch: {:03d} train_loss: {:.4f}, train_OA: {:.2f}".format(epoch + 1, train_obj, OA_TR * 100))
            model.eval()
            test_acc, test_obj, tar_v, pre_v = valid_epoch_MM(model, Label_TE_loader, criterion, optimizer, band1)
            OA_TE, AA_TE, Kappa_TE, CA_TE = output_metric(tar_v, pre_v)
            print("Epoch: {:03d} test_loss: {:.4f}, test_OA: {:.2f}, test_AA: {:.2f}, test_Kappa: {:.4f}".format(
                epoch + 1, train_obj, OA_TE * 100, AA_TE * 100, Kappa_TE))
            if OA_TE * 100 > best_checkpoint['OA_TE']:
                best_checkpoint = {'epoch': epoch, 'OA_TE': OA_TE * 100, 'AA_TE': AA_TE * 100, 'Kappa_TE': Kappa_TE,
                                   'CA_TE': CA_TE * 100}
            PATH = folder_log + args.Dataset + str(epoch) + '.pt'
            torch.save(model.state_dict(), PATH)
    toc = time.time()
    runtime = toc - tic
    print(">>> Training finished!")
    print(">>> Running time: {:.2f}".format(runtime))
    print("=================================== Results ===================================")
    print(">>> The peak performance in terms of OA is achieved at epoch", best_checkpoint['epoch'])
    print("OA: {:.2f} | AA: {:.2f} | Kappa: {:.4f}".format(best_checkpoint['OA_TE'], best_checkpoint['AA_TE'],
                                                           best_checkpoint['Kappa_TE']))
    np.set_printoptions(precision=2, suppress=True)
    print("CA: ", best_checkpoint['CA_TE'])
    output_txt_path = os.path.join(folder_log, 'precision.txt')
    write_message = "Patch size {}, weight decay {}, learning rate {}, the best epoch {}, OA {}, AA {}, Kappa {}, run time {}".format(
        args.Patches, args.Weight_decay, args.Learning_rate, best_checkpoint['epoch'],
        round(best_checkpoint['OA_TE'], 2), round(best_checkpoint['AA_TE'], 2), round(best_checkpoint['Kappa_TE'], 4),
        round(runtime, 2))
    output_txt_file = open(output_txt_path, "a")
    now = time.strftime("%c")
    output_txt_file.write(
        '=================================== Precision Log (%s) ===================================\n' % now)
    output_txt_file.write('%s\n' % write_message)
    output_txt_file.close()

