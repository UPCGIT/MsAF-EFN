import torch
from sklearn.metrics import confusion_matrix
import scipy.io as io
import numpy as np
import random
from ptflops import get_model_complexity_info

# -------------------------------------------------------------------------------
def compute_model_stats(model, input_size, print_stat=False, save_to_file=None):

    with torch.cuda.device(0):
        flops, params = get_model_complexity_info(model, input_size, as_strings=True,
                                                  print_per_layer_stat=print_stat, verbose=False)
    if save_to_file:
        with open(save_to_file, 'w') as f:
            f.write(f"Model Parameters: {params}\n")
            f.write(f"Model FLOPs: {flops}\n")
    return params, flops


# -------------------------------------------------------------------------------
def label2color(label, data_name):
    w, h = label.shape
    im = np.zeros((w, h, 3), dtype=np.uint8)
    data_name = data_name.lower()
    if data_name == 'uni':
        map = []
    else:
        return None
    for i in range(w):
        for j in range(h):
            index = int(label[i, j])
            if index == 0:
                im[i, j, :] = np.uint8([0, 0, 0])
                continue
            im[i, j, :] = np.uint8(map[index - 1])
    im = np.uint8(im)
    classif = np.uint8(np.zeros((w, h, 3)))
    classif[:, :, 0] = im[:, :, 0]
    classif[:, :, 1] = im[:, :, 1]
    classif[:, :, 2] = im[:, :, 2]
    return classif

# -------------------------------------------------------------------------------
def select_points(mask, num_classes, select_type, ratio=None, rngsd1=None):
    select_size = []
    select_pos = {}

    if select_type == 'normal':

        for i in range(num_classes):
            each_class0 = np.argwhere(mask != (0))
            each_class = np.argwhere(mask == (i + 1))
            select_size.append(each_class.shape[0])
            select_pos[i] = each_class

        total_select_pos = select_pos[0]
        for i in range(1, num_classes):
            total_select_pos = np.r_[total_select_pos, select_pos[i]]  # (695,2)
        total_select_pos = total_select_pos.astype(int)

    elif select_type == 'random':

        for i in range(num_classes):
            each_class = []
            each_class = np.argwhere(mask == (i + 1))
            lengthi = each_class.shape[0]
            num = range(1, lengthi)

            random.seed(rngsd1)
            nums = random.sample(num, int(lengthi * ratio))
            select_size.append(len(nums))
            select_pos[i] = each_class[nums, :]

        total_select_pos = select_pos[0]
        for i in range(1, num_classes):
            total_select_pos = np.r_[total_select_pos, select_pos[i]]  # (695,2)
        total_select_pos = total_select_pos.astype(int)

    return total_select_pos, select_size


# -------------------------------------------------------------------------------
def mirror_hsi(height, width, band, input_normalize, patch=5):
    padding = patch // 2
    mirror_hsi = np.zeros((height + 2 * padding, width + 2 * padding, band), dtype=float)
    mirror_hsi[padding:(padding + height), padding:(padding + width), :] = input_normalize
    for i in range(padding):
        mirror_hsi[padding:(height + padding), i, :] = input_normalize[:, padding - i - 1, :]
    for i in range(padding):
        mirror_hsi[padding:(height + padding), width + padding + i, :] = input_normalize[:, width - 1 - i, :]
    for i in range(padding):
        mirror_hsi[i, :, :] = mirror_hsi[padding * 2 - i - 1, :, :]
    for i in range(padding):
        mirror_hsi[height + padding + i, :, :] = mirror_hsi[height + padding - 1 - i, :, :]
    print("Patch size: {}".format(patch))
    print("Padded image shape: [{0},{1},{2}]".format(mirror_hsi.shape[0], mirror_hsi.shape[1], mirror_hsi.shape[2]))
    return mirror_hsi

# -------------------------------------------------------------------------------
# def gain_neighborhood_pixel(mirror_image, point, i, patch=5):hy1  1
# def gain_neighborhood_pixel(mirror_image, point, i, patch=13):
# x = point[i, 0]2
# y = point[i, 1]3
# print(f"x: {x}, y: {y}")
# temp_image = mirror_image[x:(x + patch), y:(y + patch), :]4
# print(f"Shape of temp_image: {temp_image.shape}")
# manual_x = max(0, x - 6)
# manual_y = max(0, y - 6)
# manual_temp_image = mirror_image[manual_x:(manual_x + 13), manual_y:(manual_y + 13), :]
# print(f"Shape of manual temp_image: {manual_temp_image.shape}")
# return temp_image3
# -------------------------------------------------------------------------------
def gain_neighborhood_pixel(mirror_image, point, i, patch=5):
    x = point[i, 0]
    y = point[i, 1]
    # print(f"x: {x}, y: {y}")
    temp_image = mirror_image[x:(x + patch), y:(y + patch), :]

    return temp_image

# -------------------------------------------------------------------------------
def prepare_data(mirror_image, label, band, select_point, patch):
    x_select = np.zeros((select_point.shape[0], patch, patch, band), dtype=float)
    y_select = np.zeros(select_point.shape[0], dtype=float)
    for i in range(select_point.shape[0]):
        x_select[i, :, :, :] = gain_neighborhood_pixel(mirror_image, select_point, i, patch)
        # print(f"Shape of extracted neighborhood: {x_select[i, :, :, :].shape}")13*13*126
        y_select[i] = label[select_point[i][0], select_point[i][1]] - 1
    return x_select, y_select


# -------------------------------------------------------------------------------
def train_and_test_label(number_train, number_test, num_classes):
    y_train = []
    y_test = []
    for i in range(num_classes):
        for j in range(number_train[i]):
            y_train.append(i)
        for k in range(number_test[i]):
            y_test.append(i)

    y_train = np.array(y_train)
    y_test = np.array(y_test)
    print("y_train: shape = {}, type = {}".format(y_train.shape, y_train.dtype))
    print("y_test: shape = {}, type = {}".format(y_test.shape, y_test.dtype))
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    return y_train, y_test

# -------------------------------------------------------------------------------
class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt

# -------------------------------------------------------------------------------
def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res, target, pred.squeeze()

# -------------------------------------------------------------------------------
def train_epoch_MM(model, train_loader, criterion, optimizer, band1):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])
    for batch_idx, (batch_data, batch_target) in enumerate(train_loader):
        batch_data = batch_data.cuda()
        batch_target = batch_target.cuda()
        optimizer.zero_grad()
        # print(f'HSI data shape: {batch_data[:, 0:band1, :, :].shape}')
        # print(f'LiDAR data shape: {batch_data[:, band1:, :, :].shape}')
        batch_pred = model(batch_data[:, 0:band1, :, :], batch_data[:, band1:, :, :])
        loss = criterion(batch_pred, batch_target)
        loss.backward()
        optimizer.step()
        prec1, t, p = accuracy(batch_pred, batch_target, topk=(1,))
        n = batch_data.shape[0]
        objs.update(loss.data, n)
        top1.update(prec1[0].data, n)
        tar = np.append(tar, t.data.cpu().numpy())
        pre = np.append(pre, p.data.cpu().numpy())
    return top1.avg, objs.avg, tar, pre

# -------------------------------------------------------------------------------
def train_epoch_MM_one(model, train_loader, criterion, optimizer, band1):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])
    for batch_idx, (batch_data, batch_target) in enumerate(train_loader):
        batch_data = batch_data.cuda()
        batch_target = batch_target.cuda()
        optimizer.zero_grad()
        batch_pred = model(batch_data)
        loss = criterion(batch_pred, batch_target)
        loss.backward()
        optimizer.step()

        prec1, t, p = accuracy(batch_pred, batch_target, topk=(1,))
        n = batch_data.shape[0]
        objs.update(loss.data, n)
        top1.update(prec1[0].data, n)
        tar = np.append(tar, t.data.cpu().numpy())
        pre = np.append(pre, p.data.cpu().numpy())
    return top1.avg, objs.avg, tar, pre

# -------------------------------------------------------------------------------
def valid_epoch_MM(model, valid_loader, criterion, optimizer, band1):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])
    for batch_idx, (batch_data, batch_target) in enumerate(valid_loader):
        batch_data = batch_data.cuda()
        batch_target = batch_target.cuda()

        batch_pred = model(batch_data[:, 0:band1, :, :], batch_data[:, band1:, :, :])

        loss = criterion(batch_pred, batch_target)

        prec1, t, p = accuracy(batch_pred, batch_target, topk=(1,))
        n = batch_data.shape[0]
        objs.update(loss.data, n)
        top1.update(prec1[0].data, n)
        tar = np.append(tar, t.data.cpu().numpy())
        pre = np.append(pre, p.data.cpu().numpy())

    return top1.avg, objs.avg, tar, pre

# -------------------------------------------------------------------------------
def valid_epoch_MM_one(model, valid_loader, criterion, optimizer, band1):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])
    for batch_idx, (batch_data, batch_target) in enumerate(valid_loader):
        batch_data = batch_data.cuda()
        batch_target = batch_target.cuda()

        batch_pred = model(batch_data)

        loss = criterion(batch_pred, batch_target)

        prec1, t, p = accuracy(batch_pred, batch_target, topk=(1,))
        n = batch_data.shape[0]
        objs.update(loss.data, n)
        top1.update(prec1[0].data, n)
        tar = np.append(tar, t.data.cpu().numpy())
        pre = np.append(pre, p.data.cpu().numpy())

    return top1.avg, objs.avg, tar, pre

# -------------------------------------------------------------------------------
def test_epoch(model, test_loader, criterion, optimizer):
    pre = np.array([])
    for batch_idx, (batch_data, batch_target) in enumerate(test_loader):
        batch_data = batch_data.cuda()
        batch_target = batch_target.cuda()

        batch_pred = model(batch_data)

        _, pred = batch_pred.topk(1, 1, True, True)
        pp = pred.squeeze()
        pre = np.append(pre, pp.data.cpu().numpy())
    return pre


# -------------------------------------------------------------------------------
def output_metric(tar, pre):
    matrix = confusion_matrix(tar, pre)
    OA, AA_mean, Kappa, AA = cal_results(matrix)
    return OA, AA_mean, Kappa, AA


# -------------------------------------------------------------------------------
def cal_results(matrix):
    shape = np.shape(matrix)
    number = 0
    sum = 0
    AA = np.zeros([shape[0]], dtype=float)
    for i in range(shape[0]):
        number += matrix[i, i]
        AA[i] = matrix[i, i] / np.sum(matrix[i, :])
        sum += np.sum(matrix[i, :]) * np.sum(matrix[:, i])
    OA = number / np.sum(matrix)
    AA_mean = np.mean(AA)
    pe = sum / (np.sum(matrix) ** 2)
    Kappa = (OA - pe) / (1 - pe)
    return OA, AA_mean, Kappa, AA


# -------------------------------------------------------------------------------
def mynorm(data, norm_type):
    data_norm = np.zeros(data.shape)
    if norm_type == 'bandwise':
        for i in range(data.shape[2]):
            data_max = np.max(data[:, :, i])
            data_min = np.min(data[:, :, i])
            data_norm[:, :, i] = (data[:, :, i] - data_min) / (data_max - data_min)
    elif norm_type == 'pixelwise':
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                data_max = np.max(data[i, j, :])
                data_min = np.min(data[i, j, :])
                data_norm[i, j, :] = (data[i, j, :] - data_min) / (data_max - data_min)
    return data_norm
