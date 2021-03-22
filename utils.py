import torch
import math
import torch.nn.functional as F
import random
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from PIL import Image
from rasterize import mydrawPNG
import numpy as np
import torchvision
from matplotlib import pyplot as plt

def to_Five_Point(sketch_points, max_seq_length):
    len_seq = len(sketch_points[:, 0])
    new_seq = np.zeros((max_seq_length, 5))
    new_seq[0:len_seq, :2] = sketch_points[:, :2]
    new_seq[0:len_seq, 3] = sketch_points[:, 2]
    new_seq[0:len_seq, 2] = 1 - new_seq[0:len_seq, 3]
    new_seq[(len_seq - 1):, 4] = 1
    new_seq[(len_seq - 1), 2:4] = 0
    new_seq = np.concatenate((np.zeros((1, 5)), new_seq), axis=0)
    return new_seq, len_seq

def to_normal_strokes(big_stroke):
  """Convert from stroke-5 format (from sketch-rnn paper) back to stroke-3."""
  l = 0
  for i in range(len(big_stroke)):
      if big_stroke[i, 4] > 0:
        l = i
        break
  if l == 0:
      l = len(big_stroke)
  result = np.zeros((l, 3))
  result[:, 0:2] = big_stroke[0:l, 0:2]
  result[:, 2] = big_stroke[0:l, 3]
  result[-1, 2] = 1.
  return result

def sketch_point_Inp(sketch, step, count):
    # xmin, xmax = sketch[:, 0].min(), sketch[:, 0].max()
    # ymin, ymax = sketch[:, 1].min(), sketch[:, 1].max()
    #
    # sketch[:, 0] = ((sketch[:, 0] - xmin) / float(xmax - xmin)) * (255. - 60.) + 30.
    # sketch[:, 1] = ((sketch[:, 1] - ymin) / float(ymax - ymin)) * (255. - 60.) + 30.
    sketch = sketch.astype(np.int64)

    fig = plt.figure(frameon=False, figsize=(2.56, 2.56))
    xlim = [0, 255]
    ylim = [0, 255]

    stroke_list = np.split(sketch[:, :2], np.where(sketch[:, 2])[0] + 1, axis=0)[:-1]

    for stroke in stroke_list:
        stroke = stroke[:, :2].astype(np.int64)
        plt.plot(stroke[:, 0], stroke[:, 1], '.', linestyle='solid', linewidth=1.0, markersize=15)
        # plt.scatter(stroke[:, 0], stroke[:, 1])
    plt.gca().invert_yaxis();
    plt.axis('off')

    plt.savefig('./gen/' + str(step) + '_' + str(count) + 'Inp.png', bbox_inches='tight',
                pad_inches=0.3, dpi=1200)

# def sketch_point(sample_targ, step, count):
#     fig = plt.figure(frameon=False, figsize=(2.56, 2.56))
#     xlim = [0, 255]
#     ylim = [0, 255]
#     for stroke in sample_targ:
#         stroke = stroke[:, :2].astype(np.int64)
#         plt.plot(stroke[:, 0], stroke[:, 1], '.', linestyle='solid', linewidth=1.0, markersize=5)
#         # plt.scatter(stroke[:, 0], stroke[:, 1])
#         plt.gca().invert_yaxis();
#         plt.axis('off')
#
#     plt.savefig('./gen/' + str(step) + '_' + str(count) + 'redraw.png', bbox_inches='tight',
#                 pad_inches=0, dpi=1200)
#
#
# def eval_redraw_CVPR(target, output, seq_len, step):
#     count = 0
#     for sample_targ, sample_gen, seq in zip(target, output, seq_len):
#         count = count + 1
#         sample_gen = sample_gen.cpu().numpy()[:seq]
#         sample_targ = sample_targ.cpu().numpy()
#         sample_targ = to_stroke_list(to_normal_strokes(sample_targ))
#         sample_gen = to_stroke_list(to_normal_strokes(sample_gen))
#
#         fig = plt.figure(frameon=False, figsize=(2.56, 2.56))
#         xlim = [0, 255]
#         ylim = [0, 255]
#         for stroke in sample_targ:
#             stroke = stroke[:, :2].astype(np.int64)
#             plt.plot(stroke[:, 0], stroke[:, 1], '.', linestyle='solid', linewidth=1.0, markersize=5)
#             # plt.scatter(stroke[:, 0], stroke[:, 1])
#             plt.gca().invert_yaxis();
#             plt.axis('off')
#
#         plt.savefig('./gen/'+ str(step) + '_' + str(count) + 'redraw.png', bbox_inches='tight',
#                     pad_inches=0, dpi=1200)


def eval_redraw(target, output, seq_len, step, date_time_folder, type, num_print=8, side=1):

    batch_redraw = []

    for sample_targ, sample_gen, seq in zip(target[:num_print], output[:num_print], seq_len[:num_print]):

        sample_gen = sample_gen.cpu().numpy()[:seq]
        sample_targ = sample_targ.cpu().numpy()
        sample_targ = to_normal_strokes(sample_targ)
        sample_gen = to_normal_strokes(sample_gen)


        sample_gen[:, :2] = np.round(sample_gen[:, :2] * side)
        image_gen, _ = mydrawPNG(sample_gen)
        image_gen = Image.fromarray(image_gen).convert('RGB')


        sample_targ[:, :2] = np.round(sample_targ[:, :2] * side)
        image_targ, _ = mydrawPNG(sample_targ)
        image_targ = Image.fromarray(image_targ).convert('RGB')

        batch_redraw.append(torch.from_numpy(np.array(image_targ)).permute(2, 0, 1))
        batch_redraw.append(torch.from_numpy(np.array(image_gen)).permute(2, 0, 1))

    batch_redraw = torch.stack(batch_redraw).float()
    torchvision.utils.save_image(batch_redraw, date_time_folder + '/sketch_Viz/' + type + '_' + str(step) + '_.jpg', normalize=True, nrow=2)

def to_stroke_list(sketch):
    ## sketch: an `.npz` style sketch from QuickDraw
    sketch = np.vstack((np.array([0, 0, 0]), sketch))
    sketch[:, :2] = np.cumsum(sketch[:, :2], axis=0)

    # range normalization
    xmin, xmax = sketch[:, 0].min(), sketch[:, 0].max()
    ymin, ymax = sketch[:, 1].min(), sketch[:, 1].max()

    sketch[:, 0] = ((sketch[:, 0] - xmin) / float(xmax - xmin)) * (255. - 60.) + 30.
    sketch[:, 1] = ((sketch[:, 1] - ymin) / float(ymax - ymin)) * (255. - 60.) + 30.
    sketch = sketch.astype(np.int64)

    stroke_list = np.split(sketch[:, :2], np.where(sketch[:, 2])[0] + 1, axis=0)

    if stroke_list[-1].size == 0:
        stroke_list = stroke_list[:-1]

    if len(stroke_list) == 0:
        stroke_list = [sketch[:, :2]]
        # print('error')
    return stroke_list

class Jigsaw_Utility(object):

    """
    https://research.wmz.ninja/articles/2018/03/teaching-a-neural-network-to-solve-jigsaw-puzzles.html
    https://github.com/HeddaCohenIndelman/Learning-Gumbel-Sinkhorn-Permutations-w-Pytorch/blob/master/my_sinkhorn_ops.py#L343
    """
    def __init__(self, num_split = 3, image_size =(256,256)):
        self.perm_inds = []
        for a in range(num_split):
            for b in range(num_split):
                self.perm_inds.append([a, b])

        self.num_split = num_split
        self.patch_size = math.floor(image_size[0]/num_split)
        self.image_size = image_size
        self.interpolated_size = (self.patch_size*num_split, self.patch_size*num_split)

    def permute_patch(self, images):
        images = F.interpolate(images, self.interpolated_size)
        batch_image, batch_perms = [], []

        for image in images:
            perm = torch.randperm(self.num_split**2).to(device)
            image_shuffle = torch.zeros(image.size()).type_as(image.data)
            for i_num, i_perm in enumerate(perm):
                x_source, y_source =  self.perm_inds[i_perm]
                x_target, y_target = self.perm_inds[i_num]
                image_shuffle[:, x_target * self.patch_size: (x_target + 1) * self.patch_size, y_target * self.patch_size: (y_target + 1) * self.patch_size] \
                    = image[:, x_source * self.patch_size: (x_source + 1) * self.patch_size,
                                                           y_source * self.patch_size: (y_source + 1) * self.patch_size]
            batch_image.append(image_shuffle)
            batch_perms.append(perm)

        return torch.stack(batch_image), torch.stack(batch_perms)

    def restore_patch(self, images, perms):

        images = F.interpolate(images, self.interpolated_size)
        batch_image, batch_perms = [], []

        for image, perm in zip(images, perms):
            image_restore = torch.zeros(image.size()).type_as(image.data)
            for i_num, i_perm in enumerate(perm):
                x_source, y_source = self.perm_inds[(perm == i_num).nonzero().item()]
                x_target, y_target = self.perm_inds[i_num]
                image_restore[:, x_target * self.patch_size: (x_target + 1) * self.patch_size,
                y_target * self.patch_size: (y_target + 1) * self.patch_size] \
                    = image[:, x_source * self.patch_size: (x_source + 1) * self.patch_size,
                      y_source * self.patch_size: (y_source + 1) * self.patch_size]

            batch_image.append(image_restore)

        return torch.stack(batch_image)

    def perm2vecmat(self, perms):

        batch_size = perms.shape[0]
        y_onehot = torch.zeros(batch_size, self.num_split**2, self.num_split**2).to(device)
        for num, perm in enumerate(perms):
            y_onehot[num][torch.arange(self.num_split**2), perm] = 1.
        return y_onehot.view(batch_size, -1)

    def vecmat2perm(self, y_onehot):
        batch_size = y_onehot.size()[0]
        y_onehot = y_onehot.view(batch_size, self.num_split**2, self.num_split**2)
        _, ind = y_onehot.max(2)
        return ind


    def my_sinkhorn(self, log_alpha, n_iters=20):

        # n = log_alpha.size()[1]
        log_alpha = log_alpha.view(-1, self.num_split ** 2, self.num_split ** 2)
        log_alpha = log_alpha.view(-1, self.num_split**2, self.num_split**2)

        for i in range(n_iters):
            # avoid in-place
            log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=2, keepdim=True)).view(-1, self.num_split**2, 1)
            log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=1, keepdim=True)).view(-1, 1, self.num_split**2)

        return torch.exp(log_alpha).view(-1, (self.num_split**2) * (self.num_split**2))

    def compute_acc(self, p_pred, p_true, average=True):
        n = torch.sum((torch.sum(p_pred == p_true, 1) == self.num_split**2).float())
        patch_level_acc = torch.sum(p_pred == p_true, 1).float()
        if average:
            return n / p_pred.size()[0]*100, patch_level_acc.mean()/(self.num_split**2)*100
        else:
            return n