import torch
import math
import torch.nn.functional as F
import random
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def collate_self(batch):
    batch_mod = {'sketch_img': [], 'sketch_boxes': [],
                 'sketch_label': [], 'sketch_path': []
                 }
    for i_batch in batch:
        batch_mod['sketch_img'].append(i_batch['sketch_img'])
        batch_mod['sketch_boxes'].append(torch.tensor(i_batch['sketch_boxes']).float())
        batch_mod['sketch_label'].append(i_batch['sketch_label'])
        batch_mod['sketch_path'].append(i_batch['sketch_path'])

    batch_mod['sketch_img'] = torch.stack(batch_mod['sketch_img'], dim=0)
    batch_mod['sketch_label'] = torch.tensor(batch_mod['sketch_label'])

    return batch_mod

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