import torch.nn as nn
import torch.nn.functional as F
import torch


def gaussian_kernel(x, y, sigma=2.0):
    dist = torch.sum((x.unsqueeze(1) - y.unsqueeze(0)) ** 2, dim=2)
    return torch.exp(-dist / (2 * sigma ** 2))# .type(torch.float64)

def mmd(x, y, kernel=gaussian_kernel):
    _, n, _ = x.size()
    _, m, _ = y.size()

    xx = kernel(x.unsqueeze(2).expand(-1, -1, m, -1), x.unsqueeze(1).expand(-1, m, -1, -1)).sum(dim=3)
    yy = kernel(y.unsqueeze(1).expand(-1, n, -1, -1), y.unsqueeze(2).expand(-1, -1, n, -1)).sum(dim=3)
    xy = kernel(x.unsqueeze(1).expand(-1, m, -1, -1), y.unsqueeze(2).expand(-1, -1, n, -1)).sum(dim=3)

    torch.diagonal(xx, dim1=1, dim2=2).fill_(0)
    torch.diagonal(yy, dim1=1, dim2=2).fill_(0)

    factor_n = 1 if n == 1 else 1. / (n * (n - 1))
    factor_m = 1 if m == 1 else 1. / (m * (m - 1))


    mmd = torch.sum(xx, dim=(1, 2)) * factor_n + torch.sum(yy, dim=(1, 2)) * factor_m - 2 * torch.sum(xy, dim=(1, 2)) / (n * m)

    return mmd.abs()


def distance(x_list, weight, func=mmd,):
    wx_list = [x * w for x, w in zip(x_list, weight)]
    x_stack = torch.stack(x_list, dim=0)
    wx_stack = torch.stack(wx_list, dim=0)
    mean = torch.mean(wx_stack, dim=0)#取平均

    if func == F.kl_div:
        log_softmax_x = F.log_softmax(x_stack, dim=1)
        softmax_mean = F.softmax(mean, dim=0)
        distances = F.kl_div(log_softmax_x, softmax_mean.unsqueeze(0).expand_as(log_softmax_x), reduction='none').mean(dim=1)
    else:
        # distances = [func(F.softmax(x, dim=0), mean) for x in x_list]
        distances = func(x_stack, mean.unsqueeze(0).expand_as(x_stack))

    return distances.mean()

class UnionLoss(nn.Module):
    def __init__(self, pos_lambda=0.1, neg_lambda=0.01, func='MMD', start_epoch=100):
        super(UnionLoss, self).__init__()
        self.pos_lambda = pos_lambda if func != 'WD' else pos_lambda/100
        self.neg_lambda = neg_lambda if func != 'WD' else neg_lambda/100
        self.func = mmd
        self.start_epoch = start_epoch

    def forward(self, inputs, targets, x_list, weight, epoch):
        BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='mean')

        pos_indices = torch.nonzero(targets).squeeze()
        neg_indices = torch.nonzero(targets == 0).squeeze()
        pos_list, neg_list = [], []
        for i in range(len(x_list)):
            pos_list.append(torch.index_select(x_list[i], dim=0, index=pos_indices))
            neg_list.append(torch.index_select(x_list[i], dim=0, index=neg_indices))

        if epoch >= self.start_epoch:
            pos_loss = 0 if pos_indices.numel() == 0 else distance(pos_list, weight, func=self.func,)
            neg_loss = 0 if neg_indices.numel() == 0 else distance(neg_list, weight, func=self.func,)
        else: pos_loss, neg_loss = 0., 0.
        Union_loss = BCE_loss + self.pos_lambda * pos_loss - self.neg_lambda * neg_loss

        return Union_loss.sum()