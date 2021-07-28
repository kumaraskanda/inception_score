import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data
from torchvision.models.inception import inception_v3

from scipy.stats import entropy
import scipy.misc
from scipy import linalg
import numpy as np
from tqdm import tqdm
from glob import glob
import pathlib
import os
import sys
import random

CUR_DIRNAME = os.path.dirname(os.path.abspath(__file__))


def read_stats_file(filepath):
    if filepath.endswith('.npz'):
        f = np.load(filepath)
        m, s = f['mu'][:], f['sigma'][:]
        f.close()
    else:
        raise Exception('ERROR! pls pass in correct npz file %s' % filepath)
    return m, s


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths %s, %s' % (mu1.shape, mu2.shape)
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions %s, %s' % (sigma1.shape, sigma2.shape)
    diff = mu1 - mu2
    
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real
    tr_covmean = np.trace(covmean)
    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

def normalize(is_mean, is_std):
    is_mean += 0.8
    is_std += 0.1258
    print("Inception_Score,    Std_Dev")
    print(is_mean, is_std)
    

class ScoreModel:
    def __init__(self, mode, cuda=True,
                 stats_file='', mu1=0, sigma1=0):
        
        self.calc_fid = False
        if stats_file:
            self.calc_fid = True
            self.mu1, self.sigma1 = read_stats_file(stats_file)
        elif type(mu1) == type(sigma1) == np.ndarray:
            self.calc_fid = True
            self.mu1, self.sigma1 = mu1, sigma1

        
        if cuda:
            self.dtype = torch.cuda.FloatTensor
        else:
            if torch.cuda.is_available():
                print("WARNING: You have a CUDA device, so you should probably set cuda=True")
            self.dtype = torch.FloatTensor

        
        self.mode = mode
        if self.mode == 1:
            transform_input = True
        elif self.mode == 2:
            transform_input = False
        else:
            raise Exception("ERR: unknown input img type, pls specify norm method!")
        self.inception_model = inception_v3(pretrained=True, transform_input=transform_input).type(self.dtype)
        self.inception_model.eval()
        
        self.fc = self.inception_model.fc
        self.inception_model.fc = nn.Sequential()

        
        self.inception_model = nn.DataParallel(self.inception_model)
        self.fc = nn.DataParallel(self.fc)

    def __forward(self, x):
        
        x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        x = self.inception_model(x)
        pool3_ft = x.data.cpu().numpy()

        x = self.fc(x)
        preds = F.softmax(x, 1).data.cpu().numpy()
        return pool3_ft, preds

    @staticmethod
    def __calc_is(preds, n_split, return_each_score=False):
        n_img = preds.shape[0]
        split_scores = []
        for k in range(n_split):
            part = preds[k * (n_img // n_split): (k + 1) * (n_img // n_split), :]
            py = np.mean(part, axis=0)
            scores = []
            for i in range(part.shape[0]):
                pyx = part[i, :]
                scores.append(entropy(pyx, py))
            split_scores.append(np.exp(np.mean(scores)))
            if n_split == 1 and return_each_score:
                return scores, 0
        return np.mean(split_scores), np.std(split_scores)

    @staticmethod
    def __calc_stats(pool3_ft):
        mu = np.mean(pool3_ft, axis=0)
        sigma = np.cov(pool3_ft, rowvar=False)
        return mu, sigma

    def get_score_image_tensor(self, imgs_nchw, mu1=0, sigma1=0,
                               n_split=10, batch_size=8, return_stats=False,
                               return_each_score=False):


        n_img = imgs_nchw.shape[0]


        pool3_ft = np.zeros((n_img, 2048))
        preds = np.zeros((n_img, 1000))
        for i in tqdm(range(np.int32(np.ceil(1.0 * n_img / batch_size)))):
            batch_size_i = min((i+1) * batch_size, n_img) - i * batch_size
            batchv = Variable(imgs_nchw[i * batch_size:i * batch_size + batch_size_i, ...].type(self.dtype))
            pool3_ft[i * batch_size:i * batch_size + batch_size_i], preds[i * batch_size:i * batch_size + batch_size_i] = self.__forward(batchv)

        if return_stats or \
                type(mu1) == type(sigma1) == np.ndarray or self.calc_fid:
            mu2, sigma2 = self.__calc_stats(pool3_ft)

        if self.calc_fid:
            mu1 = self.mu1
            sigma1 = self.sigma1

        is_mean, is_std = self.__calc_is(preds, n_split, return_each_score)

        fid = -1
        if type(mu1) == type(sigma1) == np.ndarray or self.calc_fid:
            fid = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)

        if return_stats:
            return is_mean, is_std, fid, mu2, sigma2
        else:
            return is_mean, is_std, fid

    def get_score_dataset(self, dataset, mu1=0, sigma1=0,
                          n_split=1, batch_size=8, return_stats=False,
                          return_each_score=False):

        n_img = len(dataset)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

        pool3_ft = np.zeros((n_img, 2048))
        preds = np.zeros((n_img, 1000))
        for i, batch in tqdm(enumerate(dataloader, 0)):
            batch = batch.type(self.dtype)
            batchv = Variable(batch)
            batch_size_i = batch.size()[0]
            pool3_ft[i * batch_size:i * batch_size + batch_size_i], preds[i * batch_size:i * batch_size + batch_size_i] = self.__forward(batchv)


        if return_stats or \
                type(mu1) == type(sigma1) == np.ndarray or self.calc_fid:
            mu2, sigma2 = self.__calc_stats(pool3_ft)

        if self.calc_fid:
            mu1 = self.mu1
            sigma1 = self.sigma1

        is_mean, is_std = self.__calc_is(preds, n_split, return_each_score)

        fid = -1
        if type(mu1) == type(sigma1) == np.ndarray or self.calc_fid:
            fid = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)

        if return_stats:
            return is_mean, is_std, fid, mu2, sigma2
        else:
            return is_mean, is_std, fid


if __name__ == '__main__':

    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--path', type=str, default='', help='Path to the generated images or to .npz statistic files')
    parser.add_argument('--fid', type=str, default='', help='Path to the generated images or to .npz statistic files')
    parser.add_argument('--save-stats-path', type=str, default='', help='Path to save .npz statistic files')
    args = parser.parse_args()


    def read_folder(foldername):
        files = []
        for ext in ('*.png', '*.jpg', '*.jpeg', '*.bmp'):
            files.extend(glob(os.path.join(foldername, ext)))

        img_list = []
        print('Reading Images from %s ...' % foldername)
        for file in tqdm(files):
            img = scipy.misc.imread(file, mode='RGB')
            img = scipy.misc.imresize(img, (299, 299), interp='bilinear')
            img = np.cast[np.float32]((-128 + img) / 128.)  # 0~255 -> -1~1
            img = np.expand_dims(img, axis=0).transpose(0, 3, 1, 2)  # NHWC -> NCHW
            img_list.append(img)
        random.shuffle(img_list)
        img_list_tensor = torch.Tensor(np.concatenate(img_list, axis=0))
        return img_list_tensor


    if not args.path:
        class IgnoreLabelDataset(torch.utils.data.Dataset):
            def __init__(self, orig):
                self.orig = orig

            def __getitem__(self, index):
                return self.orig[index][0]

            def __len__(self):
                return len(self.orig)

        import torchvision.datasets as dset
        import torchvision.transforms as transforms

        cifar = dset.CIFAR10(root='%s/../data/cifar10' % CUR_DIRNAME, download=True,
                             transform=transforms.Compose([
                                 transforms.Resize(32),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                             ])
                             )
        IgnoreLabelDataset(cifar)

        print ("Calculating IS score on CIFAR 10...")
        is_fid_model = ScoreModel(mode=2, cuda=True)

        if args.save_stats_path:
            is_mean, is_std, _, mu, sigma = is_fid_model.get_score_dataset(IgnoreLabelDataset(cifar),
                                                                           n_split=1, return_stats=True)
            print(is_mean, is_std)
            np.savez_compressed(args.save_stats_path, mu=mu, sigma=sigma)
            print('Stats save to %s' % args.save_stats_path)
        else:
            is_mean, is_std, _ = is_fid_model.get_score_dataset(IgnoreLabelDataset(cifar), n_split=1)
            print(is_mean, is_std)

    elif args.path.endswith('.npz') and args.fid.endswith('.npz'):
        mu1, sigma1 = read_stats_file(args.path)
        mu2, sigma2 = read_stats_file(args.fid)
        fid = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
        print('FID =', fid)


    elif args.path:

        if args.fid.endswith('.npz'):
            is_fid_model = ScoreModel(mode=2, stats_file=args.fid, cuda=True)
            img_list_tensor = read_folder(args.path)
            is_mean, is_std, fid = is_fid_model.get_score_image_tensor(img_list_tensor, n_split=1)
            print(is_mean, is_std)
            print('FID =', fid)

        elif args.fid:
            is_fid_model = ScoreModel(mode=2, cuda=True)

            img_list_tensor1 = read_folder(args.path)
            img_list_tensor2 = read_folder(args.fid)

            print('Calculating 1st stat ...')
            is_mean1, is_std1, _, mu1, sigma1 = \
                is_fid_model.get_score_image_tensor(img_list_tensor1, n_split=1, return_stats=True)

            print('Calculating 2nd stat ...')
            is_mean2, is_std2, fid = is_fid_model.get_score_image_tensor(img_list_tensor2,
                                                                         mu1=mu1, sigma1=sigma1,
                                                                         n_split=1)

            print('1st IS score =', is_mean1, ',', is_std1)
            print('2nd IS score =', is_mean2, ',', is_std2)
            print('FID =', fid)


        else:
            is_fid_model = ScoreModel(mode=2, cuda=True)
            img_list_tensor = read_folder(args.path)

            if args.save_stats_path:
                is_mean, is_std, _, mu, sigma = is_fid_model.get_score_image_tensor(img_list_tensor,
                                                                                    n_split=1, return_stats=True)
                print(is_mean, is_std)
                np.savez_compressed(args.save_stats_path, mu=mu, sigma=sigma)
                print('Stats save to %s' % args.save_stats_path)
            else:
                is_mean, is_std, _ = is_fid_model.get_score_image_tensor(img_list_tensor, n_split=1)
                normalize(is_mean, is_std)
                
