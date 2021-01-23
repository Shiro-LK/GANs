# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 14:51:14 2020

@author: Shiro

    
"""

import torch
from torch import nn
from torch.nn import  functional as F
import numpy as np
from torchvision.models import inception_v3
import scipy.linalg



class FID():
    """
        Frechet Inception distance 
        d(X, Y) = || mu_x - mu_y ||² + Tr( sigma_x + sigma_y -2 sqrt(sigma_x sigma_y) )

    """
    def __init__(self, device):
        self.network = inception_v3(pretrained=True)
        self.network.fc = torch.nn.Identity()
        self.network = self.network.to(device)
        self.device = device
        self.network.eval()
        
        
    def cov(self, x, rowvar=False, bias=False, ddof=None, aweights=None):
        """
            Estimates covariance matrix like numpy.cov
            from @https://github.com/pytorch/pytorch/issues/19037
        """
        # ensure at least 2D
        if x.dim() == 1:
            x = x.view(-1, 1)
    
        # treat each column as a data point, each row as a variable
        if rowvar and x.shape[0] != 1:
            x = x.t()
    
        if ddof is None:
            if bias == 0:
                ddof = 1
            else:
                ddof = 0
    
        w = aweights
        if w is not None:
            if not torch.is_tensor(w):
                w = torch.tensor(w, dtype=torch.float)
            w_sum = torch.sum(w)
            avg = torch.sum(x * (w/w_sum)[:,None], 0)
        else:
            avg = torch.mean(x, 0)
    
        # Determine the normalization
        if w is None:
            fact = x.shape[0] - ddof
        elif ddof == 0:
            fact = w_sum
        elif aweights is None:
            fact = w_sum - ddof
        else:
            fact = w_sum - ddof * torch.sum(w * w) / w_sum
    
        xm = x.sub(avg.expand_as(x))
    
        if w is None:
            X_T = xm.t()
        else:
            X_T = torch.mm(torch.diag(w), xm).t()
    
        c = torch.mm(X_T, xm)
        c = c / fact
    
        return c.squeeze() 
    
    def preprocess(self, img):
        """

        Parameters
        ----------
        img : Tensor
            images.

        Returns
        -------
        img : Tensor
            Resized images for inception network (299,299)

        """
        img = torch.nn.functional.interpolate(img, size=(299, 299), mode='bilinear', align_corners=False)
        return img
    
    def extract_features(self, reals, fakes):
        """
            extract features 

        """
        reals = self.preprocess(reals)
        fakes = self.preprocess(fakes)
        
        reals_features = self.network(reals)
        fakes_features = self.network(fakes)
        return reals_features, fakes_features
    def matrix_sqrt(self, x):
        '''
        Function that takes in a matrix and returns the square root of that matrix.
        For an input matrix A, the output matrix B would be such that B @ B is the matrix A.
        Parameters:
            x: a matrix
            
        function which comes from Coursera
        '''
        y = x.cpu().detach().numpy()
        y = scipy.linalg.sqrtm(y)
        return torch.Tensor(y.real, device=x.device)
    
    def __call__(self, reals_features, fakes_features):
        """
            Compute the fid between the distribution of the fakes and reals images
        """

        mu_x = reals_features.mean(0)
        mu_y = fakes_features.mean(0)
        
        sigma_x = self.cov(reals_features)
        sigma_y = self.cov(fakes_features)
        
        A = torch.norm(mu_x - mu_y, 2) ** 2
        B = torch.trace(sigma_x) + torch.trace(sigma_y)
        C = -2 * torch.trace(self.matrix_sqrt( sigma_x.matmul(sigma_y) ) )

        return A + B + C
    
    
if __name__ == "__main__":
    fid = FID()
    X1 = torch.as_tensor(np.random.uniform(0,1, (4, 3, 224,224 ))).float()
    X2 = torch.as_tensor(np.random.uniform(0,1, (4, 3, 224,224 ))).float()
    
    X1_feats, X2_feats = fid.extract_features(X1, X2)
    print(fid(X1_feats, X2_feats))