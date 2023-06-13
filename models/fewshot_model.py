import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from models.encoder_sort import Res101Encoder


class FewShotSeg(nn.Module):

    def __init__(self, pretrained_weights="deeplabv3"):
        super().__init__()

        # Encoder
        self.encoder = Res101Encoder(replace_stride_with_dilation=[True, True, False],
                                     pretrained_weights=pretrained_weights)  # or "resnet101"
        self.device = torch.device('cuda')
        self.scaler = 20.0
        self.criterion = nn.NLLLoss()
        self.alpha = torch.Tensor([1.0, 0.])


    def forward(self, supp_imgs, supp_mask, qry_imgs, train=False, t_loss_scaler=1, n_iters=20):
        """
        Args:
            supp_imgs: support images
                way x shot x [B x 3 x H x W], list of lists of tensors
            fore_mask: foreground masks for support images
                way x shot x [B x H x W], list of lists of tensors
            back_mask: background masks for support images
                way x shot x [B x H x W], list of lists of tensors
            qry_imgs: query images
                N x [B x 3 x H x W], list of tensors
        """

        self.n_ways = len(supp_imgs)  # 1
        self.n_shots = len(supp_imgs[0])  # 1
        self.n_queries = len(qry_imgs)  # 1
        assert self.n_ways == 1  # for now only one-way, because not every shot has multiple sub-images
        assert self.n_queries == 1

        qry_bs = qry_imgs[0].shape[0]    # 1
        supp_bs = supp_imgs[0][0].shape[0]      # 1
        img_size = supp_imgs[0][0].shape[-2:]
        supp_mask = torch.stack([torch.stack(way, dim=0) for way in supp_mask],
                                dim=0).view(supp_bs, self.n_ways, self.n_shots, *img_size)  # B x Wa x Sh x H x W

        ###### Extract features ######
        imgs_concat = torch.cat([torch.cat(way, dim=0) for way in supp_imgs]
                                + [torch.cat(qry_imgs, dim=0), ], dim=0)     
        # encoder output
        img_fts, tao, t1 = self.encoder(imgs_concat)  
        supp_fts = [img_fts[dic][:self.n_ways * self.n_shots * supp_bs].view(  # B x Wa x Sh x C x H' x W'
            supp_bs, self.n_ways, self.n_shots, -1, *img_fts[dic].shape[-2:]) for _, dic in enumerate(img_fts)]  

        qry_fts = [img_fts[dic][self.n_ways * self.n_shots * supp_bs:].view(  # B x N x C x H' x W'
            qry_bs, self.n_queries, -1, *img_fts[dic].shape[-2:]) for _, dic in enumerate(img_fts)]

        ##### Get threshold #######
        self.t = tao[self.n_ways * self.n_shots * supp_bs:]  # t for query features   
        self.thresh_pred = [self.t for _ in range(self.n_ways)]  
        ##### Get rate #######
        self.t1 = t1[self.n_ways * self.n_shots * supp_bs:]  # t for query features  
        self.t1 = torch.sigmoid(self.t1) 


        ###### Compute loss ######
        align_loss = torch.zeros(1).to(self.device)
        mse_loss = torch.zeros(1).to(self.device)
        outputs = []
        for epi in range(supp_bs):
            ###### Extract prototypes ######
            if supp_mask[epi][0].sum() == 0:
                supp_fts_ = [[[self.getFeatures(supp_fts[n][[epi], way, shot], supp_mask[[epi], way, shot])  
                               for shot in range(self.n_shots)] for way in range(self.n_ways)] for n in
                             range(len(supp_fts))]   
                fg_prototypes = [self.getPrototype(supp_fts_[n]) for n in range(len(supp_fts))] 

                ###### Get query predictions ######
                qry_pred = [torch.stack(   # (1, 512, 64, 64) (1, 512) (1, 1)
                    [self.getPred(qry_fts[n][epi], fg_prototypes[n][way], self.thresh_pred[way])       
                     for way in range(self.n_ways)], dim=1) for n in range(len(qry_fts))]  
                
                qry_pred_up = [F.interpolate(qry_pred[n], size=img_size, mode='bilinear', align_corners=True)
                               for n in range(len(qry_fts))]   
                ###### Combine predictions of different feature maps ######
                pred = [self.alpha[n] * qry_pred_up[n] for n in range(len(qry_fts))]
                preds = torch.sum(torch.stack(pred, dim=0), dim=0) / torch.sum(self.alpha)   
                preds = torch.cat((1.0 - preds, preds), dim=1)  
                outputs.append(preds)

            else:
                features_supp = [
                    [[self.get_features(supp_fts[n][[epi], way, shot], supp_mask[[epi], way, shot])
                      for shot in range(self.n_shots)] for way in range(self.n_ways)] for n in
                    range(len(supp_fts))]  
                prototypes = [self.get_all_prototypes(features_supp[n]) for n in range(len(features_supp))] 

                fg_prototypes = [[self.get_mean_prototype(prototypes[n][way]) for way in range(self.n_ways)]
                                 for n in range(len(qry_fts))]    
                # sort and select
                select_prototypes = [[self.sort_select_fts(prototypes[n][way], fg_prototypes[n][way], self.t1)
                                  for way in range(self.n_ways)] for n in range(len(qry_fts))]   
                select_prototypes = [[self.get_mean_prototype(select_prototypes[n][way]) for way in range(self.n_ways)]
                                 for n in range(len(qry_fts))] 
                
                ###### Get query predictions ######
                qry_pred = [torch.stack(  # (1, 512, 64, 64) (1, 512) (1, 1)
                    [self.getPred(qry_fts[n][epi], select_prototypes[n][way], self.thresh_pred[way])  
                     for way in range(self.n_ways)], dim=1) for n in range(len(qry_fts))]  
                
                qry_pred_up = [F.interpolate(qry_pred[n], size=img_size, mode='bilinear', align_corners=True)
                               for n in range(len(qry_fts))]  
                ###### Combine predictions of different feature maps ######
                pred = [self.alpha[n] * qry_pred_up[n] for n in range(len(qry_fts))]
                preds = torch.sum(torch.stack(pred, dim=0), dim=0) / torch.sum(self.alpha)  
                preds = torch.cat((1.0 - preds, preds), dim=1)  
                outputs.append(preds)


       
        output = torch.stack(outputs, dim=1)  
        output = output.view(-1, *output.shape[2:])    

        return output


    def getPred(self, fts, prototype, thresh):
        """
        Calculate the distance between features and prototypes
        Args:
            fts: input features
                expect shape: N x C x H x W
            prototype: prototype of one semantic class
                expect shape: 1 x C
        """

        sim = -F.cosine_similarity(fts, prototype[..., None, None], dim=1) * self.scaler
        pred = 1.0 - torch.sigmoid(0.5 * (sim - thresh))   # ([1, 64, 64])

        return pred

    def getFeatures(self, fts, mask):
        """
        Extract foreground and background features via masked average pooling

        Args:
            fts: input features, expect shape: 1 x C x H' x W'
            mask: binary mask, expect shape: 1 x H x W
        """

        fts = F.interpolate(fts, size=mask.shape[-2:], mode='bilinear', align_corners=True)  

        # masked fg features
        masked_fts = torch.sum(fts * mask[None, ...], dim=(-2, -1)) \
                     / (mask[None, ...].sum(dim=(-2, -1)) + 1e-5)  

        return masked_fts

    def getPrototype(self, fg_fts):
        """
        Average the features to obtain the prototype

        Args:
            fg_fts: lists of list of foreground features for each way/shot
                expect shape: Wa x Sh x [1 x C] (1, 1, (1, 512))
            bg_fts: lists of list of background features for each way/shot
                expect shape: Wa x Sh x [1 x C]
        """
        n_ways, n_shots = len(fg_fts), len(fg_fts[0])
        fg_prototypes = [torch.sum(torch.cat([tr for tr in way], dim=0), dim=0, keepdim=True) / n_shots for way in
                         fg_fts]  

        return fg_prototypes

    def get_features(self, features, mask):
   
        features_trans = F.interpolate(features, size=mask.shape[-2:], mode='bilinear',
                                       align_corners=True)  
        features_trans = features_trans.squeeze(0)
        features_trans = features_trans.permute(1, 2, 0)
        features_trans = features_trans.view(features_trans.shape[-2] * features_trans.shape[-3],
                                             features_trans.shape[-1])
        mask = mask.squeeze(0).view(-1)
        indx = mask == 1
        features_trans = features_trans[indx]    

        return features_trans

    def get_all_prototypes(self, fg_fts):

        n_ways, n_shots = len(fg_fts), len(fg_fts[0])
        prototypes = [sum([shot for shot in way]) / n_shots for way in fg_fts]
        return prototypes

    def get_mean_prototype(self, prototypes):
       
        return torch.mean(prototypes, dim=0).unsqueeze(0)
   

    def sort_select_fts(self, fts, prototype, rate):

        a = np.array([0.9])
        a_tensor = torch.from_numpy(a).unsqueeze(0).cuda()
        rate = a_tensor
        sim = F.cosine_similarity(fts, prototype, dim=1)
        index = sim >= rate.squeeze(0)
        if index.sum() != 0:
            fts = fts[index]

        return fts







