import numpy as np
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from sim_suction_model.utils.pointnet2_model import Pointnet2_scorenet


class ScoreNet(nn.Module):
    def __init__(self, training=True):
        super(ScoreNet, self).__init__()
        self.is_training = training
        self.extrat_featurePN2 = Pointnet2_scorenet(input_chann=6, k_score=1)
        self.criterion_cls = nn.NLLLoss(reduction='mean')
        self.criterion_reg = nn.MSELoss(reduction='mean')
        self.criterion_cross=nn.CrossEntropyLoss()
        self.criterion_cls_multi = torch.nn.BCEWithLogitsLoss(reduction='mean')
        self.criterion_smoothL1 = nn.SmoothL1Loss(reduction='mean')
        
    def compute_loss(self, pscore, tscore):
        
        '''
          Input:  
            pscore_location : [B,N] 
            tscore_location : [B,N]
        '''


        loss_point_score = self.criterion_reg(pscore, tscore.float())
       
        return loss_point_score

    def forward(self, pc, pc_score):
        '''
         Input:
          pc              :[B,A,24]
          pc_score        :[B,A]
          pc_label        :[B,A]
         Output:
          all_feature     :[B,A,Feature(128)])
          output_score    :[B,A]
          loss
        '''
        B, N, _ = pc.shape
    
        all_feature, output_score = self.extrat_featurePN2(pc[:,:,:6].permute(0,2,1)) 
        all_feature = all_feature.transpose(2,1) 
        loss = None
        if self.is_training and pc_score is not None:#
            loss = self.compute_loss(output_score, pc_score)
            return all_feature, output_score, loss

        else:
            
            # Step 1: Noise Removal using Percentile
            scores=output_score[:, :, 0:1]
            scores=scores.to('cpu').numpy().squeeze()
            threshold = 0.2
            filtered_indices = [i for i, score in enumerate(scores) if score >= threshold]
            filtered_scores = [scores[i] for i in filtered_indices]
            
            # Step 2: Sorting the Scores
            sorted_order = np.argsort(filtered_scores)[::-1]
            sorted_scores = [filtered_scores[i] for i in sorted_order]
            sorted_indices = [filtered_indices[i] for i in sorted_order]
            
            if sorted_scores==[]:
                return None, None

            # Get top 1 score
            top_1_score = [sorted_scores[0]]
            top_1_index = [sorted_indices[0]]
            
            # Step 3: Selecting the Top Percentages
            results = {'top_1': (top_1_score, top_1_index)}
            percentages=[1, 5, 10]
            for percentage in percentages:
                num_top = int(percentage / 100 * len(sorted_scores))
                top_scores = sorted_scores[:num_top]
                top_indices = sorted_indices[:num_top]
                results[f"top_{percentage}%"] = (top_scores, top_indices)

            return results, scores
     

if __name__ == '__main__':
    pass
