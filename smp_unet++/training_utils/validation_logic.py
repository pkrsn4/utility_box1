from tqdm.auto import tqdm
import yaml

import torch
import torch.nn.functional as F
from torch import Tensor

from metrics import *
from vis_utils import rand_vis


def validate(model,device, valid_loader ,epoch, train_config):
    
    training_root = f"{train_config['training_root']}/{train_config['run_name']}"
    amp = train_config['amp']
    
    batch_idx = 0
    val_metrics_list = []
    
    with torch.inference_mode():
        with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
            for batch in tqdm(valid_loader, desc ='Validation'):
                X, true = batch[0], batch[1]
                X = X.to(device=device, dtype=torch.float32)
                true = true.to(device).float() 
                
                pred = model(X)
                pred = pred.squeeze(1)
                sigmoid_pred = F.sigmoid(pred).float() 
                thresh_pred  = (sigmoid_pred>0.5).float()
                
                val_bce_loss =  xentropy_loss(pred, true)
                val_dice_loss = dice_loss(thresh_pred, true)
                #val_var_loss  =  var_loss(sigmoid_pred, true)
                
                val_loss = val_bce_loss+val_dice_loss#+val_var_loss
                
                val_acc  = binary_accuracy(thresh_pred, true)
                val_dice = dice_coeff(thresh_pred, true, reduce_batch_first = True)
                val_iou  = iou(thresh_pred,true)
                

                rand_vis(X,true, pred, epoch, training_root, phase = 'validation')
                
                val_metrics_list.append({'epoch': epoch,
                                         'batch_idx':batch_idx,
                                         'val_accuracy': val_acc,
                                         'val_dice_score': val_dice.item(),
                                         'val_iou': val_iou.item(),
                                         'val_dice_loss': val_dice_loss.item(),
                                         'val_bce_loss': val_bce_loss.item(),
                                         #'val_var_loss':val_var_loss.item(),
                                         'val_loss': val_loss.item(),
                                       })
                batch_idx += 1
    
    return(val_metrics_list)