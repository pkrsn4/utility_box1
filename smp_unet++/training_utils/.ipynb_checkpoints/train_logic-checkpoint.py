from tqdm.auto import tqdm
import pandas as pd

import torch
import torch.nn.functional as F

from metrics import *
from vis_utils import rand_vis


def train(model, 
          optimizer,
          grad_scaler,
          learning_rate,
          device, 
          train_loader,
          epoch,
          train_config
         ):
    
    training_root = f"{train_config['training_root']}/{train_config['run_name']}"
    amp = train_config['amp']
    gradient_clipping = train_config['gradient_clipping']
    metrics_dir = f"{training_root}/metrics"
    
    batch_idx = 1
    train_metrics_list = []
    
    for batch in tqdm(train_loader, total= len(train_loader), desc = f"Training Epoch: {epoch}"):
        X, true = batch[0], batch[1]
        
        X = X.to(device=device, dtype=torch.float32)
        true = true.to(device).float() 
    
        with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
            pred = model(X)
            pred = pred.squeeze(1)
            sigmoid_pred = F.sigmoid(pred)
            thresh_pred  = (sigmoid_pred>0.5).float() 
            
            #Loss Computations
            train_bce_loss  =  xentropy_loss(pred, true) #BCE With Logit
            train_dice_loss =  dice_loss(thresh_pred, true)
            #train_var_loss  =  var_loss(sigmoid_pred, true)
            
            train_loss = train_bce_loss+train_dice_loss#+train_var_loss
            
        train_acc  = binary_accuracy(thresh_pred, true)
        train_dice = dice_coeff(thresh_pred, true, reduce_batch_first = True)
        train_iou  = iou(thresh_pred, true)
            
        optimizer.zero_grad(set_to_none=True)
        grad_scaler.scale(train_loss).backward()
        grad_scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
        grad_scaler.step(optimizer)
        grad_scaler.update()

        rand_vis(X,true,pred,epoch, training_root, phase = 'train')
        
        train_metrics_list.append({'epoch': epoch,
                                   'batch_idx':batch_idx,
                                   'train_accuracy':train_acc,
                                   'train_dice_score': train_dice.item(),
                                   'train_iou': train_iou.item(),
                                   'train_bce_loss': train_bce_loss.item(),
                                   'train_dice_loss': train_dice_loss.item(),
                                   #'train_var_loss': train_var_loss.item(),
                                   'train_loss': train_loss.item()
                                  })
        pd.DataFrame(train_metrics_list).to_csv(f'{metrics_dir}/train_metrics_per_batch_for_epoch{epoch}.csv', index = False)
        
        batch_idx += 1
    
    return(model, optimizer, grad_scaler, learning_rate, train_metrics_list)