import torch
import torch.nn as nn

def dice_coeff(pred: torch.Tensor, true: torch.Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert pred.size() == true.size()
    assert pred.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if pred.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (pred * true).sum(dim=sum_dim)
    sets_sum = pred.sum(dim=sum_dim) + true.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()

def dice_loss(pred: torch.Tensor, true: torch.Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(pred, true, reduce_batch_first=True)

def var_loss(pred: torch.Tensor, true: torch.Tensor):
    """`pred` and `true` must be of torch.float32. Assuming of shape NxHxWxC."""
    
    target_probs = pred * true
    background_probs = pred * (1 - true)
    
    # Filter out zeros
    non_zero_background = background_probs[background_probs != 0]
    non_zero_target = target_probs[target_probs != 0]

    # Compute variance if there are enough non-zero elements
    if non_zero_background.numel() > 1:
        var_background = torch.var(non_zero_background)
    else:
        var_background = torch.tensor(0.0, dtype=pred.dtype, device=pred.device)

    if non_zero_target.numel() > 1:
        var_target = torch.var(non_zero_target)
    else:
        var_target = torch.tensor(0.0, dtype=pred.dtype, device=pred.device)

    # Compute the loss
    loss = (0.2)*var_background + (0.8)*var_target

    return loss


xentropy_loss = nn.BCEWithLogitsLoss()

def iou(pred: torch.Tensor, true: torch.Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of IOU coefficient for all batches, or for a single mask
    assert pred.size() == true.size()
    assert pred.dim() == 3 or not reduce_batch_first
    
    sum_dim = (-1, -2) if pred.dim() == 2 or not reduce_batch_first else (-1, -2, -3)
    
    intersection = (pred * true).sum(dim=sum_dim)
    union = pred.sum(dim=sum_dim) + true.sum(dim=sum_dim) - intersection
    iou = (intersection + epsilon) / (union + epsilon)
    
    return iou.mean()

def binary_accuracy(pred: torch.Tensor, true: torch.Tensor):
    """
    Returns the Average Accuracy 
    """
    assert pred.size() == true.size()
    
    correct_predictions = (pred.data == true.data)
    
    # Compute the number of correct predictions
    num_correct = correct_predictions.sum().detach().cpu().numpy()
    
    # Calculate accuracy
    accuracy = num_correct / (pred.size(0) * pred.size(-1) * pred.size(-2))
    
    return accuracy