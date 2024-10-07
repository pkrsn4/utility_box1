import matplotlib.pyplot as plt
import random
from tabulate import tabulate


def rand_vis(X,y_true,y_pred,epoch, training_root, phase):

    val_images_dir = f"{training_root}/val_images"
    metrics_dir = f"{training_root}/metrics"
    train_images_dir = f"{training_root}/train_images"


    if y_pred.grad_fn is not None:
        y_pred = y_pred.detach()
    
    sample_vis = random.randint(0, X.shape[0]-1)

    image_vis = X[sample_vis]
    true_mask_vis = y_true[sample_vis]
    pred_mask_vis = y_pred[sample_vis]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))  # 1 row, 3 columns

    axes[0].imshow(image_vis.cpu().numpy().transpose((1, 2, 0)))
    axes[0].axis('off')
    axes[0].set_title('Image')

    axes[1].imshow(true_mask_vis.cpu().numpy())  # Use cmap='gray' for grayscale images
    axes[1].axis('off')
    axes[1].set_title('True Mask')

    axes[2].imshow(pred_mask_vis.cpu().numpy())  # Use cmap='gray' for grayscale images
    axes[2].axis('off')
    axes[2].set_title('Predicted Mask')
    
    plt.tight_layout()  # Adjust spacing between subplots
    #plt.show()

    if phase  == 'validation':
        fig.savefig(f'{val_images_dir}/epoch{epoch}_{sample_vis}.png')
    else:
        fig.savefig(f'{train_images_dir}/epoch{epoch}_{sample_vis}.png')
    plt.close(fig)
    

def save_avg_val_metrics(val_metrics_per_epoch, epoch, training_root):
    
    metrics_dir = f"{training_root}/metrics"
    
    # Calculate rounded metrics
    avg_val_accuracy = round(val_metrics_per_epoch['val_accuracy'].mean() * 100, 4)
    avg_val_dice_score = round(val_metrics_per_epoch['val_dice_score'].mean() * 100, 4)
    avg_val_iou = round(val_metrics_per_epoch['val_iou'].mean() * 100, 4)
    total_val_loss = round(val_metrics_per_epoch['val_loss'].sum(), 4)
    total_val_dice_loss = round(val_metrics_per_epoch['val_dice_loss'].sum(), 4)
    total_val_bce_loss = round(val_metrics_per_epoch['val_bce_loss'].sum(), 4)
    #val_var_loss =  round(val_metrics_per_epoch['val_var_loss'].sum(), 4)

    # Organize data in a list of tuples for tabulate
    table = [
        ("Metric", "Value"),
        ("Dice Score", f"{avg_val_dice_score} %"),
        ("Avg IoU", f"{avg_val_iou} %"),
        ("Accuracy", f"{avg_val_accuracy} %"),
        ("Total Val Loss", f"{total_val_loss}"),
        ("Total BCE Loss",f"{total_val_bce_loss}"),
        ("Total Dice Loss",f"{total_val_dice_loss}"),
        #("Total Var Loss", f"{val_var_loss}"),
    ]

    # Create the table string with a boundary box
    table_str = tabulate(table, headers="firstrow", tablefmt="grid")

    # Render the table string and heading using matplotlib
    fig, ax = plt.subplots(figsize=(5, 3))  # Adjust figure size as needed
    ax.text(0.5, 0.6, table_str, ha='center', va='center', fontsize=12, family='monospace')
    ax.text(0.5, 1.45, f"Validation Metrics Summary, Epoch: {epoch}", ha='center', va='center', fontsize=14, weight='bold')
    ax.axis('off')

    # Save the figure as an image
    #plt.savefig(, bbox_inches='tight', pad_inches=0.1)
    plt.savefig(f"{metrics_dir}/avg_val_metrics_epoch{epoch}.png", bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)