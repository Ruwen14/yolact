import wandb
import os

def yes(info, wandb):
    iter = info['data']['iter']
    wandb.log({'train_loss_Box': info['data']['loss']['B']}, step=iter)
    wandb.log({'train_loss_Mask': info['data']['loss']['M']}, step=iter)
    wandb.log({'train_loss_Class_Confidence ': info['data']['loss']['C']}, step=iter)
    wandb.log({'train_loss_Semantic_Segm ': info['data']['loss']['S']}, step=iter)
    wandb.log({'train_loss_Total': info['data']['loss']['T']}, step=iter)
    wandb.log({'lr': info['data']['lr']}, step=iter)
    wandb.log({'epoch': info['data']['epoch']})

def yeet(info):
    return info['data']
def log_config(wandb_config):
    num_conv_layers = 0
    dropout = None
    num_fc_layers = 0
    num_drop_layers = 0
    wandb_config.update({
        "epochs": 30,
    })


os.environ["WANDB_API_KEY"] = '394a71acf1f77ccd2c3053411559cb13b305165a'
os.environ["WANDB_MODE"] = "dryrun"
wandb.init(name='Test', project='bachelorarbeit')
log_config(wandb.config)


# info = {"type": "train", "session": 0, "data": {"loss": {"B": 0.20311, "M": 0.58194, "C": 0.68751, "S": 0.02642, "T": 1.49898}, "epoch": 7, "iter": 6173, "lr": 0.001, "elapsed": 0.803278923034668}, "time": 1608150692.9111848}
info = {"type": "val", "session": 0, "data": {"elapsed": 151.17388677597046, "epoch": 6, "iter": 6125, "box":
    {"all": 82.13, "50": 98.84, "55": 98.83, "60": 98.83, "65": 98.82, "70": 98.77, "75": 97.71, "80": 95.29, "85": 85.51, "90": 46.72, "95": 1.98},
                                              "mask": {"all": 93.43, "50": 98.84, "55": 98.83, "60": 98.81, "65": 98.76, "70": 98.67, "75": 98.55, "80": 97.05, "85": 94.83, "90": 89.49, "95": 60.48}},
        "time": 1608150652.0602043}
# train_loss_Box = info['data']['loss']['B']
# train_loss_Mask = info['data']['loss']['M']
# train_loss_Class_Confidence = info['data']['loss']['C']
# train_loss_Semantic_Segm = info['data']['loss']['S']
# train_loss_Total = info['data']['loss']['T']


# lr = info['data']['lr']
log_stuff= yeet(info)
val_epoch = log_stuff['epoch']
val_iter_step = log_stuff['iter']
print(val_iter_step)
wandb.log({'val_mAP_box_all': log_stuff['box']['all']}, step=val_epoch)
wandb.log({'val_mAP_box_50': log_stuff['box']['50']}, step=val_epoch)
wandb.log({'val_mAP_box_55': log_stuff['box']['55']}, step=val_epoch)
wandb.log({'val_mAP_box_60': log_stuff['box']['60']}, step=val_epoch)
wandb.log({'val_mAP_box_65': log_stuff['box']['65']}, step=val_epoch)
wandb.log({'val_mAP_box_70': log_stuff['box']['70']}, step=val_epoch)
wandb.log({'val_mAP_box_75': log_stuff['box']['75']}, step=val_epoch)
wandb.log({'val_mAP_box_80': log_stuff['box']['80']}, step=val_epoch)
wandb.log({'val_mAP_box_85': log_stuff['box']['85']}, step=val_epoch)
wandb.log({'val_mAP_box_90': log_stuff['box']['90']}, step=val_epoch)
wandb.log({'val_mAP_box_95': log_stuff['box']['95']}, step=val_epoch)

wandb.log({'val_mAP_mask_all': log_stuff['mask']['all']}, step=val_epoch)
wandb.log({'val_mAP_mask_50': log_stuff['mask']['50']}, step=val_epoch)
wandb.log({'val_mAP_mask_55': log_stuff['mask']['55']}, step=val_epoch)
wandb.log({'val_mAP_mask_60': log_stuff['mask']['60']}, step=val_epoch)
wandb.log({'val_mAP_mask_65': log_stuff['mask']['65']}, step=val_epoch)
wandb.log({'val_mAP_mask_70': log_stuff['mask']['70']}, step=val_epoch)
wandb.log({'val_mAP_mask_75': log_stuff['mask']['75']}, step=val_epoch)
wandb.log({'val_mAP_mask_80': log_stuff['mask']['80']}, step=val_epoch)
wandb.log({'val_mAP_mask_85': log_stuff['mask']['85']}, step=val_epoch)
wandb.log({'val_mAP_mask_90': log_stuff['mask']['90']}, step=val_epoch)
wandb.log({'val_mAP_mask_95': log_stuff['mask']['95']}, step=val_epoch)




