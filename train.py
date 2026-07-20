import random
import os
from options.options import Options
from util import *
from util import get_bal_sampler
import torch
import numpy as np
from tensorboardX import SummaryWriter
from models.trainer import SAP_Trainer
def seed_torch(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    # Options
    options = Options()
    opt = options.parse()
    seed_torch(opt.seed)
    log_path = os.path.join('./check_points', opt.experiment_name)
    os.makedirs(log_path, exist_ok=True)
    sap_log_dir = os.path.join(log_path, 'train_sap')
    model_save_dir = os.path.join(sap_log_dir, 'model')
    os.makedirs(model_save_dir, exist_ok=True)

    train_writer = SummaryWriter(sap_log_dir)
    from models.network.network import SAP_Net
    Logger(os.path.join(sap_log_dir, 'train_sap.log'))
    options.print_options()
    print(f"SAP log dir: {sap_log_dir}")
    print(f"SAP model save dir: {model_save_dir}")
    # Data Load
    train_dataset = get_dataset_train(opt.train_data_root, opt.train_classes)
    print(f"Train dataset size: {len(train_dataset)}")
    sampler = get_bal_sampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=opt.batch_size,
                                                shuffle=False,
                                                sampler=sampler,
                                                drop_last=True,
                                                num_workers=opt.num_workers) 
    val_dataset = get_dataset_test(opt.val_data_root, opt.val_classes)
    print(f"Val dataset size: {len(val_dataset)}")
    sampler = get_bal_sampler(val_dataset)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                                batch_size=opt.batch_size,
                                                shuffle=False,
                                                sampler=sampler,
                                                drop_last=True,
                                                num_workers=opt.num_workers)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training device: {device}")
    print(f"Train steps per epoch: {len(train_loader)}")
    print(f"Val steps per epoch: {len(val_loader)}")

    model = SAP_Net(r=opt.rank,alpha=opt.alpha,target_layer_idx=opt.target_layer_idx)
    model.clip_vision.print_trainable_parameters()
    if device == "cuda" and torch.cuda.device_count() > 1:
        print(f"Using DataParallel with {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)
    trainer = SAP_Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=opt.epochs,
        save_dir=model_save_dir,
        writer=train_writer,
        log_interval=20,
    )

    trainer.fit()
    train_writer.close()



