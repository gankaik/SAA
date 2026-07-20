import os
import torch
import numpy as np
import random

from tqdm import tqdm

from options.options import Options
from util import Logger, read_yaml
from util import  get_dataset_test
from util import get_bal_sampler
from torch.utils.data.sampler import WeightedRandomSampler
from sklearn.metrics import accuracy_score, average_precision_score, roc_curve
from torch.cuda.amp import GradScaler
from models.network.network import SAP_Net

def load_checkpoint_compat(model, checkpoint_path):
    """Load checkpoint with a strict-first, compatibility fallback strategy."""
    model_load = torch.load(checkpoint_path, map_location='cpu')
    state_dict = model_load.get('model_state_dict', model_load)

    model_keys = list(model.state_dict().keys())
    has_module_prefix = any(k.startswith('module.') for k in state_dict.keys())
    model_expects_module = any(k.startswith('module.') for k in model_keys)
    if has_module_prefix and not model_expects_module:
        state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
    elif (not has_module_prefix) and model_expects_module:
        state_dict = {f"module.{k}": v for k, v in state_dict.items()}

    try:
        model.load_state_dict(state_dict)
        print(f"Checkpoint loaded with strict=True: {checkpoint_path}")
    except RuntimeError as err:
        print("Strict checkpoint loading failed, fallback to strict=False.")
        print(f"Reason: {err}")
        incompat = model.load_state_dict(state_dict, strict=False)
        print(
            f"Fallback loaded. missing_keys={len(incompat.missing_keys)}, "
            f"unexpected_keys={len(incompat.unexpected_keys)}"
        )
        if incompat.missing_keys:
            print("Sample missing keys:", incompat.missing_keys[:8])
        if incompat.unexpected_keys:
            print("Sample unexpected keys:", incompat.unexpected_keys[:8])

vals = ['progan', 'stylegan', 'biggan', 'cyclegan', 'stargan', 'gaugan', 'deepfake', 'seeingdark', 'san', 'crn', 'imle', 'guided', 'ldm_200', 'ldm_200_cfg', 'ldm_100', 'glide_100_27', 'glide_50_27', 'glide_100_10', 'dalle']
multiclass = [1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
vals_GenImage = ['ADM', 'BigGAN', 'glide', 'Midjourney', 'stable_diffusion_v_1_4', 'stable_diffusion_v_1_5', 'VQDM', 'wukong']
multiclass_GenImage = [0, 0, 0, 0, 0, 0, 0, 0]
vals_AIGI=['BlendFace',"BLIP","CommunityAI","DALLE-3","E4S","FaceSwap","FLUX1-dev","GLIDE","Imagen3","Infinite_ID","InstantID","InSwap","IP_Adapter","Midjourney","PhotoMaker","ProGAN","R3GAN","SD3","SDXL","SimSwap","SocialRF","StyleGAN-XL","StyleGAN3","StyleSwim","WFIR"]
multiclass_AIGI=[0]*len(vals_AIGI)
vals_Comm=['akool',"Doubao_videos","google-imagen","hunyuan","kling","midjourney","sora","veo","chatgpt-image","flux","hailuo","ideogram","lexica","nano_banana","stable-diffusion"]
multiclass_Comm=[0]*len(vals_Comm)
def seed_torch(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True



if __name__ == '__main__':
    seed_torch(3407)

    # Options
    options = Options()
    opt = options.parse()
    log_dir = os.path.join('./check_points', opt.experiment_name)
    os.makedirs(log_dir, exist_ok=True)
    Logger(os.path.join(log_dir, 'evaluation_log.log'))

    classes = ''
    model=SAP_Net(r=opt.rank)
    load_checkpoint_compat(model, opt.weights)
    model.cuda()
    model.eval()
    print(f"LOAD {opt.weights}!!!!!!")

    scaler = GradScaler()

    accs = []
    aps = []
    # test
    if 'GenImage' in opt.eval_data_root:
        vals = vals_GenImage
        multiclass = multiclass_GenImage
    elif 'AIGI' in opt.eval_data_root:
        vals = vals_AIGI
        multiclass = multiclass_AIGI
    elif 'Comm' in opt.eval_data_root:
        vals = vals_Comm
        multiclass = multiclass_Comm
    for val_id, val in enumerate(vals):
        sub_test_data_root = '{}/{}'.format(opt.eval_data_root, val)
        if multiclass[val_id] == 1:
            classes = os.listdir(sub_test_data_root)
        else:
            classes = ['']

        val_dataset = get_dataset_test(sub_test_data_root, classes)
        sampler = get_bal_sampler(val_dataset)
        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=opt.batch_size,
                                                 shuffle=False,
                                                 drop_last=True,
                                                 sampler=sampler,
                                                 num_workers=opt.num_workers)

        val_accs = []
        val_aps = []
        val_targets = []
        val_scores = []

        for data, target in tqdm(val_loader):
            data, target = data.cuda(), target.cuda()
            with torch.amp.autocast('cuda'):

                with torch.no_grad():

                    batch = model(data)
                    if isinstance(batch, (tuple, list)):
                        batch = batch[0]
                    pre_prob = torch.sigmoid(batch).cpu().view(-1)
                    target = target.cpu().view(-1)
                    acc = accuracy_score(target.numpy(), (pre_prob.numpy() > 0.5).astype(np.int64))
                    ap = average_precision_score(target.numpy(), pre_prob.numpy())

                    val_accs.append(acc)
                    val_aps.append(ap)

        val_mean_acc = np.mean(val_accs)
        val_mean_ap = np.mean(val_aps)
        print(
            "({} {:10}) acc: {:.2f}; ap: {:.2f}".format(val_id + 1, val, val_mean_acc * 100, val_mean_ap * 100))
        accs.append(val_mean_acc)
        aps.append(val_mean_ap)

    mean_acc = np.mean(accs) * 100
    mean_ap = np.mean(aps) * 100
    print("({} {:10}) acc: {:.2f}; ap: {:.2f}".format('*', 'Mean', mean_acc, mean_ap))
    print('*'*60)
