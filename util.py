import sys
import yaml
from torch.utils.data import ConcatDataset, WeightedRandomSampler
import numpy as np
from collections import OrderedDict
import os
import random
import torch
import types

# Compatibility shim for older torchvision versions that still reference torch._six.
if not hasattr(torch, "_six"):
    torch._six = types.SimpleNamespace(string_classes=(str, bytes))

from torchvision import transforms, datasets
from PIL import Image
import cv2
import math
from scipy.ndimage.filters import gaussian_filter
from io import BytesIO
sys.path.append(os.path.dirname(__file__))
try:
    from options.options import Options
except ModuleNotFoundError:
    # Try to ensure this package's directory is on sys.path and retry
    this_dir = os.path.dirname(os.path.abspath(__file__))
    if this_dir not in sys.path:
        sys.path.insert(0, this_dir)
    try:
        from options.options import Options
    except Exception:
        # Fallback: load options/options.py directly by filename
        options_path = os.path.join(this_dir, 'options', 'options.py')
        if os.path.exists(options_path):
            import importlib.util
            spec = importlib.util.spec_from_file_location('options.options', options_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            Options = getattr(module, 'Options')
        else:
            raise

opt=Options().parse()
class Logger(object):
    def __init__(self, outfile):
        self.terminal = sys.__stdout__
        self.log = open(outfile, "a")
        sys.stdout = self

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()
        sys.stdout = self.terminal


def read_yaml(path):
    try:
        with open(path, 'r', encoding='utf-8') as file:
            yaml_data = yaml.safe_load(file)
        return yaml_data
    except FileNotFoundError:
        print(f"File {path} not found.")
    except yaml.YAMLError as exc:
        print(f"Error in YAML file {path}: {exc}")
    except Exception as exc:
        print(f"An error occurred while reading the YAML file {path}: {exc}")


def print_yaml_params(data):
    for key, value in data.items():
        print(str(key) + ' : ' + str(value))

def translate_duplicate(img, cropSize=224):
    if min(img.size) < cropSize:
        width, height = img.size

        new_width = width * math.ceil(cropSize / width)
        new_height = height * math.ceil(cropSize / height)

        new_img = Image.new('RGB', (new_width, new_height))
        for i in range(0, new_width, width):
            for j in range(0, new_height, height):
                new_img.paste(img, (i, j))
        return new_img
    else:
        return img

def random_rotate(img):
    angle = random.choice([0, 90, 180, 270])
    return img.rotate(angle)
def data_augment(img, opt):
    img = np.array(img)
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
        img = np.repeat(img, 3, axis=2)

    if random.random() < opt.blur_prob:
        sig = sample_continuous(opt.blur_sig)
        gaussian_blur(img, sig)

    if random.random() < opt.jpg_prob:
        method = sample_discrete(opt.jpg_method)
        qual = sample_discrete(opt.jpg_qual)
        img = jpeg_from_key(img, qual, method)

    return Image.fromarray(img)



def sample_continuous(s):
    s=s.split(',')
    s= [float(i) for i in s]
    if len(s) == 1:
        return s[0]
    if len(s) == 2:
        rg = s[1] - s[0]
        return random.random() * rg + s[0]
    raise ValueError("Length of iterable s should be 1 or 2.")


def sample_discrete(s):
    s=s.split(',')
    if len(s) == 1:
        return s[0]
    return random.choice(s)


def gaussian_blur(img, sigma):
    gaussian_filter(img[:,:,0], output=img[:,:,0], sigma=sigma)
    gaussian_filter(img[:,:,1], output=img[:,:,1], sigma=sigma)
    gaussian_filter(img[:,:,2], output=img[:,:,2], sigma=sigma)


def cv2_jpg(img, compress_val):
    compress_val=int(compress_val)
    img_cv2 = img[:,:,::-1]
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_val]
    result, encimg = cv2.imencode('.jpg', img_cv2, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg[:,:,::-1]


def pil_jpg(img, compress_val):
    compress_val=int(compress_val)
    out = BytesIO()
    img = Image.fromarray(img)
    img.save(out, format='jpeg', quality=compress_val)
    img = Image.open(out)
    # load from memory before ByteIO closes
    img = np.array(img)
    out.close()
    return img


jpeg_dict = {'cv2': cv2_jpg, 'pil': pil_jpg}
def jpeg_from_key(img, compress_val, key):
    method = jpeg_dict[key]
    return method(img, compress_val)
def get_train_transform():
    # ===================== Revised preprocessing pipeline =====================
    # 1. Start with an empty transform list.
    transform_list = [
        transforms.Lambda(translate_duplicate),
        transforms.RandomCrop((224, 224)),
        # Enable random rotation.
        transforms.Lambda(random_rotate),
        transforms.RandomHorizontalFlip(0.5),
    ]

    # 2. Conditionally add data augmentation without appending None.
    if opt.use_data_augument:
        transform_list.append(transforms.Lambda(lambda img: data_augment(img, opt)))

    # 3. Add normalization last.
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )
    ])

    # 4. Build the final Compose object without None entries.
    transform = transforms.Compose(transform_list)
    return transform
def get_dataset_train(root, classes, max_samples=12000):
    random.seed(42)  # Set the random seed for reproducibility.
    # Accept either a list of class names or a space/comma-separated string
    if isinstance(classes, str):
        classes = classes.replace(',', ' ').split()
    dataset_list = []
    transform = get_train_transform()
    for category in classes:
        if category == 'progan':  # Categories stored in subfolders.
            max_samples_per_class = max_samples // len(os.listdir(os.path.join(root,category)))
            for path in os.listdir(os.path.join(root,category)):
                category_path = os.path.join(root, category, path)
                sub_dataset = datasets.ImageFolder(root=category_path, transform=transform)
                total_samples = len(sub_dataset)
                if max_samples_per_class is not None:
                    if total_samples > max_samples_per_class:
                        # Too many samples: downsample without replacement to the target size.
                        random_indices = random.sample(range(total_samples), max_samples_per_class)
                        sub_dataset = torch.utils.data.Subset(sub_dataset, random_indices)
                    elif total_samples < max_samples_per_class:
                        # Too few samples: resample with replacement to the target size.
                        random_indices = random.choices(range(total_samples), k=max_samples_per_class)
                        sub_dataset = torch.utils.data.Subset(sub_dataset, random_indices)
                dataset_list.append(sub_dataset)
        else:
            max_samples_per_class = max_samples
            category_path = os.path.join(root, category)
            sub_dataset = datasets.ImageFolder(root=category_path, transform=transform)
            total_samples = len(sub_dataset)
            if max_samples_per_class is not None:
                if total_samples > max_samples_per_class:
                    # Too many samples: downsample without replacement to the target size.
                    random_indices = random.sample(range(total_samples), max_samples_per_class)
                    sub_dataset = torch.utils.data.Subset(sub_dataset, random_indices)
                elif total_samples < max_samples_per_class:
                    # Too few samples: resample with replacement to the target size.
                    random_indices = random.choices(range(total_samples), k=max_samples_per_class)
                    sub_dataset = torch.utils.data.Subset(sub_dataset, random_indices)
            dataset_list.append(sub_dataset)
    return torch.utils.data.ConcatDataset(dataset_list)

def get_dataset_test(root, classes):
    dataset_list = []
    transform = transforms.Compose([
            transforms.Lambda(translate_duplicate),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
        ])
    if isinstance(classes, str):
        classes = classes.replace(',', ' ').split()
    for category in classes:
        if category == 'progan':  # Categories stored in subfolders.
            for path in os.listdir(os.path.join(root,category)):
                category_path = os.path.join(root, category, path)
                sub_dataset = datasets.ImageFolder(root=category_path, transform=transform)
                dataset_list.append(sub_dataset)
        else:
            category_path = os.path.join(root, category)
            sub_dataset = datasets.ImageFolder(root=category_path, transform=transform)
            dataset_list.append(sub_dataset)
    return torch.utils.data.ConcatDataset(dataset_list)
def get_bal_sampler(dataset):
    targets = []
    for sub_dataset in dataset.datasets:
        # Handle both Subset and non-Subset datasets.
        if isinstance(sub_dataset, torch.utils.data.Subset):
            # For Subset, read targets from the original dataset.
            original_targets = sub_dataset.dataset.targets
            # Then gather the targets for the selected indices.
            current_targets = [original_targets[i] for i in sub_dataset.indices]
        else:
            # For a plain ImageFolder, use targets directly.
            current_targets = sub_dataset.targets
        
        targets.extend(current_targets)

    class_counts = np.bincount(targets)
    weights = 1.0 / class_counts[targets]
    weights = torch.from_numpy(weights.astype(np.float32))
    sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
    return sampler


def load_checkpoint(model, weights):
    checkpoint = torch.load(weights)
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except:
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # Remove the `module.` prefix.
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
