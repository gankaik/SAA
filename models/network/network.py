import torch
import torch.nn as nn
import sys
import types
import random
from transformers import CLIPVisionModel
# Compatibility shim: older huggingface_hub versions may not expose huggingface_hub.errors.
try:
    from huggingface_hub.errors import HFValidationError  # noqa: F401
except Exception:
    try:
        from huggingface_hub.utils._validators import HFValidationError

        errors_module = types.ModuleType("huggingface_hub.errors")
        errors_module.HFValidationError = HFValidationError
        sys.modules["huggingface_hub.errors"] = errors_module
    except Exception:
        pass
from peft import LoraConfig, get_peft_model


def set_seed(seed=42, deterministic=True):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        # Use deterministic behavior where possible for reproducibility.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class ViTMixStyle(nn.Module):
    def __init__(self, p=0.5, alpha=0.1, eps=1e-6):
        """
        ViT version of MixStyle.
        :param p: Probability of applying MixStyle (0.5 is usually a good default).
        :param alpha: Beta distribution parameter that controls the mixing strength.
        :param eps: Small constant to avoid division by zero.
        """
        super().__init__()
        self.p = p
        self.beta = torch.distributions.Beta(alpha, alpha)
        self.eps = eps

    def forward(self, x, perm=None):
        # Only apply during training, and only with probability p.
        if not self.training or random.random() > self.p:
            return x

        B, S, C = x.shape

        # Compute mean and standard deviation along the sequence dimension (ViT-style instance normalization).
        mu = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        sig = torch.sqrt(var + self.eps)

        # Normalize features to remove style information.
        x_normed = (x - mu) / sig

        # Generate a random permutation within the batch to borrow style statistics.
        if perm is None:
            perm = torch.randperm(B, device=x.device)

        # Gather shuffled style statistics.
        mu_perm = mu[perm]
        sig_perm = sig[perm]

        # Sample mixing coefficients from the Beta distribution on the same device.
        lam = self.beta.sample((B, 1, 1)).to(x.device)

        # Mix the current style with the borrowed style.
        mu_mix = mu * lam + mu_perm * (1 - lam)
        sig_mix = sig * lam + sig_perm * (1 - lam)

        # Inject the mixed style back into the normalized features.
        x_mixed = x_normed * sig_mix + mu_mix

        return x_mixed

class SAP_Net(nn.Module):
    def __init__(self, pretrained_model_name="openai/clip-vit-large-patch14", r=8,alpha=0.1,target_layer_idx=18,seed=42):
        super().__init__()
        set_seed(seed)
        # 1. Load the CLIP backbone and freeze it.
        self.clip_vision = CLIPVisionModel.from_pretrained(pretrained_model_name)
        for param in self.clip_vision.parameters():
            param.requires_grad = False
            
        # 2. Inject LoRA adapters.
        lora_config = LoraConfig(
            r=r, lora_alpha=16, 
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
            lora_dropout=0.0, bias="none"
        )
        self.clip_vision = get_peft_model(self.clip_vision, lora_config)
        
        # 3. Classification head.
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1024, 1)
        )
        
        # 4. Initialize the MixStyle module.
        # alpha=0.1 is a commonly used generalization-friendly value.
        self.mixstyle = ViTMixStyle(p=0.5, alpha=alpha)
        
        # 5. Register hooks (mid-to-deep layers are usually a good choice, e.g. layer 18).
        try:
            self.encoder_layers = self.clip_vision.base_model.model.vision_model.encoder.layers
        except AttributeError:
            self.encoder_layers = self.clip_vision.vision_model.encoder.layers
            
        # Apply MixStyle on layers 5 to 9, where semantic features are forming but before the final decision layer.
        self.target_layer_idx = target_layer_idx
        self._register_mixstyle_hook()

    def _register_mixstyle_hook(self):
        layer_indices = self.target_layer_idx
        if isinstance(layer_indices, int):
            layer_indices = (layer_indices,)

        def hook_fn(module, inputs, output):
            hidden_states = output[0] if isinstance(output, (tuple, list)) else output

            # 1. Separate the [CLS] token.
            cls_token = hidden_states[:, 0:1, :]
            patch_tokens = hidden_states[:, 1:, :]

            # 2. If targets are provided, apply MixStyle only on fake samples and
            # build permutation only from fake indices.
            current_targets = getattr(module, "_current_targets", None)
            if current_targets is None:
                perm = torch.randperm(patch_tokens.size(0), device=patch_tokens.device)
                mixed_patch_tokens = self.mixstyle(patch_tokens, perm=perm)
            else:
                targets = current_targets.view(-1).to(patch_tokens.device)
                fake_mask = targets >= 0.5
                fake_indices = torch.nonzero(fake_mask, as_tuple=False).view(-1)

                mixed_patch_tokens = patch_tokens
                if fake_indices.numel() > 0:
                    fake_patch_tokens = patch_tokens[fake_indices]
                    fake_perm = torch.randperm(fake_indices.numel(), device=patch_tokens.device)
                    mixed_fake_patch_tokens = self.mixstyle(fake_patch_tokens, perm=fake_perm)

                    mixed_patch_tokens = patch_tokens.clone()
                    mixed_patch_tokens[fake_indices] = mixed_fake_patch_tokens

            # 3. Recombine the untouched global semantics with the perturbed local features.
            mixed_states = torch.cat([cls_token, mixed_patch_tokens], dim=1)

            if isinstance(output, (tuple, list)):
                return (mixed_states,) + tuple(output[1:])
            return mixed_states

        self.hook_handles = []
        for idx in layer_indices:
            target_layer = self.encoder_layers[idx]
            self.hook_handles.append(target_layer.register_forward_hook(hook_fn))


    def forward(self, x, targets=None):
        # The hook automatically triggers MixStyle in the configured layer range.
        layer_indices = self.target_layer_idx if isinstance(self.target_layer_idx, (tuple, list)) else (self.target_layer_idx,)
        for idx in layer_indices:
            self.encoder_layers[idx]._current_targets = targets

        outputs = self.clip_vision(pixel_values=x)
        logits = self.classifier(outputs.pooler_output)

        for idx in layer_indices:
            if hasattr(self.encoder_layers[idx], "_current_targets"):
                del self.encoder_layers[idx]._current_targets

        return logits
