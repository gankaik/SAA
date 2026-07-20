import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import os
import csv
import time

class SAP_Trainer:
    def __init__(self, model, train_loader, val_loader, device="cuda", epochs=10, save_dir=".", writer=None, log_interval=20):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epochs = epochs  # Paper setting: 10 epochs [cite: 844]
        self.save_dir = save_dir
        self.writer = writer
        self.log_interval = log_interval
        os.makedirs(self.save_dir, exist_ok=True)
        self.metrics_path = os.path.join(self.save_dir, "sap_metrics.csv")
        if not os.path.exists(self.metrics_path):
            with open(self.metrics_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr", "epoch_time_sec"])
        
        # 1. Loss function: binary cross-entropy (BCE).
        # BCEWithLogitsLoss includes Sigmoid internally, which is numerically more stable.
        self.criterion = nn.BCEWithLogitsLoss()
        
        # 2. Optimizer: AdamW [cite: 844].
        # Only optimize parameters that require gradients, i.e. LoRA and classifier head parameters.
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.AdamW(trainable_params, lr=1e-4) # Initial learning rate: 1e-4 [cite: 845]
        
        # 3. Learning rate scheduler: cosine decay [cite: 845].
        # Decay from 1e-4 to 1e-5 (eta_min=1e-5).
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=1e-5)

    def train_one_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs} [Train]")
        for step, (inputs, targets) in enumerate(pbar, start=1):
            # The paper specifies resizing input images to 224x224.
            inputs = inputs.to(self.device)
            targets = targets.to(self.device).float().unsqueeze(1) # [batch_size, 1]
            
            self.optimizer.zero_grad()
            logits = self.model(inputs,targets=targets)  # Forward pass with optional targets for MixStyle
            loss = self.criterion(logits, targets)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Compute accuracy.
            preds = torch.sigmoid(logits) > 0.5
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            avg_loss = total_loss / step
            avg_acc = correct / max(1, total)
            
            pbar.set_postfix({'loss': f"{avg_loss:.4f}", 'acc': f"{avg_acc:.4f}"})

            if self.log_interval > 0 and (step % self.log_interval == 0 or step == len(self.train_loader)):
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"[Train] epoch={epoch+1}/{self.epochs} step={step}/{len(self.train_loader)} loss={avg_loss:.4f} acc={avg_acc:.4f} lr={current_lr:.6e}")
            
        return total_loss / max(1, len(self.train_loader)), correct / max(1, total)

    @torch.no_grad()
    def validate(self, epoch):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1}/{self.epochs} [Val]")
        for inputs, targets in pbar:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device).float().unsqueeze(1)
            
            logits = self.model(inputs)
            loss = self.criterion(logits, targets)
            
            total_loss += loss.item()
            preds = torch.sigmoid(logits) > 0.5
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            
        val_acc = correct / max(1, total)
        val_loss = total_loss / max(1, len(self.val_loader))
        print(f"[Val] epoch={epoch+1}/{self.epochs} loss={val_loss:.4f} acc={val_acc:.4f}")
        return val_loss, val_acc

    def _append_metrics(self, epoch, train_loss, train_acc, val_loss, val_acc, lr, epoch_time_sec):
        with open(self.metrics_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                f"{train_loss:.6f}",
                f"{train_acc:.6f}",
                f"{val_loss:.6f}",
                f"{val_acc:.6f}",
                f"{lr:.8f}",
                f"{epoch_time_sec:.2f}",
            ])

    def fit(self):
        best_acc = 0.0
        for epoch in range(self.epochs):
            epoch_start = time.time()
            train_loss, train_acc = self.train_one_epoch(epoch)
            val_loss, val_acc = self.validate(epoch)
            
            # Step the learning rate scheduler [cite: 845].
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]
            epoch_time_sec = time.time() - epoch_start
            print(
                f"[Epoch {epoch+1}/{self.epochs}] "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
                f"lr={current_lr:.6e} time={epoch_time_sec:.2f}s"
            )

            if self.writer is not None:
                self.writer.add_scalar("train/loss", train_loss, epoch + 1)
                self.writer.add_scalar("train/acc", train_acc, epoch + 1)
                self.writer.add_scalar("val/loss", val_loss, epoch + 1)
                self.writer.add_scalar("val/acc", val_acc, epoch + 1)
                self.writer.add_scalar("train/lr", current_lr, epoch + 1)

            self._append_metrics(epoch + 1, train_loss, train_acc, val_loss, val_acc, current_lr, epoch_time_sec)

            last_model_path = os.path.join(self.save_dir, "sap_last_model.pth")
            torch.save(self.model.state_dict(), last_model_path)
            
            if val_acc > best_acc:
                best_acc = val_acc
                best_model_path = os.path.join(self.save_dir, "sap_best_model.pth")
                torch.save(self.model.state_dict(), best_model_path)
                print(f"Saved new best model to: {best_model_path}")

        final_model_path = os.path.join(self.save_dir, "sap_final_model.pth")
        torch.save(self.model.state_dict(), final_model_path)
        print(f"Training finished. Final model saved to: {final_model_path}")
        print(f"Metrics csv saved to: {self.metrics_path}")