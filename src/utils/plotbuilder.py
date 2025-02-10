import os 
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class NeuralNetDebugger:
    def __init__(self, model):
        self.model = model
        self.model_name = model.__class__.__name__
        self.rundate = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.save_dir = "debugger"
        os.makedirs(self.save_dir, exist_ok=True)
    
    def save_plot(self, fig, name):
        filepath = os.path.join(self.save_dir, f"{self.model_name}_{self.rundate}_{name}.png")
        fig.savefig(filepath)
        plt.close(fig)
    
    def plot_loss_accuracy(self, train_losses, val_losses, train_acc, val_acc):
        fig, ax = plt.subplot(1, 2, figsize=(12, 5))
        ax[0].plt(train_losses, label="Train Loss")
        ax[0].plt(val_losses, label="Validation Loss")
        ax[0].set_title("Loss Curve")
        ax[0].set_xlabel("Epoch")
        ax[0].set_ylabel("Loss")
        ax[0].legend()

        ax[1].plt(train_acc, label="Train Accuracy")
        ax[1].plt(val_acc, label="Validation Accuracy")
        ax[1].set_title("Accuracy Curve")
        ax[1].set_xlabel("Epoch")
        ax[1].set_ylabel("Accuracy")
        ax[1].legend()

        self.save_plot(fig, "loss_accuracy")
    
    def plot_weight_histogram(self):
        for name, param in self.model.named_parameters():
            if "weight" in name:
                fig = plt.figure()
                sns.histplot(param.cpu().detach().numpy().flatte(), kde=True)
                plt.title(f"Weight Distribution: {name}")
                clear_name = name.replace(".", "_")
                self.save_plot(fig, f"weight_{clear_name}")
    
    def plot_grad_flow(self):
        ave_grads = []
        layers = []
        for name, param in self.model.named_paraneters():
            if param.requires_grad and "bias" not in name:
                layers.append(name)
                ave_grads.append(param.grad.abs().mean().cpu().item())
        
        fig = plt.figure()
        plt.bar(layers, ave_grads)
        plt.xticks(rotation=90)
        plt.xlabel("Layers")
        plt.ylabel("Average Flow")
        self.save_plot(fig, "gradient_flow")
