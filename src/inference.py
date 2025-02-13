import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import torchmetrics

import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

from dataset import OxfordIIITPetsFactory, TrimapClasses
from utils.loss import IoUMetric
from models.segnet import SegNet 
from models.segnetdws import DWSSegNet
from models.vggnet import  VGGNet
from models.unet import UNet 

parser = argparse.ArgumentParser(
                    prog='Segmentation Tester',
                    description='App to efficiently test different nn image segmentation architectures',
                    epilog='and may God help us')
parser.add_argument('-m', '--model')     
parser.add_argument('-s', '--save_name')     
parser.add_argument('-t', '--transform', action=argparse.BooleanOptionalAction)

model_mapping = {
    "SegNet": SegNet,
    "DWSSegNet": DWSSegNet,
    "VGGNet": VGGNet,
    "UNet": UNet,
}

def print_test_dataset_masks(model_pth, model_name, save_name, batch_size, show_plot, device,  transform):
    if model_name in model_mapping:
        model = model_mapping[MODEL_NAME](kernel_size=3).to(device)
    else:
        print(f"Wrong model name: {MODEL_NAME}")
        exit(0)
        
    model.load_state_dict(torch.load(model_pth, map_location=torch.device(device)))
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='mean')
    test_dataset = OxfordIIITPetsFactory.create(DATA_PATH+"/test", "test", transform=transform) 
    test_dataloader = DataLoader(dataset=test_dataset,
                                batch_size=batch_size,
                                shuffle=True)
    test_running_loss = 0
    test_pixel_accuracy_running = 0
    test_iou_accuracy_running = 0
    test_custom_iou_running = 0

    plt_images, plt_masks, plt_preds = None, None, None

    with torch.inference_mode():
        for idx, (img, mask) in enumerate(tqdm(test_dataloader)):
            if plt_images is None and plt_masks is None:
                plt_images = img
                plt_masks = mask

            img, mask = img.to(device), mask.to(device) 
            pred = model(img)

            mask_squeezed = mask.squeeze(dim=1)
            loss = criterion(pred, mask_squeezed)
            test_running_loss += loss.item()

            pred_sm = nn.Softmax(dim=1)(pred)
            pred_labels = pred_sm.argmax(dim=1)
            pred_labels = pred_labels.unsqueeze(1)
            pred_mask = pred_labels.to(torch.float)
            if plt_preds is None:
                plt_preds = pred_mask.cpu()

            iou = torchmetrics.classification.MulticlassJaccardIndex(3, average='micro', ignore_index=TrimapClasses.BACKGROUND)
            iou = iou.to(device)
            test_iou_accuracy_running += iou(pred_mask, mask)

            pixel_metric = torchmetrics.classification.MulticlassAccuracy(3, average='micro')
            pixel_metric = pixel_metric.to(device)
            test_pixel_accuracy_running += pixel_metric(pred_labels, mask)

            test_custom_iou_running += IoUMetric(pred, mask)
            # test_custom_iou_running = 0 
        test_loss = test_running_loss / (idx + 1)
        pixel_accuracy = test_pixel_accuracy_running / (idx + 1)
        iou_accuracy = test_iou_accuracy_running / (idx + 1)
        custom_iou = test_custom_iou_running / (idx + 1)    
    parameters_count = sum(p.numel() for p in model.parameters()) 
    title = f'{model_name} - {parameters_count} parameters\nLoss: {test_loss:.4f} Accuracy[Pixel: {pixel_accuracy:.4f}, IoU: {iou_accuracy:.4f}, Custom IoU: {custom_iou:.4f}]'

    while len(plt.get_fignums()) > 0:
        plt.close()
        
    fig = plt.figure(figsize=(10, 12))
    fig.suptitle(title, fontsize=12)
    to_image = transforms.ToPILImage()

    fig.add_subplot(3, 1, 1)
    plt.imshow(to_image(torchvision.utils.make_grid(plt_images, nrow=7)))
    plt.axis('off')
    plt.title("Targets")

    fig.add_subplot(3, 1, 2)
    plt.imshow(to_image(torchvision.utils.make_grid(plt_masks.float() / 2.0, nrow=7)))
    plt.axis('off')
    plt.title("Ground Truth Labels")

    fig.add_subplot(3, 1, 3)
    plt.imshow(to_image(torchvision.utils.make_grid(plt_preds / 2.0, nrow=7)))
    plt.axis('off')
    plt.title("Predicted Labels")
    
    print(title) 
    if show_plot is False:
        while len(plt.get_fignums()) > 0:
            plt.close()
    else:
        plt.savefig(f"results/{model_name}_{save_name}.png")

if __name__ == "__main__":
    args = parser.parse_args()

    SAVE_NAME = args.save_name
    MODEL_NAME = args.model
    MODEL_PATH = f"./saved_models/{MODEL_NAME}_{SAVE_NAME}.pth"
    DATA_PATH = "../data/OxfordPets"
    BATCH_SIZE = 21
    device = (
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )

    print_test_dataset_masks(MODEL_PATH, MODEL_NAME, SAVE_NAME, BATCH_SIZE, True, device, args.transform)
