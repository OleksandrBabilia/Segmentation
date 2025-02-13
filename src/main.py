import torch
from torch import optim, nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import argparse

from dataset import OxfordIIITPetsFactory
from utils.plotbuilder import NeuralNetDebugger
from models.segnet import SegNet 
from models.segnetdws import DWSSegNet
from models.vggnet import VGGNet 
from models.unet import UNet 
from models.unet_bilinear import UNetBilinear 

parser = argparse.ArgumentParser(
                    prog='Segmentation Tester',
                    description='App to efficiently test different nn image segmentation architectures',
                    epilog='and may God help us')
parser.add_argument('-m', '--model')     
parser.add_argument('-s', '--save_name')     
parser.add_argument('-t', '--transform', action=argparse.BooleanOptionalAction)

if __name__ == "__main__":
    args = parser.parse_args()
    LEARNING_RATE = 1e-3
    SCHEDULAR_SZIE  = 7
    SCHEDULAR_GAMMA = 0.7
    BATCH_SIZE = 32 
    EPOCHS = 20
    DATA_PATH = "../data/OxfordPets"
    MODEL_NAME = args.model
    MODEL_SAVE_PATH = f"./saved_models/{MODEL_NAME}_{args.save_name}.pth"

    device = (
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )
    model_mapping = {
        "SegNet": SegNet,
        "DWSSegNet": DWSSegNet,
        "VGGNet": VGGNet,
        "UNet": UNet,
        "UNetBilinear": UNetBilinear
    }

    dataset = OxfordIIITPetsFactory.create(DATA_PATH+"/train", "trainval", transform=args.transform)
    train_dataset, val_dataset = random_split(dataset, [0.8, 0.2])
    train_dataloader = DataLoader(dataset=train_dataset,
                                batch_size=BATCH_SIZE,
                                shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset,
                                batch_size=BATCH_SIZE,
                                shuffle=True)

    if MODEL_NAME in model_mapping:
        model = model_mapping[MODEL_NAME](kernel_size=3).to(device)
    else:
        print(f"Wrong model name: {MODEL_NAME}")
        exit(0)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # schedular = optim.lr_scheduler.StepLR(optimizer, step_size=SCHEDULAR_SZIE, gamma=SCHEDULAR_SZIE)
    schedular = None
    criterion = nn.CrossEntropyLoss(reduction='mean')
    debugger = NeuralNetDebugger(model)
    train_losses, val_losses = [], []

    for epoch in tqdm(range(EPOCHS)):
        model.train()
        train_running_loss = 0
        for idx, (img, mask) in enumerate(tqdm(train_dataloader)):
            img, mask = img.to(device), mask.to(device) 

            pred = model(img)
            mask = mask.squeeze(dim=1)
            optimizer.zero_grad()
            
            loss = criterion(pred, mask)
            train_running_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            if idx % 10 == 0: # I think there is got to be a better way for tracking loss
                train_losses.append(loss)

        train_loss = train_running_loss / (idx + 1)

        if schedular:
            schedular.step()

        model.eval()
        val_running_loss = 0
        with torch.inference_mode():
            for idx, (img, mask) in enumerate(tqdm(val_dataloader)):
                img, mask = img.to(device), mask.to(device) 
                pred = model(img)
                mask = mask.squeeze(dim=1)

                loss = criterion(pred, mask)

                val_running_loss += loss.item()

            val_loss = val_running_loss / (idx + 1)

            if idx % 10 == 0: # Here too
                val_losses.append(loss)
        print("-"*80 + "\n")
        print(f"Train Loss EPOCH {epoch+1}: {train_loss:.4f}")
        print(f"Valid Loss EPOCH {epoch+1}: {val_loss:.4f}")
        print("-"*80 + "\n")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    # debugger.plot_loss_accuracy(train_losses, val_losses, None, None)