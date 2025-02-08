import torch
from torch import optim, nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from models.segnet import SegNet 
from dataset import OxfordIIITPetsFactory


if __name__ == "__main__":
    LEARNING_RATE = 1e-3
    SCHEDULAR_SZIE  = 7
    SCHEDULAR_GAMMA = 0.7
    BATCH_SIZE = 32 
    EPOCHS = 20
    DATA_PATH = "../data/OxfordPets"
    MODEL_NAME = "SegNet"
    MODEL_SAVE_PATH = f"./saved_models/{MODEL_NAME}.pth"

    device = (
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )

    dataset = OxfordIIITPetsFactory.create(DATA_PATH+"/train", "trainval")
    train_dataset, val_dataset = random_split(dataset, [0.8, 0.2])
    train_dataloader = DataLoader(dataset=train_dataset,
                                batch_size=BATCH_SIZE,
                                shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset,
                                batch_size=BATCH_SIZE,
                                shuffle=True)

    if MODEL_NAME == "SegNet":
        model = SegNet(kernel_size=3).to(device)
    else:
        print(f"Wrong model name: {MODEL_NAME}")
        exit(0)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    schedular = optim.lr_scheduler.StepLR(optimizer, step_size=SCHEDULAR_SZIE, gamma=SCHEDULAR_SZIE)
    criterion = nn.CrossEntropyLoss(reduction='mean')

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

        print("-"*80 + "\n")
        print(f"Train Loss EPOCH {epoch+1}: {train_loss:.4f}")
        print(f"Valid Loss EPOCH {epoch+1}: {val_loss:.4f}")
        print("-"*80 + "\n")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
