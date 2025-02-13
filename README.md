## Repo for Testing Different Neural Network Architectures for Bachelor's Diploma Work at LPNU

### Dataset
The training was conducted using the **Oxford-IIIT Pet Dataset** because of a love for dogs and cats. More information about the dataset can be found [here](https://www.robots.ox.ac.uk/~vgg/data/pets/).

### Training Parameters:
- Optimizer: **Adam**
- Learning Rate: **1e-3**
- Scheduler Size: **7**
- Scheduler Gamma: **0.7**
- Batch Size: **32**
- Epochs: **20**

### Hardware:
The models were trained on an **Nvidia GeForce RTX 3080 Max-Q** with **8 GB GDDR6 VRAM**.

### List of Models and Hyperparameters:
1. **SegNet** - [Paper](https://arxiv.org/abs/1511.00561v3)  
   - **Size**: 15,274,892 parameters
   - **Loss**: 0.4629  
   - **Accuracy**:  
     - Pixel Accuracy: 0.8277  
     - IoU: 0.6460  
   - **Note**:
   The model I decided to start with is very simple but beautiful.
2. **SegNet with Depthwise Separable Convolutions** - [Paper](https://arxiv.org/abs/1704.04861)  
   - **Size**: 1,749,671 parameters
   - **Loss**: 0.4899
   - **Accuracy**:  
     - Pixel Accuracy: 0.7960
     - IoU: 0.6883
  - **Note**:
  After modifying SegNet to use Depthwise Separable Convolutions, I was able to reduce the size by ~88%, with a 4% increase in loss
3. **VGG for Segmentation** - [Paper](https://arxiv.org/abs/1409.1556v6)  
   - **Size**: 113,356,131 parameters
   - **Loss**: 0.6833
   - **Accuracy**:  
     - Pixel Accuracy: 0.7327
     - IoU: 0.5505
  - **Note**:
  I really think I've messed up here somewhere, because VGG should be better, but I don't see an error now. Will return to VGG.
4. **UNet** - [Paper](https://arxiv.org/abs/1505.04597)  
   - **Size**: 31,026,563 parameters
   - **Loss**: 0.3523
   - **Accuracy**:  
     - Pixel Accuracy: 0.8686
     - IoU: 0.697
  - **Note**:
  Linear verion of UNet. For some reason LR scheduler destroys grad on step 