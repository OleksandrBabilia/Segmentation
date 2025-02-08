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
   - **Loss**: 0.4629  
   - **Accuracy**:  
     - Pixel Accuracy: 0.8277  
     - IoU: 0.6460  
     - Custom IoU: 1.6417  
