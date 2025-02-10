import torch
import torchvision
from torchvision import transforms
from enum import IntEnum


class OxfordIIITPetsAugmented(torchvision.datasets.OxfordIIITPet):
    def __init__(
        self, root, split, target_types="segmentation", download=False, pre_transform=None,
        post_transform=None, pre_target_transform=None, post_target_transform=None, common_transform=None):
        
        super().__init__(root=root,split=split, target_types=target_types, download=download,
                         transform=pre_transform, target_transform=pre_target_transform)
        
        self.post_transform = post_transform
        self.post_target_transform = post_target_transform
        self.common_transform = common_transform

    def __len__(self):
        return super().__len__()

    def __getitem__(self, idx):
        img, mask = super().__getitem__(idx)
        
        if self.common_transform:
            entry = self.common_transform(torch.cat([img, mask], dim=0))
            img, mask = torch.split(entry, 3, dim=0)
        
        if self.post_transform:
            img = self.post_transform(img)
            
        if self.post_target_transform:
            mask = self.post_target_transform(mask)

        return (img, mask)


class TrimapClasses(IntEnum):
    PET = 0
    BACKGROUND = 1
    BORDER = 2


class OxfordIIITPetsFactory:
    device = None
    
    @classmethod
    def get_device(cls):
        if cls.device is None:
            cls.device = (
                "cuda" if torch.cuda.is_available() else
                "mps" if torch.backends.mps.is_available() else
                "cpu"
            )
        return cls.device
    
    @classmethod
    def get_transforms(cls, transform=False):
        device = cls.get_device()
        if transform:
            return transforms.Compose([
                transforms.Lambda(lambda x: x.to(device)),
                transforms.Resize((128, 128), interpolation=transforms.InterpolationMode.NEAREST),
                transforms.RandomHorizontalFlip(p=0.5),
            ])
        return transforms.Compose([
            transforms.Lambda(lambda x: x.to(device)),
            transforms.Resize((128, 128), interpolation=transforms.InterpolationMode.NEAREST),
        ])
    
    @classmethod
    def create(cls, root, split, transform=False):
        if transform:
            return OxfordIIITPetsAugmented(root, split=split, target_types="segmentation", download=False,
                                           pre_target_transform=transforms.ToTensor(), pre_transform=transforms.ToTensor(),
                                           common_transform=cls.get_transforms(transform=transform), post_transform=transforms.ColorJitter(contrast=0.3),
                                           post_target_transform=transforms.Lambda(lambda x: (255 * x).to(torch.long) -1))
        return OxfordIIITPetsAugmented(root, split=split, target_types="segmentation", download=False,
                                           pre_target_transform=transforms.ToTensor(), pre_transform=transforms.ToTensor(),
                                           common_transform=cls.get_transforms(transform=transform),
                                           post_target_transform=transforms.Lambda(lambda x: (255 * x).to(torch.long) -1))
