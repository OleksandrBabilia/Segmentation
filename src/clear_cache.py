import torch
import gc

if __name__ == "__main__":
    torch.cuda.empty_cache()
    gc.collect()