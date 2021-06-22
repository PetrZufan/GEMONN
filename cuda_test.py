import torch

def cudaTest():
    print("testing torch...")
    print(torch.rand(5, 3))
    print()
    print("testing torch and cuda...")
    print(torch.cuda.is_available())
