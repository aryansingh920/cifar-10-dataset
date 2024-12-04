import torch
import torchvision
import sys

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

def test_pytorch_setup():
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Torchvision version: {torchvision.__version__}")
    
    # Test device availability
    print("\nDevice Information:")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    
    # Create a simple tensor and test device movement
    device = get_device()
    print(f"\nUsing device: {device}")
    
    x = torch.randn(2, 3)
    try:
        x = x.to(device)
        print("Successfully created and moved tensor to device")
        print(x)
    except Exception as e:
        print(f"Error when testing tensor operations: {e}")

if __name__ == "__main__":
    test_pytorch_setup()
