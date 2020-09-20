from validate import validate_model
from train import run_training

def monitor():
    if validate_model():
        pass
    else:
        print("retraining")
        run_training()
    
if __name__ == "__main__":
    monitor()
