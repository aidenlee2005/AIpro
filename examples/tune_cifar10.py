import numpy as np
import pickle
import sys
import os
import time
import itertools

# Add framework directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../framework"))

from tensor import TensorFull as Tensor
import optimizer as nn
import operators as F

def load_cifar_batch(filename):
    with open(filename, 'rb') as f:
        datadict = pickle.load(f, encoding='bytes')
        X = datadict[b'data']
        Y = datadict[b'labels']
        X = X.reshape(10000, 3, 32, 32).astype("float32")
        Y = np.array(Y, dtype="float32")
        return X, Y

def load_cifar10(root_dir):
    xs = []
    ys = []
    for b in range(1, 6):
        f = os.path.join(root_dir, 'data_batch_%d' % (b, ))
        X, Y = load_cifar_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    
    Xte, Yte = load_cifar_batch(os.path.join(root_dir, 'test_batch'))
    return Xtr, Ytr, Xte, Yte

class CIFAR10Dataset:
    def __init__(self, X, Y, device="gpu"):
        self.X = X
        self.Y = Y
        self.device = device
        
        # Normalize
        self.X = self.X / 255.0
        # Mean/Std normalization
        mean = np.array([0.4914, 0.4822, 0.4465]).reshape(1, 3, 1, 1)
        std = np.array([0.2023, 0.1994, 0.2010]).reshape(1, 3, 1, 1)
        self.X = (self.X - mean) / std

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

def get_batch(X, Y, batch_size, device="gpu"):
    num_samples = X.shape[0]
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    
    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_idx = indices[start_idx:end_idx]
        
        batch_x = Tensor(X[batch_idx], device=device)
        batch_y = Tensor(Y[batch_idx], device=device)
        
        yield batch_x, batch_y

def create_model(dropout_p=0.0, device="gpu"):
    # VGG-style deeper model for >80% accuracy
    model = nn.Sequential(
        # Block 1
        nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, device=device),
        nn.BatchNorm2d(32, device=device),
        nn.ReLU(),
        nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, device=device),
        nn.BatchNorm2d(32, device=device),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2), # 32 -> 16

        # Block 2
        nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, device=device),
        nn.BatchNorm2d(64, device=device),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, device=device),
        nn.BatchNorm2d(64, device=device),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2), # 16 -> 8

        # Block 3
        nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, device=device),
        nn.BatchNorm2d(128, device=device),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2), # 8 -> 4
        
        nn.Flatten(),
        nn.Linear(128 * 4 * 4, 256, device=device),
        nn.ReLU(),
        nn.Dropout(dropout_p, device=device) if dropout_p > 0 else nn.Sequential(), # Optional Dropout
        nn.Linear(256, 10, device=device)
    )
    return model

def train_one_config(config, X_train, Y_train, X_test, Y_test, device="gpu", log_file=None):
    lr = config['lr']
    batch_size = config['batch_size']
    dropout_p = config.get('dropout_p', 0.0) # Default to 0 if not present
    weight_decay = config['weight_decay']
    epochs = config['epochs']
    optimizer_name = config.get('optimizer', 'SGD')
    momentum = config.get('momentum', 0.0)
    
    msg = f"\n[Config] Opt: {optimizer_name}, LR: {lr}, BS: {batch_size}, Mom: {momentum}, WD: {weight_decay}, Drop: {dropout_p}"
    print(msg)
    if log_file:
        log_file.write(msg + "\n")
        log_file.flush()
    
    model = create_model(dropout_p, device)
    criterion = nn.CrossEntropyLoss()
    
    if optimizer_name == 'Adam':
        optimizer = nn.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = nn.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    
    best_acc = 0.0
    total_time = 0
    
    for epoch in range(epochs):
        # Simple LR Decay
        if epoch == 30 or epoch == 40:
            optimizer.lr *= 0.1
            print(f"  [LR Decay] LR set to {optimizer.lr}")
            if log_file:
                log_file.write(f"  [LR Decay] LR set to {optimizer.lr}\n")

        model.train()
        epoch_start = time.time()
        total_loss = 0
        correct = 0
        total = 0
        
        # Training loop
        num_batches = 0
        for batch_x, batch_y in get_batch(X_train, Y_train, batch_size, device):
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.numpy().item()
            
            # Calculate accuracy
            pred = np.argmax(outputs.numpy(), axis=1)
            correct += np.sum(pred == batch_y.numpy())
            total += batch_y.shape[0]
            num_batches += 1
            
        epoch_time = time.time() - epoch_start
        total_time += epoch_time
        avg_loss = total_loss / num_batches
        train_acc = correct / total
        
        # Evaluation
        model.eval()
        test_correct = 0
        test_total = 0
        for batch_x, batch_y in get_batch(X_test, Y_test, batch_size, device): # Use same batch size for eval
            outputs = model(batch_x)
            pred = np.argmax(outputs.numpy(), axis=1)
            test_correct += np.sum(pred == batch_y.numpy())
            test_total += batch_y.shape[0]
            
        test_acc = test_correct / test_total
        best_acc = max(best_acc, test_acc)
        
        msg = f"  Ep {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | TrAcc: {train_acc:.4f} | TeAcc: {test_acc:.4f} | Time: {epoch_time:.2f}s"
        print(msg)
        if log_file:
            log_file.write(msg + "\n")
            log_file.flush()
        
    return best_acc, total_time

def main():
    print("Loading CIFAR-10 data...")
    data_dir = os.path.join(os.path.dirname(__file__), "../data/cifar-10-batches-py")
    X_train, Y_train, X_test, Y_test = load_cifar10(data_dir)
    
    # Pre-process data (normalization)
    mean = np.array([0.4914, 0.4822, 0.4465]).reshape(1, 3, 1, 1).astype(np.float32)
    std = np.array([0.2023, 0.1994, 0.2010]).reshape(1, 3, 1, 1).astype(np.float32)
    
    X_train = (X_train / 255.0 - mean) / std
    X_test = (X_test / 255.0 - mean) / std
    
    print(f"Train data: {X_train.shape}")
    
    # Define search space
    # Time constraint: ~40s/epoch * 50 epochs = 2000s (~33 mins) per run.
    configs = [
        # Best Config from previous run: Adam + Dropout 0.3
        {'optimizer': 'Adam', 'lr': 0.001, 'batch_size': 64, 'momentum': 0.0, 'weight_decay': 1e-4, 'epochs': 50, 'dropout_p': 0.3},
    ]
    
    results = []
    start_total_time = time.time()
    time_limit = 120 * 60 # 120 minutes
    
    print(f"Starting training with best configuration (Time limit: {time_limit/60} mins)...")
    
    log_filename = "tuning_results.txt"
    with open(log_filename, "w") as log_file:
        log_file.write(f"Starting Tuning at {time.ctime()}\n")
        
        for i, config in enumerate(configs):
            if time.time() - start_total_time > time_limit:
                print("Time limit reached. Stopping tuning.")
                break
                
            msg = f"--- Running Config {i+1}/{len(configs)} ---"
            print(msg)
            log_file.write(msg + "\n")
            
            acc, duration = train_one_config(config, X_train, Y_train, X_test, Y_test, device="gpu", log_file=log_file)
            
            result = config.copy()
            result['best_acc'] = acc
            result['duration'] = duration
            results.append(result)
            
        # Print Summary
        print("\n" + "="*100)
        print(f"{'Opt':<6} | {'LR':<8} | {'BS':<6} | {'Mom':<6} | {'WD':<8} | {'Drop':<6} | {'Epochs':<6} | {'Best Acc':<10} | {'Time(s)':<8}")
        print("-" * 100)
        
        log_file.write("\n" + "="*100 + "\n")
        log_file.write(f"{'Opt':<6} | {'LR':<8} | {'BS':<6} | {'Mom':<6} | {'WD':<8} | {'Drop':<6} | {'Epochs':<6} | {'Best Acc':<10} | {'Time(s)':<8}\n")
        log_file.write("-" * 100 + "\n")
        
        # Sort by accuracy descending
        results.sort(key=lambda x: x['best_acc'], reverse=True)
        
        for r in results:
            opt = r.get('optimizer', 'SGD')
            mom = r.get('momentum', 0.0)
            drop = r.get('dropout_p', 0.0)
            line = f"{opt:<6} | {r['lr']:<8} | {r['batch_size']:<6} | {mom:<6} | {r['weight_decay']:<8} | {drop:<6} | {r['epochs']:<6} | {r['best_acc']:.4f}     | {r['duration']:.1f}"
            print(line)
            log_file.write(line + "\n")
            
        print("="*100)
        log_file.write("="*100 + "\n")
        best_config_msg = f"Best Configuration: Opt={results[0].get('optimizer', 'SGD')}, LR={results[0]['lr']}, Drop={results[0].get('dropout_p', 0.0)}"
        print(best_config_msg)
        log_file.write(best_config_msg + "\n")

if __name__ == "__main__":
    main()
