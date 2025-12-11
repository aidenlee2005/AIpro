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

def create_model(dropout_p=0.5, device="gpu"):
    model = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, device=device),
        nn.BatchNorm2d(16, device=device),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2), # 32 -> 16
        nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, device=device),
        nn.BatchNorm2d(32, device=device),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2), # 16 -> 8
        nn.Flatten(),
        nn.Dropout(dropout_p, device=device),
        nn.Linear(32 * 8 * 8, 10, device=device)
    )
    return model

def train_one_config(config, X_train, Y_train, X_test, Y_test, device="gpu"):
    lr = config['lr']
    batch_size = config['batch_size']
    dropout_p = config['dropout_p']
    weight_decay = config['weight_decay']
    epochs = config['epochs']
    
    print(f"\n[Config] LR: {lr}, BS: {batch_size}, Drop: {dropout_p}, WD: {weight_decay}")
    
    model = create_model(dropout_p, device)
    criterion = nn.CrossEntropyLoss()
    optimizer = nn.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    
    best_acc = 0.0
    total_time = 0
    
    for epoch in range(epochs):
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
        
        print(f"  Ep {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | TrAcc: {train_acc:.4f} | TeAcc: {test_acc:.4f} | Time: {epoch_time:.2f}s")
        
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
    configs = [
        {'lr': 0.01, 'batch_size': 64, 'dropout_p': 0.5, 'weight_decay': 1e-4, 'epochs': 10}, # Baseline
        {'lr': 0.02, 'batch_size': 64, 'dropout_p': 0.5, 'weight_decay': 1e-4, 'epochs': 10}, # High LR
        {'lr': 0.005, 'batch_size': 64, 'dropout_p': 0.5, 'weight_decay': 1e-4, 'epochs': 10}, # Low LR
        {'lr': 0.01, 'batch_size': 32, 'dropout_p': 0.5, 'weight_decay': 1e-4, 'epochs': 10}, # Small Batch
        {'lr': 0.01, 'batch_size': 128, 'dropout_p': 0.5, 'weight_decay': 1e-4, 'epochs': 10}, # Large Batch
        {'lr': 0.01, 'batch_size': 64, 'dropout_p': 0.25, 'weight_decay': 1e-4, 'epochs': 10}, # Low Dropout
        {'lr': 0.01, 'batch_size': 64, 'dropout_p': 0.0, 'weight_decay': 1e-4, 'epochs': 10}, # No Dropout
        {'lr': 0.01, 'batch_size': 64, 'dropout_p': 0.5, 'weight_decay': 0.0, 'epochs': 10}, # No WD
    ]
    
    results = []
    start_total_time = time.time()
    time_limit = 30 * 60 # 30 minutes
    
    print(f"Starting hyperparameter tuning (Time limit: {time_limit/60} mins)...")
    
    for i, config in enumerate(configs):
        if time.time() - start_total_time > time_limit:
            print("Time limit reached. Stopping tuning.")
            break
            
        print(f"--- Running Config {i+1}/{len(configs)} ---")
        acc, duration = train_one_config(config, X_train, Y_train, X_test, Y_test)
        
        result = config.copy()
        result['best_acc'] = acc
        result['duration'] = duration
        results.append(result)
        
    # Print Summary
    print("\n" + "="*80)
    print(f"{'LR':<8} | {'BS':<6} | {'Drop':<6} | {'WD':<8} | {'Epochs':<6} | {'Best Acc':<10} | {'Time(s)':<8}")
    print("-" * 80)
    
    # Sort by accuracy descending
    results.sort(key=lambda x: x['best_acc'], reverse=True)
    
    for r in results:
        print(f"{r['lr']:<8} | {r['batch_size']:<6} | {r['dropout_p']:<6} | {r['weight_decay']:<8} | {r['epochs']:<6} | {r['best_acc']:.4f}     | {r['duration']:.1f}")
    print("="*80)
    print(f"Best Configuration: LR={results[0]['lr']}, BS={results[0]['batch_size']}, Drop={results[0]['dropout_p']}")

if __name__ == "__main__":
    main()
