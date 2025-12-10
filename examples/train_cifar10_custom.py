import numpy as np
import pickle
import sys
import os
import time

# Add framework directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../framework"))

from tensor_hm6 import TensorFull as Tensor
import optimizer_hm6 as nn
import operators_hm6 as F

def load_cifar_batch(filename):
    with open(filename, 'rb') as f:
        datadict = pickle.load(f, encoding='bytes')
        X = datadict[b'data']
        Y = datadict[b'labels']
        X = X.reshape(10000, 3, 32, 32).astype("float32")
        Y = np.array(Y, dtype="float32") # Backend expects float tensor for labels
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
        # Mean/Std normalization (optional but recommended)
        mean = np.array([0.4914, 0.4822, 0.4465]).reshape(1, 3, 1, 1)
        std = np.array([0.2023, 0.1994, 0.2010]).reshape(1, 3, 1, 1)
        self.X = (self.X - mean) / std

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        # Return numpy arrays, convert to Tensor in DataLoader or loop
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

def train():
    print("Loading CIFAR-10 data...")
    data_dir = os.path.join(os.path.dirname(__file__), "../data/cifar-10-batches-py")
    X_train, Y_train, X_test, Y_test = load_cifar10(data_dir)
    
    print(f"Train data: {X_train.shape}, {Y_train.shape}")
    print(f"Test data: {X_test.shape}, {Y_test.shape}")
    
    device = "gpu"
    
    # Pre-process data (normalization)
    mean = np.array([0.4914, 0.4822, 0.4465]).reshape(1, 3, 1, 1).astype(np.float32)
    std = np.array([0.2023, 0.1994, 0.2010]).reshape(1, 3, 1, 1).astype(np.float32)
    
    X_train = (X_train / 255.0 - mean) / std
    X_test = (X_test / 255.0 - mean) / std
    
    # Define Model
    # Simple CNN
    # Conv(3->16, 3x3) -> ReLU -> MaxPool(2x2) -> Conv(16->32, 3x3) -> ReLU -> MaxPool(2x2) -> Flatten -> Linear(32*8*8 -> 10)
    
    model = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, device=device),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2), # 32 -> 16
        nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, device=device),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2), # 16 -> 8
        nn.Flatten(),
        nn.Linear(32 * 8 * 8, 10, device=device)
    )
    
    criterion = nn.CrossEntropyLoss()
    optimizer = nn.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    # optimizer = nn.Adam(model.parameters(), lr=0.001)
    
    epochs = 5
    batch_size = 64
    
    print("Start training...")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        start_time = time.time()
        
        # Shuffle data
        indices = np.arange(X_train.shape[0])
        np.random.shuffle(indices)
        X_train_shuffled = X_train[indices]
        Y_train_shuffled = Y_train[indices]
        
        num_batches = int(np.ceil(X_train.shape[0] / batch_size))
        
        for i in range(num_batches):
            start = i * batch_size
            end = min(start + batch_size, X_train.shape[0])
            
            bx_np = X_train_shuffled[start:end]
            by_np = Y_train_shuffled[start:end]
            
            bx = Tensor(bx_np, device=device, requires_grad=False)
            by = Tensor(by_np, device=device, requires_grad=False)
            
            optimizer.zero_grad()
            
            logits = model(bx)
            loss = criterion(logits, by)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.numpy().item() * (end - start)
            
            # Calculate accuracy
            preds = np.argmax(logits.numpy(), axis=1)
            correct += (preds == by_np).sum()
            total += (end - start)
            
            if i % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {i}/{num_batches}, Loss: {loss.numpy().item():.4f}")
                
        avg_loss = total_loss / total
        acc = correct / total
        duration = time.time() - start_time
        print(f"Epoch {epoch+1} Done. Avg Loss: {avg_loss:.4f}, Acc: {acc:.4f}, Time: {duration:.2f}s")
        
        # Evaluation
        model.eval()
        test_correct = 0
        test_total = 0
        
        # Process test in batches to avoid OOM
        num_test_batches = int(np.ceil(X_test.shape[0] / batch_size))
        for i in range(num_test_batches):
            start = i * batch_size
            end = min(start + batch_size, X_test.shape[0])
            
            bx_np = X_test[start:end]
            by_np = Y_test[start:end]
            
            bx = Tensor(bx_np, device=device, requires_grad=False)
            
            logits = model(bx)
            preds = np.argmax(logits.numpy(), axis=1)
            
            test_correct += (preds == by_np).sum()
            test_total += (end - start)
            
        test_acc = test_correct / test_total
        print(f"Test Accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    train()
