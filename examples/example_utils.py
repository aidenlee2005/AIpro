import numpy as np
import os
import pickle
import csv
import time
import sys

# Add framework directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../framework"))
from tensor import TensorFull as Tensor

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

# Ensure stdout is line-buffered so in-place progress bars flush promptly
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)


def render_progress(current_step, total_steps, prefix="", length=30, start_time=None):
    """Render an in-place ASCII progress bar with ETA.

    If stdout is not a TTY (e.g., VS Code output panel, file redirect), fall back to
    throttled newline logging to avoid 
 spam.
    """
    if total_steps <= 0:
        return

    is_tty = sys.stdout.isatty()

    # Throttle updates when not interactive to prevent huge logs with literal \r
    if not is_tty:
        step_every = max(total_steps // 20, 1)
        if current_step < total_steps and (current_step % step_every != 0):
            return

    ratio = min(max(current_step / total_steps, 0.0), 1.0)
    filled = int(length * ratio)
    bar = "#" * filled + "-" * (length - filled)
    percent = ratio * 100

    elapsed = time.time() - start_time if start_time else None
    eta = None
    if elapsed is not None and current_step > 0:
        eta = elapsed / current_step * max(total_steps - current_step, 0)

    parts = [f"{prefix} [{bar}] {percent:5.1f}%"]
    if elapsed is not None:
        parts.append(f"elapsed:{elapsed:5.1f}s")
    if eta is not None:
        parts.append(f"eta:{eta:5.1f}s")

    line = " | ".join(parts)

    if not is_tty:
        # Print newline-based updates in non-interactive sinks
        sys.stdout.write(f"{line}\n")
        sys.stdout.flush()
        return

    last_len = getattr(render_progress, "_last_len", 0)
    padding = max(last_len - len(line), 0)
    sys.stdout.write(f"\r{line}{' ' * padding}")
    sys.stdout.flush()
    render_progress._last_len = len(line)


def progress_wrapper(iterator, total_steps, prefix, start_time=None):
    """Yield (idx, item) while showing tqdm bar if available, else ASCII bar.

    idx starts at 1. Automatically closes bar and handles newline when needed.
    """
    use_tqdm = tqdm is not None and sys.stdout.isatty()
    if use_tqdm:
        with tqdm(total=total_steps, desc=prefix, leave=False, dynamic_ncols=True) as pbar:
            for idx, item in enumerate(iterator, start=1):
                yield idx, item
                pbar.update(1)
        return

    for idx, item in enumerate(iterator, start=1):
        render_progress(idx, total_steps, prefix=prefix, start_time=start_time)
        yield idx, item
    sys.stdout.write("\n")
    sys.stdout.flush()
    setattr(render_progress, "_last_len", 0)

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
        if not os.path.exists(f):
             # Try looking in cifar-10-batches-py subdir if root_dir is just data
             f = os.path.join(root_dir, 'cifar-10-batches-py', 'data_batch_%d' % (b, ))
        
        X, Y = load_cifar_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    
    test_batch = os.path.join(root_dir, 'test_batch')
    if not os.path.exists(test_batch):
        test_batch = os.path.join(root_dir, 'cifar-10-batches-py', 'test_batch')
        
    Xte, Yte = load_cifar_batch(test_batch)
    return Xtr, Ytr, Xte, Yte

class CIFAR10Dataset:
    def __init__(self, X, Y, transform=None, mode='train'):
        self.X = X
        self.Y = Y
        self.transform = transform
        self.mode = mode
        
        # Normalization stats
        self.mean = np.array([0.4914, 0.4822, 0.4465]).reshape(1, 3, 1, 1)
        self.std = np.array([0.2023, 0.1994, 0.2010]).reshape(1, 3, 1, 1)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        # X is (N, 3, 32, 32)
        img = self.X[idx] # (3, 32, 32)
        label = self.Y[idx]
        
        # Apply transforms if any (on numpy array)
        if self.transform and self.mode == 'train':
            img = self.transform(img)
            
        # Normalize
        img = img / 255.0
        img = (img - self.mean.reshape(3, 1, 1)) / self.std.reshape(3, 1, 1)
        
        return img.astype(np.float32), label

def random_crop(img, padding=4):
    # img: (3, 32, 32)
    c, h, w = img.shape
    padded = np.pad(img, ((0,0), (padding, padding), (padding, padding)), mode='constant')
    h_new, w_new = padded.shape[1], padded.shape[2]
    
    top = np.random.randint(0, h_new - h + 1)
    left = np.random.randint(0, w_new - w + 1)
    
    return padded[:, top:top+h, left:left+w]

def random_horizontal_flip(img, p=0.5):
    if np.random.rand() < p:
        return np.flip(img, axis=2) # Flip width
    return img

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

def get_batch(dataset, batch_size, device="gpu"):
    num_samples = len(dataset)
    indices = np.arange(num_samples)
    if dataset.mode == 'train':
        np.random.shuffle(indices)

    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_idx = indices[start_idx:end_idx]

        batch_x_list = []
        batch_y_list = []

        for idx in batch_idx:
            x, y = dataset[idx]
            batch_x_list.append(x)
            batch_y_list.append(y)

        batch_x = np.stack(batch_x_list)
        batch_y = np.array(batch_y_list)

        # Create tensor once and reuse cached data when possible
        yield Tensor(batch_x, device=device), Tensor(batch_y, device=device)

def save_model(model, model_name, accuracy, save_dir="models"):
    """
    Saves the model parameters to a file.
    Filename format: {model_name}_acc{accuracy:.4f}_{timestamp}.pkl
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{model_name}_acc{accuracy:.4f}_{timestamp}.pkl"
    filepath = os.path.join(save_dir, filename)
    
    # Extract parameters as numpy arrays
    params = [p.numpy() for p in model.parameters()]
    
    with open(filepath, 'wb') as f:
        pickle.dump(params, f)
        
    print(f"Model saved to {filepath}")

def train_model(model, train_dataset, test_dataset, config, model_name, script_name, augmentation, device="gpu"):
    """
    Generic training loop to reduce boilerplate in example scripts.
    """
    import optimizer as nn
    
    # Optimizer setup
    if config['optimizer'] == 'SGD':
        optimizer = nn.SGD(model.parameters(), lr=config['lr'], momentum=config['momentum'], weight_decay=config.get('weight_decay', 0.0))
    elif config['optimizer'] == 'Adam':
        optimizer = nn.Adam(model.parameters(), lr=config['lr'], weight_decay=config.get('weight_decay', 0.0))
    else:
        raise ValueError(f"Unsupported optimizer: {config['optimizer']}")
        
    criterion = nn.CrossEntropyLoss()

    # Training Loop
    total_start_time = time.time()
    total_batches = (len(train_dataset) + config['batch_size'] - 1) // config['batch_size']

    for epoch in range(config['epochs']):
        model.train()
        epoch_start = time.time()
        correct = 0
        total = 0
        loss_sum = 0
        step_start = time.time()

        for batch_idx, (batch_x, batch_y) in progress_wrapper(
            get_batch(train_dataset, config['batch_size'], device=device),
            total_batches,
            prefix=f"Epoch {epoch+1}/{config['epochs']}",
            start_time=step_start,
        ):
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            
            # Metrics
            pred = output.numpy().argmax(axis=1)
            correct += (pred == batch_y.numpy()).sum()
            total += batch_y.shape[0]
            loss_sum += loss.numpy()

        epoch_time = time.time() - epoch_start
        tr_acc = correct / total
        print(f"Epoch {epoch+1}/{config['epochs']} - Time: {epoch_time:.2f}s - Loss: {loss_sum/total:.4f} - Acc: {tr_acc:.4f}")

    total_time = time.time() - total_start_time
    
    # Evaluation
    model.eval()
    correct = 0
    total = 0
    for batch_x, batch_y in get_batch(test_dataset, config['batch_size'], device=device):
        output = model(batch_x)
        pred = output.numpy().argmax(axis=1)
        correct += (pred == batch_y.numpy()).sum()
        total += batch_y.shape[0]
    
    te_acc = correct / total
    print(f"Test Accuracy: {te_acc:.4f}")
    
    # PyTorch Benchmark
    from pytorch_utils import run_pytorch_benchmark
    try:
        pytorch_time = run_pytorch_benchmark(model_name, batch_size=config['batch_size'])
    except Exception as e:
        print(f"PyTorch benchmark failed: {e}")
        pytorch_time = None
        
    # Log
    logger = ExperimentLogger()
    logger.log(script_name, model_name, augmentation, config, tr_acc, te_acc, total_time, pytorch_time)
    
    # Save Model
    save_model(model, model_name, te_acc)

class ExperimentLogger:
    def __init__(self, filename="new_training_log.csv"):
        # Ensure log file is always in the project root
        if not os.path.isabs(filename):
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            self.filename = os.path.join(project_root, filename)
        else:
            self.filename = filename
            
        self.headers = [
            "Timestamp", "Script", "Model", "Augmentation", "Optimizer", "LR", "BatchSize", 
            "Momentum", "WeightDecay", "Epochs", 
            "Final_TrAcc", "Final_TeAcc", "Total_Time", "Avg_Epoch_Time", 
            "PyTorch_Ref_Epoch_Time", "Speedup(PyTorch/Ours)"
        ]
        self._init_csv()

    def _init_csv(self):
        if not os.path.exists(self.filename):
            with open(self.filename, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.headers)

    def log(self, script_name, model_name, augmentation, config, final_tr_acc, final_te_acc, total_time, pytorch_ref_epoch_time=None):
        epochs = config.get('epochs', 1)
        avg_epoch_time = total_time / epochs if epochs > 0 else 0
        
        ratio = "N/A"
        if pytorch_ref_epoch_time and pytorch_ref_epoch_time > 0:
            # Speedup: How much faster is PyTorch? or Ours?
            # Usually "Speedup" means Baseline / New. If PyTorch is baseline, and we are slower...
            # Let's just log PyTorch Time / Our Time. If > 1, PyTorch is slower (we are faster).
            # But usually custom frameworks are slower.
            # Let's log PyTorch / Ours.
            ratio = f"{pytorch_ref_epoch_time / avg_epoch_time:.4f}"

        row = [
            time.strftime("%Y-%m-%d %H:%M:%S"),
            script_name,
            model_name,
            augmentation,
            config.get('optimizer', 'SGD'),
            config.get('lr', 0),
            config.get('batch_size', 0),
            config.get('momentum', 0),
            config.get('weight_decay', 0),
            epochs,
            f"{final_tr_acc:.4f}",
            f"{final_te_acc:.4f}",
            f"{total_time:.2f}",
            f"{avg_epoch_time:.4f}",
            f"{pytorch_ref_epoch_time:.4f}" if pytorch_ref_epoch_time else "N/A",
            ratio
        ]
        
        with open(self.filename, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
        print(f"Logged experiment results to {self.filename}")
