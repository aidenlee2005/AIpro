import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Regex patterns to parse logs
LOSS_PATTERN = re.compile(r"\[Epoch (\d+)/(\d+)\] loss=([0-9.]+) time=([0-9.]+)s throughput=([0-9.]+)")
ACC_PATTERN = re.compile(r"Test accuracy: ([0-9.]+)%")


def run_cmd(cmd, cwd):
    print(f"Running: {' '.join(cmd)}")
    res = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if res.returncode != 0:
        print(res.stdout)
        raise RuntimeError(f"Command failed with code {res.returncode}: {' '.join(cmd)}")
    return res.stdout


def parse_metrics(log_text):
    epochs = []
    losses = []
    times = []
    thrpts = []
    for m in LOSS_PATTERN.finditer(log_text):
        ep = int(m.group(1))
        loss = float(m.group(3))
        t = float(m.group(4))
        th = float(m.group(5))
        epochs.append(ep)
        losses.append(loss)
        times.append(t)
        thrpts.append(th)
    acc = None
    m_acc = ACC_PATTERN.search(log_text)
    if m_acc:
        acc = float(m_acc.group(1))
    return epochs, losses, times, thrpts, acc


def plot_curves(out_dir, single, ddp):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Loss curve
    plt.figure()
    plt.plot(single['epochs'], single['losses'], label='single')
    plt.plot(ddp['epochs'], ddp['losses'], label='ddp')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / 'loss_compare.png')

    # Throughput curve
    plt.figure()
    plt.plot(single['epochs'], single['thrpts'], label='single imgs/s')
    plt.plot(ddp['epochs'], ddp['thrpts'], label='ddp imgs/s')
    plt.xlabel('epoch')
    plt.ylabel('imgs/s')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / 'throughput_compare.png')


def write_summary(out_dir, single, ddp):
    out = Path(out_dir) / 'benchmark_summary.txt'
    lines = []
    lines.append('Single GPU:')
    lines.append(f"  epochs: {len(single['epochs'])}, acc: {single['acc']}")
    lines.append(f"  avg_epoch_time: {sum(single['times'])/len(single['times']):.4f}s")
    lines.append(f"  avg_throughput: {sum(single['thrpts'])/len(single['thrpts']):.1f} imgs/s")
    lines.append('DDP (multi-GPU):')
    lines.append(f"  epochs: {len(ddp['epochs'])}, acc: {ddp['acc']}")
    lines.append(f"  avg_epoch_time: {sum(ddp['times'])/len(ddp['times']):.4f}s")
    lines.append(f"  avg_throughput: {sum(ddp['thrpts'])/len(ddp['thrpts']):.1f} imgs/s")
    out.write_text('\n'.join(lines))
    print(out.read_text())


def main():
    parser = argparse.ArgumentParser(description='Run single and multi-GPU benchmarks and plot comparison.')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--backend', type=str, default='gloo', choices=['gloo', 'nccl'])
    parser.add_argument('--nproc', type=int, default=2, help='GPU count for DDP')
    parser.add_argument('--out', type=str, default='benchmark_out')
    args = parser.parse_args()

    cwd = Path(__file__).parent

    common = [f'--epochs={args.epochs}', f'--batch-size={args.batch_size}', f'--lr={args.lr}',
              f'--momentum={args.momentum}', f'--num-workers={args.num_workers}']

    # Single GPU run
    single_cmd = [sys.executable, 'lab0.py', *common]
    single_log = run_cmd(single_cmd, cwd)
    s_ep, s_loss, s_time, s_thr, s_acc = parse_metrics(single_log)

    # Multi GPU run (DDP)
    ddp_cmd = ['torchrun', f'--nproc_per_node={args.nproc}', 'lab0_ddp.py', f'--backend={args.backend}', *common]
    ddp_log = run_cmd(ddp_cmd, cwd)
    d_ep, d_loss, d_time, d_thr, d_acc = parse_metrics(ddp_log)

    single = {'epochs': s_ep, 'losses': s_loss, 'times': s_time, 'thrpts': s_thr, 'acc': s_acc}
    ddp = {'epochs': d_ep, 'losses': d_loss, 'times': d_time, 'thrpts': d_thr, 'acc': d_acc}

    plot_curves(args.out, single, ddp)
    write_summary(args.out, single, ddp)

    # Save raw logs
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / 'single.log').write_text(single_log)
    (Path(args.out) / 'ddp.log').write_text(ddp_log)


if __name__ == '__main__':
    main()
