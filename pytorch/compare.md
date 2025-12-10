**单卡模式**
Using device: cuda
Start training: epochs=10, batch_size=128, lr=0.01
Epoch 1: 100%|████████████████| 391/391 [00:02<00:00, 140.80it/s]
[Epoch 1/10] loss=1.9896 time=2.78s throughput=17988.5 imgs/s
Epoch 2: 100%|████████████████| 391/391 [00:02<00:00, 163.49it/s]
[Epoch 2/10] loss=1.5140 time=2.39s throughput=20897.2 imgs/s
Epoch 3: 100%|████████████████| 391/391 [00:02<00:00, 158.23it/s]
[Epoch 3/10] loss=1.3468 time=2.47s throughput=20222.8 imgs/s
Epoch 4: 100%|████████████████| 391/391 [00:02<00:00, 159.40it/s]
[Epoch 4/10] loss=1.2343 time=2.45s throughput=20372.3 imgs/s
Epoch 5: 100%|████████████████| 391/391 [00:02<00:00, 157.09it/s]
[Epoch 5/10] loss=1.1455 time=2.49s throughput=20076.7 imgs/s
Epoch 6: 100%|████████████████| 391/391 [00:02<00:00, 158.02it/s]
[Epoch 6/10] loss=1.0767 time=2.48s throughput=20192.4 imgs/s
Epoch 7: 100%|████████████████| 391/391 [00:02<00:00, 156.23it/s]
[Epoch 7/10] loss=1.0191 time=2.50s throughput=19969.5 imgs/s
Epoch 8: 100%|████████████████| 391/391 [00:02<00:00, 160.10it/s]
[Epoch 8/10] loss=0.9667 time=2.44s throughput=20461.7 imgs/s
Epoch 9: 100%|████████████████| 391/391 [00:02<00:00, 160.14it/s]
[Epoch 9/10] loss=0.9196 time=2.44s throughput=20466.3 imgs/s
Epoch 10: 100%|███████████████| 391/391 [00:02<00:00, 160.07it/s]
[Epoch 10/10] loss=0.8712 time=2.44s throughput=20457.0 imgs/s
Testing: 100%|███████████████████| 79/79 [00:00<00:00, 91.67it/s]
Test accuracy: 63.16%
Class 0: 59.10%
Class 1: 76.90%
Class 2: 49.10%
Class 3: 39.40%
Class 4: 64.60%
Class 5: 51.80%
Class 6: 63.80%
Class 7: 73.10%
Class 8: 72.80%
Class 9: 81.00%

**多卡模式**
[Gloo] Rank 0 is connected to 1[Gloo] Rank  peer ranks. Expected number of connected peer ranks is : 11 is connected to 1
 peer ranks. Expected number of connected peer ranks is : 1
DDP start | world_size=2, batch=128, epochs=10
[Epoch 1/10] loss=2.2042 time=2.35s throughput=21258.9 imgs/s (global)
[Epoch 2/10] loss=1.7321 time=1.64s throughput=30548.3 imgs/s (global)
[Epoch 3/10] loss=1.5177 time=1.55s throughput=32292.3 imgs/s (global)
[Epoch 4/10] loss=1.4158 time=1.53s throughput=32641.1 imgs/s (global)
[Epoch 5/10] loss=1.3133 time=1.66s throughput=30144.6 imgs/s (global)
[Epoch 6/10] loss=1.2393 time=1.48s throughput=33699.5 imgs/s (global)
[Epoch 7/10] loss=1.1721 time=1.51s throughput=33204.1 imgs/s (global)
[Epoch 8/10] loss=1.1178 time=1.52s throughput=32864.6 imgs/s (global)
[Epoch 9/10] loss=1.0847 time=1.71s throughput=29277.0 imgs/s (global)
[Epoch 10/10] loss=1.0351 time=1.70s throughput=29360.3 imgs/s (global)
Test accuracy: 61.12%
Class 0: 76.30%
Class 1: 70.30%
Class 2: 38.20%
Class 3: 24.40%
Class 4: 64.80%
Class 5: 53.60%
Class 6: 73.50%
Class 7: 68.40%
Class 8: 67.40%
Class 9: 74.30%
Saved weights to ./pytorch/cifar_net_ddp.pth