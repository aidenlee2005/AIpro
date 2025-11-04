import torch
import torch.nn.functional as F
import numpy as np

def Test1():
    input = torch.tensor(
        [[ 0.9, 0.8, 0.5 ],
        [ 0.8, 0.2, 0.6 ],
        [ 0.1, 0.4, 0.4 ],
        [ 0.5, 0.5, 0.2 ],
        [ 0.2, 0.8, 0.8 ]],
    dtype=torch.float32, requires_grad=True)

    weight = torch.tensor(
        [[ 0.9, 0.8 ],
        [ 0.5, 0.8 ],
        [ 0.2, 0.6 ]],
    dtype=torch.float32, requires_grad=True)

    bias = torch.tensor(
        [0.9, 0.8],
    dtype=torch.float32, requires_grad=True)

    output = F.linear(input, weight.t(), bias)

    grad_output = torch.tensor(
        [[ 0.9, 0.8 ],
        [ 0.5, 0.8 ],
        [ 0.2, 0.6 ],
        [ 0.1, 0.4 ],
        [ 0.4, 0.5 ]],
    dtype=torch.float32)

    output.backward(grad_output)

    #id0
    print(output.detach().numpy().round(2))
    print('#')
    
    #id1
    print(input.grad.numpy().round(2))
    print('#')
    
    #id2
    print(weight.grad.numpy().round(2))
    print('#')
    
    #id3
    print(bias.grad.numpy().round(2))
    print('#')

#Test 2

def Test2():
    input_data = torch.tensor(
    [[[[0.9, 0.8, 0.5, 0.8],
    [0.2, 0.6, 0.1, 0.4],
    [0.4, 0.5, 0.5, 0.2],
    [0.2, 0.8, 0.8, 0.3]],
    [[0.5, 0.1, 0.6, 0.1],
    [0.2, 0.9, 0.7, 0.3],
    [0.0, 0.7, 0.7, 0.2],
    [0.0, 0.5, 0.9, 0.8]],
    [[0.2, 0.3, 0.2, 0.0],
    [0.8, 0.7, 0.6, 0.8],
    [0.0, 0.7, 0.2, 0.4],
    [0.7, 0.1, 0.8, 0.9]]]],
    dtype=torch.float32, requires_grad=True)

    weight_data = torch.tensor(
    [[[[0.9, 0.8, 0.5],
    [0.8, 0.2, 0.6],
    [0.1, 0.4, 0.4]],
    [[0.5, 0.5, 0.2],
    [0.2, 0.8, 0.8],
    [0.3, 0.5, 0.1]],
    [[0.6, 0.1, 0.2],
    [0.9, 0.7, 0.3],
    [0.0, 0.7, 0.7]]],
    [[[0.2, 0.0, 0.5],
    [0.9, 0.8, 0.2],
    [0.3, 0.2, 0.0]],
    [[0.8, 0.7, 0.6],
    [0.8, 0.0, 0.7],
    [0.2, 0.4, 0.7]],
    [[0.1, 0.8, 0.9],
    [0.8, 0.3, 0.8],
    [0.2, 0.9, 0.6]]]],
    dtype=torch.float32, requires_grad=True)

    output = F.conv2d(input_data, weight_data, stride=1, padding=1)
    #id4
    print(output.detach().numpy().round(2))
    print('#')
    
    grad_output = torch.tensor(
    [[[[0.9, 0.8, 0.5, 0.8],
    [0.2, 0.6, 0.1, 0.4],
    [0.4, 0.5, 0.5, 0.2],
    [0.2, 0.8, 0.8, 0.3]],
    [[0.5, 0.1, 0.6, 0.1],
    [0.2, 0.9, 0.7, 0.3],
    [0.0, 0.7, 0.7, 0.2],
    [0.0, 0.5, 0.9, 0.8]]]],
    dtype=torch.float32)

    output.backward(grad_output)
    
    #id5
    print(input_data.grad.detach().numpy().round(2))
    print('#')
    
    #id6
    print(weight_data.grad.detach().numpy().round(2))
    print('#')
    
def Test3():
    input = torch.tensor([[[[ 0.9, 0.8, 0.5, 0.8 ],
                        [ 0.2, 0.6, 0.1, 0.4 ],
                        [ 0.4, 0.5, 0.5, 0.2 ],
                        [ 0.2, 0.8, 0.8, 0.3 ]],
                       [[ 0.5, 0.1, 0.6, 0.1 ],
                        [ 0.2, 0.9, 0.7, 0.3 ],
                        [ 0, 0.7, 0.7, 0.2 ],
                        [ 0, 0.5, 0.9, 0.8 ]],
                       [[ 0.2, 0.3, 0.2, 0 ],
                        [ 0.8, 0.7, 0.6, 0.8 ],
                        [ 0, 0.7, 0.2, 0.4 ],
                        [ 0.7, 0.1, 0.8, 0.9 ]]]], dtype=torch.float32, requires_grad=True)

    # 前向 maxpool
    output, indices = torch.nn.functional.max_pool2d(input, kernel_size=2, stride=2, return_indices=True)
    
    #id7
    print(output.detach().numpy().round(2))
    print('#')

    grad_output = torch.tensor([[[[ 0.9, 0.8 ],
                                  [ 0.5, 0.8 ]],
                                 [[ 0.2, 0.6 ],
                                  [ 0.1, 0.4 ]],
                                 [[ 0.4, 0.5 ],
                                  [ 0.5, 0.2 ]]]], dtype=torch.float32)
    output.backward(grad_output)
    
    #id8
    print(input.grad.detach().numpy().round(2))
    print('#')
    
def Test4():
    input = np.array([
        [0.9, 0.8, 0.5, 0.8],
        [0.2, 0.6, 0.1, 0.4],
        [0.4, 0.5, 0.5, 0.2]
    ], dtype=np.float32)
    target = np.array([3, 3, 1], dtype=np.int64)
    x = torch.tensor(input, requires_grad=True)
    y = torch.tensor(target)

    softmax_out = F.softmax(x, dim=1)
    
    #id9
    print(softmax_out.detach().numpy())
    print('#')
    
    loss = F.cross_entropy(x, y, reduction='mean')
    
    #id10
    print(loss.item())
    print('#')

    loss.backward()
    
    #id11
    print(x.grad.detach().numpy())
    print('#')
    
Test1()
Test2()
Test3()
Test4()


    
