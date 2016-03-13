# Lutorpy

Lutorpy is a two-way bridge between Python/Numpy and Lua/Torch, allowing use using Torch packages(nn, rnn etc.) with numpy inside python.

Support pythonic features such as zero-base indexing and automatic prepending self to function.

This library is based on [lupa](https://github.com/scoder/lupa), please refer to [lupa](https://github.com/scoder/lupa) for more detailed usage.

# Installation
``` bash

git clone https://github.com/oeway/lutorpy.git
cd lutorpy
python setup.py install     # use sudo if needed

```


# Getting Start

## import lutorpy and bootstrap globals
``` python
import lutorpy as lua
# set the python globals() and __builtins__ to lua,
# so all the lua global variables can be seen in python globals()
lua.set_globals(globals(), __builtins__)
```
## hello world

``` python
lua.execute(' greeting = "hello world" ')
print(greeting)
```

### Note: alternative way to use lua
if you don't want to mess the python global variables, you can skip the previous line, but you need to access lua global variables through lua.globals(). 

Note that if you do this, all the following code should change acorrdingly.

``` python
import lutorpy as lua
lg = lua.globals()
lua.execute(' greeting = "hello world" ')
print(lg.greeting)
# without set_globals you have to use lua.require instead of require
lua.require("torch")
```
### zero-based indexing or one-based indexing
```python
# by default, it's zero-based indexing
lua.LuaRuntime(zero_based_index=True)
b = lua.eval(' {"one", "zero"} ') 
print(b[1])

# you could switch to one-based indexing
lua.LuaRuntime(zero_based_index=False)
b = lua.eval(' {"one", "zero"} ') 
print(b[1])
```

## execute lua code

``` python
a = lua.eval(' {11, 22} ') # define a lua table with two elements
print(a[0])

lua.execute(' b={33, 44} ') # define another lua table with two elements
print(b[0])
print(b[1])

```

## use torch
``` python
require("torch")
z = torch.Tensor(4,5,6,2)
print(torch.isTensor(z))

s = torch.LongStorage(6)
print(torch.type(s))
```

## convert torch tensor to numpy array


``` python
require('torch')

t = torch.randn(10,10)
print(t)
arr = t.asNumpyArray()
print(arr)

```

### use cudaTensor
``` python
require('cutorch')
t = torch.randn(10,10)
cudat = t.cuda(t)
arr = cudat.asNumpyArray()
print(arr)
```
                                
## convert/copy numpy array to torch tensor

Note: both torch tensor and cuda tensor are supported

``` python
arr = np.random.randn(100)
print(arr)
t = lua.array2tensor(arr)
print(t)

t2 = torch.Tensor(10,10)
t2.copyNumpyArray(arr)
print(t2)

# use cudaTensor
t3 = torch.CudaTensor(10,10)
t3.copyNumpyArray(arr)
print(t3)

# or, convert torch tensor to cuda tensor
require('cutorch')
cudat = t.cuda(t)
print(cudat)

```

## load image and use nn module
``` python
require("image")
img_rgb = image.lena()
print(img_rgb.size(img_rgb))
img = image.rgb2y(img_rgb)
print(img.size(img))

# use SpatialConvolution from nn to process the image
require("nn")
n = nn.SpatialConvolution(1,16,12,12)
res = n.forward(n, img)
print(res.size(res))

```

## build a simple model

``` python
mlp = nn.Sequential()

module = nn.Linear(10, 5)
mlp.add(mlp, module)

print(module.weight)
print(module.bias)

print(module.gradWeight)
print(module.gradBias)

x = torch.Tensor(10) #10 inputs

# pass self to the function
y = mlp.forward(mlp, x)
print(y)

```

## prepending 'self' as the first argument automatically
In lua, there are use 'mlp:add(module)' to use function without pass self to the function. In python, it's done by default, There are two ways to prepend 'self' to a lua function.

The first way is inline prepending by add '_' to before any function name, then it will try to return a prepended version of the function:
``` python
mlp = nn.Sequential()
module = nn.Linear(10, 5)

# lua style
mlp.add(mlp, module)

# inline prepending
mlp._add(module) # equaliant to mlp:add(module) in lua
```

The second way is using lua.bs to bootstrap the function.

``` python
mlp = nn.Sequential()
module = nn.Linear(10, 5)
# bootstrap the add function
lua.bs(mlp,'add')
# now we can use add without passing self as the first arugment
mlp.add(module)
```

## build another model and training it

Train a model to perform XOR operation (see [this torch tutorial](https://github.com/torch/nn/blob/master/doc/training.md)).

``` python
require("nn")
mlp = nn.Sequential()
mlp._add(nn.Linear(2, 20)) # 2 input nodes, 20 hidden nodes
mlp._add(nn.Tanh())
mlp._add(nn.Linear(20, 1)) # 1 output nodes

criterion = nn.MSECriterion() 

for i in range(2500):
    # random sample
    input= torch.randn(2)    # normally distributed example in 2d
    output= torch.Tensor(1)
    if input[0]*input[1] > 0:  # calculate label for XOR function
        output[0] = -1 # output[0] = -1
    else:
        output[0] = 1 # output[0] = 1
    
    # feed it to the neural network and the criterion
    criterion._forward(mlp._forward(input), output)

    # train over this example in 3 steps
    # (1) zero the accumulation of the gradients
    mlp._zeroGradParameters()
    # (2) accumulate gradients
    mlp._backward(input, criterion.backward(criterion, mlp.output, output))
    # (3) update parameters with a 0.01 learning rate
    mlp._updateParameters(0.01)

```
## test the model

``` python

x = torch.Tensor(2)
x[0] =  0.5; x[1] =  0.5; print(mlp._forward(x))
x[0] =  0.5; x[1] = -0.5; print(mlp._forward(x))
x[0] = -0.5; x[1] =  0.5; print(mlp._forward(x))
x[0] = -0.5; x[1] = -0.5; print(mlp._forward(x))

```

# Acknowledge

This is a project inspired by [lunatic-python](https://github.com/bastibe/lunatic-python) and [lupa](https://github.com/scoder/lupa), and it's based on lupa.
