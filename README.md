# Lutorpy

Lutorpy is a libray built for deep learning with torch in python,  by a two-way bridge between Python/Numpy and Lua/Torch, you can use any Torch modules(nn, rnn etc.) in python, and easily convert variables(array and tensor) between torch and numpy.

# Features

* import any lua/torch module to python and use it like python moduels
* use lua objects directly in python, conversion between python object and lua object are done automatically
* support zero-base indexing (lua uses 1-based indexing)
* automatic prepending self to function by "._" syntax, easily convert ":" operator in lua to python
* create torch tensor from numpy array with torch.fromNumpyArray(arr)
* use tensor.asNumpyarray() to create a numpy array which share the memory with torch tensor

# Installation
You need to install torch before you start
``` bash
# in a terminal, run the commands WITHOUT sudo
git clone https://github.com/torch/distro.git ~/torch --recursive
cd ~/torch; bash install-deps;
./install.sh
```
Then, you can use luarocks to install torch/lua modules
``` bash
luarocks install nn
```
If you don't have numpy installed, install it by pip
``` bash
pip install numpy
```
Now, we are ready to install lutorpy
``` bash
git clone https://github.com/imodpasteur/lutorpy.git
cd lutorpy
sudo python setup.py install
```
#### note that it has been tested on ubuntu, please report issue if you encountered error.
# Quick Start

## basic usage
``` python
## boot strap lutorpy
import lutorpy as lua
lua.set_globals(globals(), __builtins__)
## enable zero-based index
lua.LuaRuntime(zero_based_index=True)

## use require("MODULE") to import lua modules
require("nn")

## run lua code in python with minimal modification:  replace ":" to "._"
t = torch.DoubleTensor(10,3)
print(t._size()) # corresponding lua version is t:size()
# or you can pass 'self' manually
print(t.size(t))

## use zero-based index
t[0][1] = 24
print(t[0][1])

## convert torch tensor to numpy array
### Note: the underlying object are sharing the same memory, so the conversion is instant
arr = t.asNumpyArray()
print(arr.shape)
```
## example: multi-layer perception
``` python
## minimal example of multi-layer perception(without training code)
mlp = nn.Sequential()
mlp._add(nn.Linear(100, 30))
mlp._add(nn.Tanh())
mlp._add(nn.Linear(30, 10))

## generate a numpy array and convert it to torch tensor
import numpy as np
xn = np.random.randn(100)
xt = torch.fromNumpyArray(xn)
## process with the neural network
yt = mlp._forward(xt)
print(yt)

## or for example, you can plot your result with matplotlib
yn = yt.asNumpyArray()
import matplotlib.pyplot as plt
plt.plot(yn)

## cheers! you get the hang of lutorpy.
## you can have a look at the step-by-step tutorial and more complete example.
```


# Step-by-step tutorial

## import lutorpy and bootstrap globals

Note: the following setup is mandatory for going through this tutorial.

``` python
import lutorpy as lua
# setup runtime and use zero-based index
lua.LuaRuntime(zero_based_index=True)
# set the python globals() and __builtins__ to lua,
# so all the lua global variables can be seen in python globals()
lua.set_globals(globals(), __builtins__)
```
## hello world

``` python
lua.execute(' greeting = "hello world" ')
print(greeting)
```

### Alternative way to use lua
if you don't want to mess the python global variables, you can skip the previous line, but you need to access lua global variables through lua.globals(). 

Note that if you do this, all the following code should change acorrdingly.

```
import lutorpy as lua
lg = lua.globals()
lua.execute(' greeting = "hello world" ')
print(lg.greeting)
# without set_globals you have to use lua.require instead of require
lua.require("torch")
```
###  Alternatively you could also switch back to one-based indexing

Note that if you do this, all the following code should change acorrdingly.

```
lua.LuaRuntime(zero_based_index=False)
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
                                
## convert numpy array to torch tensor

Note: both torch tensor and cuda tensor are supported.

``` python
arr = np.random.randn(100)
print(arr)
t = torch.fromNumpyArray(arr)
print(t)

```

## convert to/from cudaTensor
``` python
require('cutorch')
t = torch.randn(10,10)
cudat = t._cuda()
#convert cudaTensor to floatTensor before convert to numpy array
arr = cudat._float().asNumpyArray()
print(arr)

arr = np.random.randn(100)
print(arr)
t = torch.fromNumpyArray(arr)
cudat = t._cuda()
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
In lua, we use syntax like 'mlp:add(module)' to use a function without pass self to the function. But in python, it's done by default, there are two ways to prepend 'self' to a lua function in lutorpy.

The first way is inline prepending by add '_' to before any function name, then it will try to return a prepended version of the function:
``` python
mlp = nn.Sequential()
module = nn.Linear(10, 5)

# lua style
mlp.add(mlp, module)

# inline prepending
mlp._add(module) # equaliant to mlp:add(module) in lua
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

# More usage and details
Lutorpy is built upon [lupa](https://github.com/scoder/lupa), there are more features provided by lupa could be also useful, please check it out.

# Acknowledge

This is a project inspired by [lunatic-python](https://github.com/bastibe/lunatic-python) and [lupa](https://github.com/scoder/lupa), and it's based on lupa.
