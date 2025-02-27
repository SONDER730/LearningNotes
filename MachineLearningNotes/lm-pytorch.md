title: lm_pytorch
date: 2024-06-29 21:14:37
tags: LLM-NOTE



# 线性神经网络

##  向量链式法则

- ### 标量链式法则

[![image-20220107231957396](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/07/image-01)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/07/image-01)

- ### 拓展到向量

  > 需要注意维数的变化
  >
  > 下图三种情况分别对应：
  >
  > 1. y为标量，x为向量
  > 2. y为标量，x为矩阵
  > 3. y、x为矩阵

[![image-20220107231931153](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/07/image-02)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/07/image-02)

------

##### 例1（标量对向量求导）



> 这里应该是用分子布局，所以是X转置

 [![image-20220107233527373](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/07/image-03)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/07/image-03)

##### 例2（涉及到矩阵的情况）



> X是mxn的矩阵,w为n维向量，y为m维向量； z对Xw-y做L2 norm,为标量； 过程与例一大体一致；

 [![image-20220107233753066](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/07/image-04)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/07/image-04)

------

> 由于在神经网络动辄几百层，手动进行链式求导是很困难的，因此我们需要借助自动求导

------

### 自动求导

- 含义：计算一个函数在指定值上的导数

- 自动求导有别于

  - 符号求导

    ![image-20220107235547201](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/07/image-05)

  - 数值求导

    ![image-20220107235611767](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/07/image-06)

为了更好地理解自动求导，下面引入计算图的概念

#### 计算图

- 将代码分解成操作子

- 将计算表示成一个**无环图**

  > 下图自底向上其实就类似于链式求导过程

[![image-20220107235918270](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/07/image-07)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/07/image-07)



- 计算图有两种构造方式

  - 显示构造

    > 可以理解为先定义公式再代值
    >
    > Tensorflow/Theano/MXNet

     [![image-20220108000740736](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/07/image-08)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/07/image-08)

  - 隐式构造

    > 系统将所有的计算记录下来
    >
    > Pytorch/MXNet

     [![image-20220108001154208](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/07/image-09)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/07/image-09)

#### 自动求导的两种模式

- 正向累积

   [![image-20220108001506326](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/07/image-10)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/07/image-10)

- 反向累积（反向传递back propagation）

   [![image-20220108001517029](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/07/image-11)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/07/image-11)

 **反向累积计算过程**

[![image-20220108001847175](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/07/image-12)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/07/image-12)

> 反向累积的正向过程：自底向上，需要存储中间结果
>
> 反向累积的反向过程：自顶向下，可以去除不需要的枝（图中的x应为w）
>
>  [![image-20220108002228166](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/07/image-13)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/07/image-13)

#### 复杂度比较

- 反向累积

  - 时间复杂度：O(n),n是操作子数
    - 通常正向和反向的代价类似
  - 空间复杂度：O(n)
    - 存储正向过程所有的中间结果

- 正向累积

  > 每次计算一个变量的梯度时都需要将所有节点扫一遍

  - 时间复杂度：O(n)
  - 空间复杂度：O(1)

### 代码部分

```
#对y = x.Tx关于列向量x求导
import torch

x = torch.arange(4.0)
x
```

```
tensor([0., 1., 2., 3.])
```



```
#存储梯度
x.requires_grad_(True)#等价于x = torch.arange(4.0,requires_grad=True)
x.grad#默认值是None
```

```
y = torch.dot(x,x)
y
#PyTorch隐式地构造计算图，grad_fn用于记录梯度计算
```

```
tensor(14., grad_fn=<DotBackward0>)
```

**通过调用反向传播函数来自动计算y关于x每个分量的梯度**

```
y.backward()
x.grad
```

```
tensor([0., 2., 4., 6.])
```



```
x.grad==2*x#验证
```

```
tensor([True, True, True, True])
```



```
# 在默认情况下，PyTorch会累积梯度，我们需要清除之前的值
x.grad.zero_()#如果没有这一步结果就会加累上之前的梯度值，变为[1,5,9,13]
y = x.sum()
y.backward()
x.grad
```

```
tensor([1., 1., 1., 1.])
```



```
x.grad.zero_()
y=x*x#哈达玛积，对应元素相乘

#在深度学习中我们一般不计算微分矩阵
#而是计算批量中每个样本单独计算的偏导数之和

y.sum().backward()#等价于y.backword(torch.ones(len(x)))
x.grad
```

```
tensor([0., 2., 4., 6.])
```



**将某些计算移动到记录的计算图之外**

```
# 后可用于用于将神经网络的一些参数固定住
x.grad.zero_()
y = x*x
u = y.detach()#把y当作常数
z = u*x

z.sum().backward()
x.grad == u
```

```
tensor([True, True, True, True])
```



```
x.grad.zero_()
y.sum().backward()
x.grad == 2*x
```

```
tensor([True, True, True, True])
```

**即使构建函数的计算图需要用过Python控制流，仍然可以计算得到的变量的梯度**

**这也是隐式构造的优势，因为它会存储梯度计算的计算图，再次计算时执行反向过程就可以**

```
def f(a):
    b = a * 2
    while b.norm()<1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c

a = torch.randn(size=(),requires_grad=True)
d = f(a)
d.backward()

a.grad == d / a
```

## 线性回归+基础优化算法

### 线性回归

- **线性模型**

  - 输入：$x=[x_1,x_2,...,x_n]^T$

  - 线性模型需要确定一个n维权重和一个标量偏差$\omega=[\omega_1,\omega_2,...,\omega_n]^T,b$

  - 输出 ：$y=\omega_1x_1+\omega_2x_2+...+\omega_nx_n+b$，

    向量版本的是 𝑦=<𝜔,𝑥>+𝑏

  - 线性模型可以看作是单层神经网络

    > - 神经网络源于神经科学
    > - 最早的神经网络是源自神经科学的，但是时至今日，很多神经网络已经远远高于神经科学，可解释性也不是很强，不必纠结

- 衡量估计质量

  - 我们需要估计模型的预估值和真实值之间的差距，例如房屋售价和股价

  - 假设$y$是真实值，$\tilde{y}$是估计值，我们可以比较

    𝑙(𝑦,$\tilde{y}$)=(1/2)*(𝑦−$\tilde{y}$)2，这个叫做**平方损失**

- **训练数据**

  - 收集一些数据点来决定参数值（权重$\omega$和偏差$b$），例如6个月内被卖掉的房子。

  - 这被称之为训练数据

  - 通常越多越好。需要注意的是，现实世界的数据都是有限的，但是为了训练出精确的参数往往需要训练数据越多越好，当训练数据不足的时候，我们还需要进行额外处理。

  - 假设我们有n个样本，记为

    𝑋=[𝑥1,𝑥2,...,𝑥𝑛]𝑇,𝑦=[𝑦1,𝑦2,...𝑦𝑛]𝑇

    𝑋的每一行是一个样本，$y$的每一行是一个输出的实数值。

- **参数学习**

  - **训练损失**。但我们训练参数的时候，需要定义一个损失函数来衡量参数的好坏，应用前文提过的平方损失有公式：

  - ![image-20240701223140819](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20240701223140819.png)

     𝑙(𝑋,𝑥,𝜔,𝑏)=(1/2𝑛)*∑(𝑦𝑖−$\tilde{y}$)^2=(1/2𝑛)*||𝑦i−𝑋𝜔−𝑏||^2

  - **最小化损失来学习参数**。训练参数的目的就是使损失函数的值尽可能小（这意味着预估值和真实值更接近）。最后求得的参数值可表示为：

    $\omega^*,b^*=argmin_{\omega,b}l(X,x,\omega,b)$

- **显示解** 

  - 线性回归有显示解，即可以直接矩阵数学运算，得到参数w和b的最优解，而不是用梯度下降，牛顿法等参数优化方式一点点逼近最优解。

  - **推导过程**：

    - 为了方便矩阵表示和计算，将偏差加入权重，$X\gets[X,1],\omega\gets[\omega,b]$

    - 损失函数是凸函数，最优解满足导数为0，可解出显示解

      令$\frac{\partial}{\partial\omega} l(X,y,\omega)=0$

      有$\frac{1}{n}(y-X\omega)^TX=0$

      解得$\omega^*=(X^TX)^{-1}X^Ty$

- 总结

  - 线性回归是对n维输入的加权，外加偏差
  - 使用**平方损失**来衡量预测值和真实值之间的误差
  - **线性回归有显示解**
  - 线性回归可以看作单层神经网络

### 基础优化算法

- **梯度下降**

  - 当模型没有显示解的时候，应用梯度下降法逼近最优解。
  - 梯度下降法的具体步骤：
    - 挑选一个初始值$\omega_0$
    - 重复迭代参数，迭代公式为：$\omega_t=\omega_{t-1}-\lambda\frac{\partial l}{\partial\omega_{t-1} } $
      - **−𝜕𝑙𝜕𝜔𝑡−1为函数值下降最快的方向，学习率$\lambda$为学习步长。**
  - 选择学习率
    - 学习率$\lambda$为学习步长，代表了沿负梯度方向走了多远，这是超参数（人为指定的的值，不是训练得到的）
    - 学习率不能太大，也不能太小，需要选取适当。

- **小批量随机梯度下降**

  - 在整个训练集上算梯度太贵了

    - 在实际应用中，很少直接应用梯度下降法，这是因为每次更新都需要计算训练集上所有的样本，耗费时间太长。一个深度神经网络模型，迭代一次可能需要数分钟甚至数小时。

  - 为了减少运算代价，我们可以==随机采样==b个样本$i_1,i_2,...,i_b$来近似损失，损失函数为：

     1𝑏∑𝑖∈𝐼𝑏𝑙(𝑥𝑖,𝑦𝑖,𝜔) ,

    其中**b是批量大小(batch size)，也是超参数**

  - **选择批量大小**

    - b也不能太大：内存消耗增加；浪费计算资源，一个极端的情况是可能会重复选取很多差不多的样本，浪费计算资源
    - b也不能太小：每次计算量太小，很难以并行，不能最大限度利用GPU资源

- **总结**

  - 梯度下降通过不断**沿着负梯度方向**更新参数求解
  - 小批量随机梯度下降是深度学习默认的求解算法（简单，稳定）
  - **两个重要的超参数：批量大小（batch size），学习率（lr）**

### 线性回归的从零开始实现

```
import random
import torch
from d2l import torch as d2l

def synthetic_data(w, b, num_examples):
    # 生成特征矩阵 X，形状为 (num_examples, len(w))
    X = torch.normal(0, 1, (num_examples, len(w)))
    # 生成标签 y
    y = torch.matmul(X, w) + b
    # 向标签添加噪声
    y += torch.normal(0, 0.01, y.shape)
    # 返回特征和标签，标签的形状调整为 (-1, 1)
    return X, y.reshape((-1, 1))

# 定义真实的权重和偏置
true_w = torch.tensor([2, -3.4])
true_b = 4.2
# 生成特征和标签
features, labels = synthetic_data(true_w, true_b, 1000)

# 打印前五个特征和标签
print("Features:", features[:5])
print("Labels:", labels[:5])

d2l.set_figsize()
d2l.plt.scatter(features[:,1].detach().numpy(),labels.detach().numpy(),1)
```

```
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i:min(i + batch_size, num_examples)]
        )
        yield features[batch_indices], labels[batch_indices]

# 测试数据迭代器
batch_size = 10
for X, y in data_iter(batch_size, features, labels):
    print(X, '\n', y)
    break
```

```
#初始化参数
w =torch.normal(0,0.01,(2,1),requires_grad=true)
b = torch.zeros(1,requires_grad=true)

#定义模型
def linreg(x,w,b):
	return torch.matmul(x,w)+b 
	
#定义损失函数
def squared_loss(y_hat,y):
	return (y_hat-y.reshape(y_hat.shape))**2/2

#定义优化算法
def sgd(params,lr,batch_size):
	with torch.no_grad():
		for param in params:
			param-=lr*param.grad/batch_size
			param.grad.zeros_()
```



```
## 训练过程
lr = 0.03
num_epoches = 3
net = linreg
loss = squared_loss
for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)  # 计算损失
        l.sum().backward()  # 反向传播计算梯度
        sgd([w, b], lr, batch_size)  # 更新参数
    with torch.no_grad():  # 禁止自动计算梯度
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')  # 打印每个 epoch 的损失

```

### 新型回归的简洁实现

```
import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l

true_w = torch.tensor([2, -3.4])
true_b = 4.2
# 生成特征和标签
features, labels = synthetic_data(true_w, true_b, 1000)
```

调用框架中现有的API来读取数据

```
import torch
from torch.utils.data import TensorDataset, DataLoader

def load_array(data_arrays, batch_size, is_train=True):
    dataset = TensorDataset(*data_arrays)
    return DataLoader(dataset, batch_size=batch_size, shuffle=is_train)

batch_size = 10
data_iter = load_array((features, labels), batch_size)
```

使用框架的预定义好的层

```
from torch import nn
net = nn.Sequential(nn.Linear(2,1))

net[0].weight.data.normal(0,0.01)
net[0].bias.data.fill_(0)
```

```
# 计算均方误差使用MSELoss类
loss = nn.MSELoss()

# 实例化SGD
trainer = torch.optim.SGD(net.parameters(),lr=0.03)
```

```
num_epochs=3
for epoch in range(num_epochs):
	for X ,y in data_iter:
		l = loss(net(X),y)
		trainer.zero_grad()
		l.backward()
		trainer.step()
	l = loss(net(features),labels)
	print(f'epoch {epoch + 1},loss {l:f}')
```



## Softmax 回归 + 损失函数 + 图片分类数据集

### 回归VS分类：

- 回归估计一个连续值
- 分类预测一个离散类别

[![image](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/09/09-01.png)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/09/09-01.png)

#### 从回归到多类分类：

##### 回归：

- 单连续数值输出
- 自然区间R
- 跟真实值的区别作为损失

[![image](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/09/09-02.png)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/09/09-02.png)

#### 分类：

- 通常多个输出

- 输出i是预测为第i类的置信度

  [![image](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/09/09-03.png)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/09/09-03.png)

##### 均方损失：



- 对类别进行一位有效编码

  ![image-20240702004009211](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20240702004009211.png)

- 使用均方损失训练

- 最大值为预测 ![](http://latex.codecogs.com/gif.latex?\\ \hat{y}=\underset {i}{argmax}\quad o^{i} )

##### 无校验比例

- 对类别进行一位有效编码
- 最大值为预测 ![](http://latex.codecogs.com/gif.latex?\\ \hat{y}=\underset {i}{argmax}\quad o^{i} )
- 需要更置信的识别正确类（大余量） ![](http://latex.codecogs.com/gif.latex?\\ o_y-o_i\geq\Delta(y,i) )

##### 校验比例

- 输出匹配概率（非负，和为1） ![](http://latex.codecogs.com/gif.latex?\\ \hat{y}=softmax(o) )

  ![](http://latex.codecogs.com/gif.latex?\\ \hat{y_i}=\frac{exp(o_i)}{\sum_{k} exp(o_k)} )

- 概率y和$\hat{y}$的区别作为损失

### Softmax和交叉熵损失

- 交叉熵用来衡量两个概率的区别$H(p,q)=\sum_{i} -p_{i}log(q_i)$
- 将它作为损失 ![](http://latex.codecogs.com/gif.latex?\\ l(y,\hat{y})=-\sum_{i}y_{i}log\hat{y_{i}}=-log\hat{y_y} )
- 其梯度是真实概率和预测概率的区别 ![](http://latex.codecogs.com/gif.latex?\\ \partial_{o_{i}}l(y,\hat{y})=softmax(o)*{i}-y*{i} )

> Softmax回归是一个多类分类模型
>
> 使用Softmax操作子得到每个类的预测置信度
>
> 使用交叉熵来衡量和预测标号的区别

### 损失函数



[![image](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/09/09-04.png)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/09/09-04.png)

#### L2 Loss

![](http://latex.codecogs.com/gif.latex?\\ l(y,y^{'})=\frac{1}{2}(y-y^{'})^2 )

[![image](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/09/09-05.png)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/09/09-05.png)

> 梯度会随着结果逼近而下降

#### L1 Loss



![](http://latex.codecogs.com/gif.latex?\\ l(y,y^{'})=\lvert y-y^{'}\rvert )

[![image](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/09/09-06.png)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/09/09-06.png)

> 梯度保持不变，但在0处梯度随机

#### Huber's Robust Loss



[![image](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/09/09-07.png)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/09/09-07.png)

> 结合L1 Loss和L2 Loss的优点

### 图片分类数据集

#### Fashion-MNIST数据集：

```
import torchvision
import torch
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l 

d2l.use_svg_display()
```

- 读取数据集

  ```
  # 图片类型转变
  trans=transforms.ToTensor()
  
  mnist_train=torchvision.datasets.FashionMNIST(root="../data",train=True,                                              transform=trans,download=True)
  mnist_test=torchvision.datasets.FashionMNIST(root="../data",train=False,                                             transform=trans,download=True)
  ```

- 数据集内图片大小

  ```
  mnist_train[0][0].shape
  torch.Size([1, 28, 28])
  ```

  表示图片为单通道（黑白）的28X28的图片

- 显示数据集图像

  ```
  X,y = next(iter(data.DataLoader(mnist_train,batch_size=18)))
  show_images(X.reshape(18,28,28),2,9,titles=get_fashion_mnist_labels(y))
  ```

  

  [![image](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/09/09-08.png)](https://github.com/MLNLP-World/DeepLearning-MuLi-[Notes/blob/main/imgs/09/09-08.png)

  读取一小批数据

```python
batch_size = 256
def get_dataloader_workers():
	return 4
train_iter = data.Dataloader(mnist_train,batch_size,shuffle=True,num_workers=get_dataLoader_workers())

timer = d2l.Timer()
for X,y in train_iter:
	continue
f'{timer.stop():.2f} sec'
```

### 从零实现softmax回归

#### softmax:

$$ softmax(X)*{ij}=\frac{exp(X*{ij})}{\sum_{k} exp(X_{ik})} $$

```
import torch
from IPthon import display
from d2l import torch as d2l 

batch_size = 256
train_iter,test_iter = d2l.load_data_fasion_mnist(batch_size) 
```

将图像展平，每个图像看做长度为784的向量，因为数据集有十个类别，所以网络输出维度为10。以此设定参数大小并初始化：

```
num_inputs = 784
num_outputs = 10

W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)
```

实现softmax回归模型：

```
def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition
#  矩阵中的非常大或非常小的元素可能造成数值上溢或下溢

def net(X):
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)
```

 实现交叉熵损失函数：

```
def cross_entropy(y_hat, y):
    return - torch.log(y_hat[range(len(y_hat)), y])
```

计算正确率：

```
def accuracy(y_hat, y):  
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())
 
```

评估net精度

```
def evaluate_accuracy(net, data_iter):  
    """计算在指定数据集上模型的精度"""
    if isinstance(net, torch.nn.Module):
        net.eval()
    metric = Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]
```

```
class Accumulator:  
    """在n个变量上累加"""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
```

softmax回归的训练

```
def train_epoch_ch3(net, train_iter, loss, updater):  #@save
    """训练模型一个迭代周期（定义见第3章）"""
    # 将模型设置为训练模式，不再计算grad之类的内容
    if isinstance(net, torch.nn.Module):
        net.train()
    # 训练损失总和、训练准确度总和、样本数
    metric = Accumulator(3)
    for X, y in train_iter:
        # 计算梯度并更新参数
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            # 使用PyTorch内置的优化器和损失函数
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            # 使用定制的优化器和损失函数
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2]
```

定义一个在动画中绘制数据的实用程序类

```
class Animator:  #@save
    """在动画中绘制数据"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        # 增量地绘制多条线
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)
```

定义训练模型：

```
def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):  
    """训练模型（定义见第3章）"""
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
                        
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics
    
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc
```

小批量随机梯度下降优化模型的损失函数

```
lr = 0.1

def updater(batch_size):
    return d2l.sgd([W, b], lr, batch_size)
```

预测：

```
def predict_ch3(net, test_iter, n=6):  
    """预测标签（定义见第3章）"""
    for X, y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true +'\n' + pred for true, pred in zip(trues, preds)]
    d2l.show_images(
        X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])

predict_ch3(net, test_iter)
```

[![image](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/09/09-09.png)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/09/09-09.png)

### softmax的简洁实现

> 调用torch内的网络层

```
import torch
from torch import nn
from d2l import torch as d2l

batch_size=256
train_iter,test_iter=d2l.load_data_fashion_mnist(batch_size)
# PyTorch不会隐式地调整输入的形状。因此，
# 我们在线性层前定义了展平层（flatten），来调整网络输入的形状
net=nn.Sequential(nn.Flatten(),nn.Linear(784,10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight,std=0.01)

net.apply(init_weights)
## 在交叉熵损失函数中传递未归一化的预测，并同时计算softmax及其对数
loss=nn.CrossEntropyLoss()

trainer=torch.optim.SGD(net.parameters(),lr=0.1)

num_epochs=10
d2l.train_ch3(net,train_iter,test_iter,loss,num_epochs,trainer)
```



# 多层感知机

## 多层感知机（MLP）

### 感知机

#### 定义

从现在的观点来看，感知机实际上就是神经网络中的一个神经单元

[![感知机](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/10/%E6%84%9F%E7%9F%A5%E6%9C%BA.png)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/10/感知机.png)

感知机能解决二分类问题，但与线性回归和softmax回归有所区别：线性回归与softmax回归的输出均为实数，softmax回归的输出同时还满足概率公理。

#### 训练

训练感知机的伪代码如下：

```
initialize w = 0 and b = 0
repeat
    #此处表达式小于0代表预测结果错误，表示预测值与真实值同号
    #等价batchsize为1的梯度下降
    if y_i[<w,x_i>+b] <= 0 then
        w=w + yixi
        b=b + yi
    end if
until all classified correctly
```

可以看出这等价于使用如下损失函数的随机梯度下降（batch_size=1）: $$ \ell(y,\bold x,\bold w)=max(0,-y<\bold w,\bold x>)\ =max(0,-y\bold w^T\bold x) $$ 当预测错误时，偏导数为 $$ \frac{\partial \ell}{\partial \bold w}=-y\cdot \bold x $$

注：此处为了方便计算，将偏置项b归入w中的最后一维，并在特征x中相应的最后一维加入常数1

#### 收敛定理

设数据在特征空间能被半径为r的圆（球）覆盖，并且分类时有余量（即$\sigma$函数的输入不会取使输出模棱两可的值）$y(\bold x^T\bold w)\geq \rho$，若初始参数满足$|\bold w|^2+b^2 \leq 1$，则感知机保证在$\frac{r^2+1}{\rho ^2}$步内收敛

[收敛性的证明](https://zhuanlan.zhihu.com/p/46762820)

在前面的课程中我们学习了softmax回归，线性回归，他们有将输入向量与一个权重向量做内积再与一个偏置相加得到一个值的过程： $$ O =W^TX+b $$ 这个过程被称为仿射变换，它是一个带有偏置项的线性变换，它最终产生的模型被称为线性模型，线性模型的特点是只能以线性的方式对特征空间进行划分：

[![线性化分](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/10/%E7%BA%BF%E6%80%A7%E5%88%92%E5%88%86.png)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/10/线性划分.png)

然而，这种线性划分依赖于线性假设，是非常不可靠的

- 线性假设意味着单调假设，这是不可靠的：
  - 对于人体的体温与健康情况的建模，人体在37℃时最为健康，过小过大均有风险，然而这不是单调的
- 线性假设意味着特征与预测存在线性相关性，这也是不可靠的：
  - 如果预测一个人偿还债务的可能性，那这个人的资产从0万元增至5万元和从100万元增至105万元对应的偿还债务的可能性的增幅肯定是不相等的，也就是不线性相关的
- 线性模型的评估标准是有位置依赖性的，这是不可靠的：
  - 如果需要判断图片中的动物是猫还是狗，对于图片中一个像素的权重的改变永远是不可靠的，因为如果将图片翻转，它的类别不会改变，但是线性模型不具备这种性质，像素的权重将会失效

课程中所提到的例子是XOR问题，即希望模型能预测出XOR分类（分割图片中的一三象限与二四象限）：

[![XOR问题](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/10/XOR%E9%97%AE%E9%A2%98.png)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/10/XOR问题.png)

### 多层感知机

#### XOR问题的多层次解决

仍以XOR问题为例，XOR问题的一个解决思路是分类两次，先按x轴分类为+和-，再按y轴分类为+和-，最后将两个分类结果相乘，+即为一三象限，-即为二四象限：

[![多层分类XOR1](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/10/%E5%A4%9A%E5%B1%82%E5%88%86%E7%B1%BBXOR1.png)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/10/多层分类XOR1.png)

[![多层分类XOR2](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/10/%E5%A4%9A%E5%B1%82%E5%88%86%E7%B1%BBXOR2.png)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/10/多层分类XOR2.png)

这实际上将信息进行了多层次的传递：

[![XOR信息多层次传递](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/10/XOR%E4%BF%A1%E6%81%AF%E5%A4%9A%E5%B1%82%E6%AC%A1%E4%BC%A0%E9%80%92.png)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/10/XOR信息多层次传递.png)

其中蓝色为按X坐标的正负进行的分类，橙色为按Y坐标的正负进行的分类，灰色为将二者信息的综合，这就实现了用多层次的线性模型对非线性进行预测

#### 多层感知机

有了XOR问题的解决经验，可以想到如果将多个感知机堆叠起来，形成具有多个层次的结构，如图：

[![单隐藏层](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/10/%E5%8D%95%E9%9A%90%E8%97%8F%E5%B1%82.png)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/10/单隐藏层.png)

这里的模型称为多层感知机，第一层圆圈$x_1,x_2,x_3,x_4$称为输入（实际上他并非感知机），之后的一层称为隐藏层，由5个感知机构成，他们均以前一层的信息作为输入，最后是输出层，以前一层隐藏层的结果作为输入。除了输入的信息和最后一层的感知机以外，其余的层均称为隐藏层，隐藏层的设置为模型一个重要的超参数，这里的模型有一个隐藏层。

#### 激活函数

但是仅仅有线性变换是不够的，如果我们简单的将多个线性变换按层次叠加，由于线性变换的结果仍为线性变换，所以最终的结果等价于线性变换，与单个感知机并无区别，反而加大了模型，浪费了资源，为了防止这个问题，需要对每个单元（感知机）的输出通过激活函数进行处理再交由下一层的感知机进行运算，这些激活函数就是解决非线性问题的关键。

*激活函数*（activation function）通过计算加权和并加上偏置来确定神经元是否应该被激活，它们将输入信号转换为输出的可微运算。大多数激活函数都是非线性的。

主要的激活函数有：

##### ReLU函数

最受欢迎的激活函数是*修正线性单元*（Rectified linear unit，*ReLU*），因为它实现简单，同时在各种预测任务中表现良好。**ReLU提供了一种非常简单的非线性变换**。给定元素$x$，ReLU函数被定义为该元素与$0$的最大值： $$ \operatorname{ReLU}(x) = \max(x, 0) $$ ReLU函数通过将相应的活性值设为0，仅保留正元素并丢弃所有负元素。为了直观感受一下，我们可以画出函数的曲线图。正如从图中所看到，激活函数是分段线性的。使用ReLU的原因是，它求导表现得特别好：要么让参数消失，要么让参数通过。这使得优化表现的更好，并且ReLU减轻了困扰以往神经网络的梯度消失问题

##### sigmoid函数

**对于一个定义域在$\mathbb{R}$中的输入，\*sigmoid函数\*将输入变换为区间(0,1)上的输出**。 因此，sigmoid通常称为*挤压函数*（squashing function）：它将范围$（-\infty, \infty）$中的任意输入压缩到区间（0,1）中的某个值： $$ \operatorname{sigmoid}(x) = \frac{1}{1 + e^{-x}}. $$ 在基于梯度的学习中，sigmoid函数是一个自然的选择，因为它是一个平滑的、可微的阈值单元近似。当我们想要将输出视作二元分类问题的概率时，sigmoid仍然被广泛用作输出单元上的激活函数（你可以将sigmoid视为softmax的特例）。然而，sigmoid在隐藏层中已经较少使用，它在大部分时候被更简单、更容易训练的ReLU所取代。

##### tanh函数

与sigmoid函数类似，**tanh(双曲正切)函数也能将其输入压缩转换到区间(-1,1)上**。tanh函数的公式如下： $$ \operatorname{tanh}(x) = \frac{1 - e^{-2x}}{1 + e^{-2x}} $$

### 代码实现

#### 从零实现

```
import torch
from torch import nn
from d2l import torch as d2l

batch_size=256
train_iter,test_iter = d2l.load_data_fashion_mnist(batch_size)
```

实现一个具有单隐藏层的多层感知机，具有256个隐藏单元。

Fashion-MNIST中的每个图像由 28×28=784个灰度像素值组成。 所有图像共分为10个类别。 忽略像素之间的空间结构， 我们可以将每个图像视为具有784个输入特征 和10个类的简单分类数据集。

```
num_inputs, num_outputs, num_hiddens = 784, 10, 256

W1 = nn.Parameter(torch.radn(num_inputs,num_hiddens,requires_grad = True)* 0.01)
b1 = nn.Parameter(torch.zeros(num_hiddens,requires_grad = True))
W2 = nn.Parameter(torch.randn(num_hiddens,num_outputs,requires_grad = True)* 0.01)
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))

params = [W1, b1, W2, b2]
```

实现ReLU激活函数

```
def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)
```

实现模型

```
def net(X):
	X = X.reshape((-1,num_inputs))
	H = relu(X@W1 + b1)
	return (H@W2 + b2)
```

损失函数 直接使用高级API中的内置函数来计算softmax和交叉熵损失

```
loss = nn.CrossEntropyLoss(reduction='none')
```

```
num_epochs, lr = 10, 0.1
updater = torch.optim.SGD(params, lr=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)
```

#### 简化代码

```
import torch
from torch import nn
from d2l import torch as d2l

net = nn.Sequential(nn.Flatten(),
                    nn.Linear(784, 256),
                    nn.ReLU(),
                    nn.Linear(256, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights)

batch_size, lr, num_epochs = 256, 0.1, 10
loss = nn.CrossEntropyLoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), lr=lr)

train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
```





## 模型选择 + 过拟合和欠拟合

### 概念区分

#### 训练误差与泛化误差

训练误差基于训练集，是在训练数据上的误差

泛化误差基于测试集，是在新数据上的误差

#### 验证数据集与测试数据集

验证数据集：一个用来评估模型好坏的数据集

测试数据集：只用一次的数据集

### K-则交叉验证

（在没有足够多数据时使用）

算法：

​	将训练数据分割成K块

​	使用第i块作为验证数据集，其余作为训练数据集

​	报告k个验证集误差的平均，常用k=5/10

### 过拟合和欠拟合

![image-20240702223819681](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20240702223819681.png)

![image-20240702224134621](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20240702224134621.png)

#### 估计模型容量

##### 难以在不同的种类算法之间比较

例如树模型和神经网络

##### 给定一个模型种类，将有两个主要因素

参数个数

参数值的选择范围

#### 数据复杂度

样本个数

样本的元素个数

时间/空间结构

多样性

### 代码（多项式回归）

![image-20240702225202168](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20240702225202168.png)

```
import math
import numpy as np
import torch
from torch import nn
from d2l import torch as d2l

max_degree = 20  # 多项式的最大阶数
n_train, n_test = 100, 100  # 训练和测试数据集大小
true_w = np.zeros(max_degree)  # 分配大量的空间
true_w[0:4] = np.array([5, 1.2, -3.4, 5.6])  #后面全是噪音

## 构造数据集
features = np.random.normal(size=(n_train + n_test, 1))  #随机正态分布生成数据与测试集
np.random.shuffle(features)    #随机化
poly_features = np.power(features, np.arange(max_degree).reshape(1, -1))   #生成幂指数
for i in range(max_degree):
    poly_features[:, i] /= math.gamma(i + 1)  # gamma(n)=(n-1)!
# labels的维度:(n_train+n_test,)
labels = np.dot(poly_features, true_w)
labels += np.random.normal(scale=0.1, size=labels.shape)

# NumPy ndarray转换为tensor
true_w, features, poly_features, labels = [torch.tensor(x, dtype=
    torch.float32) for x in [true_w, features, poly_features, labels]]
```

对模型进行训练和测试

```
def evaluate_loss(net, data_iter, loss):  #@save
    """评估给定数据集上模型的损失"""
    metric = d2l.Accumulator(2)  # 损失的总和,样本数量
    for X, y in data_iter:
        out = net(X)
        y = y.reshape(out.shape)
        l = loss(out, y)
        metric.add(l.sum(), l.numel())
    return metric[0] / metric[1]
```

**定义训练函数**

```
def train(train_features, test_features, train_labels, test_labels,num_epochs=400):
    loss = nn.MSELoss(reduction='none')
    input_shape = train_features.shape[-1]
    # 不设置偏置，因为我们已经在多项式中实现了它
    net = nn.Sequential(nn.Linear(input_shape, 1, bias=False))
    batch_size = min(10, train_labels.shape[0])
    train_iter = d2l.load_array((train_features, train_labels.reshape(-1,1)),batch_size)
    test_iter = d2l.load_array((test_features, test_labels.reshape(-1,1)),batch_size, is_train=False)
    trainer = torch.optim.SGD(net.parameters(), lr=0.01)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss', yscale='log',xlim=[1, num_epochs], ylim=[1e-3, 1e2],legend=['train', 'test'])
    for epoch in range(num_epochs):
        d2l.train_epoch_ch3(net, train_iter, loss, trainer)
        if epoch == 0 or (epoch + 1) % 20 == 0:
            animator.add(epoch + 1, (evaluate_loss(net, train_iter, loss),evaluate_loss(net, test_iter, loss)))
    print('weight:', net[0].weight.data.numpy())
```



## 权重衰退（正则化模型的技术）

在训练参数化机器学习模型时， *权重衰减*（weight decay）是最广泛使用的正则化的技术之一， 它通常也被称为𝐿2*正则化*。 这项技术通过函数与零的距离来衡量函数的复杂度， 因为在所有函数𝑓中，函数𝑓=0（所有输入都得到值0） 在某种意义上是最简单的。

一种简单的方法是通过线性函数 𝑓(𝑥)=𝑤⊤𝑥 中的权重向量的某个范数来度量其复杂性， 例如∥𝑤∥2。 要保证权重向量比较小， 最常用方法是将其范数作为惩罚项加到最小化损失的问题中。 将原来的训练目标*最小化训练标签上的预测损失*， 调整为*最小化预测损失和惩罚项之和*。 现在，如果我们的权重向量增长的太大， 我们的学习算法可能会更集中于最小化权重范数∥𝑤∥2。

### 硬性限制/直观理解

**使用均方范数作为硬性限制**

我们的优化目标仍然是[![img](https://camo.githubusercontent.com/c268c1ccd2378c3872e8678310498cca00c9fd8f26bfdb0ce3cf4e983d7a1d56/687474703a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e6c617465783f6d696e2535437370616365253543656c6c28253543626f6c64253742772537442c6229)](https://camo.githubusercontent.com/c268c1ccd2378c3872e8678310498cca00c9fd8f26bfdb0ce3cf4e983d7a1d56/687474703a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e6c617465783f6d696e2535437370616365253543656c6c28253543626f6c64253742772537442c6229)，只是额外对[![img](https://camo.githubusercontent.com/ff9b452cab8da34e38960130e11bdb43c17681b4d715f35495b631be82f83c1d/687474703a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e6c617465783f253543626f6c6425374277253744)](https://camo.githubusercontent.com/ff9b452cab8da34e38960130e11bdb43c17681b4d715f35495b631be82f83c1d/687474703a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e6c617465783f253543626f6c6425374277253744)添加一个限制条件[![img](https://camo.githubusercontent.com/60135f218d6eff45f9d2a2b226b09f667b1e5862c12b77709fe9fe000ce42fdf/687474703a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e6c617465783f253743253743253543626f6c6425374277253744253743253743253545322535436c6571736c616e742535437468657461)](https://camo.githubusercontent.com/60135f218d6eff45f9d2a2b226b09f667b1e5862c12b77709fe9fe000ce42fdf/687474703a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e6c617465783f253743253743253543626f6c6425374277253744253743253743253545322535436c6571736c616e742535437468657461)，即权重的各项平方和小于一个特定的常数[![img](https://camo.githubusercontent.com/2ee9f22d00819650d1da57b8e92428653b1baaeb0a5efbd4f95ee745612fc9a7/687474703a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e6c617465783f2535437468657461)](https://camo.githubusercontent.com/2ee9f22d00819650d1da57b8e92428653b1baaeb0a5efbd4f95ee745612fc9a7/687474703a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e6c617465783f2535437468657461)。那么设定一个较小的[![img](https://camo.githubusercontent.com/2ee9f22d00819650d1da57b8e92428653b1baaeb0a5efbd4f95ee745612fc9a7/687474703a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e6c617465783f2535437468657461)](https://camo.githubusercontent.com/2ee9f22d00819650d1da57b8e92428653b1baaeb0a5efbd4f95ee745612fc9a7/687474703a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e6c617465783f2535437468657461)就会使得[![img](https://camo.githubusercontent.com/ff9b452cab8da34e38960130e11bdb43c17681b4d715f35495b631be82f83c1d/687474703a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e6c617465783f253543626f6c6425374277253744)](https://camo.githubusercontent.com/ff9b452cab8da34e38960130e11bdb43c17681b4d715f35495b631be82f83c1d/687474703a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e6c617465783f253543626f6c6425374277253744)中每个元素的值都不会太大。

通常不会限制偏移b，理论上讲b表示整个数据在零点上的偏移，因此是不应该限制的，但实践中限制与否对结果都没什么影响。

**吴恩达课程中对这一现象的解释是w是高维向量，已经包 含了绝大多数参数足以表达高方差问题，b作为单个数字对结果的影响就会很小.**

**小的[![img](https://camo.githubusercontent.com/2ee9f22d00819650d1da57b8e92428653b1baaeb0a5efbd4f95ee745612fc9a7/687474703a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e6c617465783f2535437468657461)](https://camo.githubusercontent.com/2ee9f22d00819650d1da57b8e92428653b1baaeb0a5efbd4f95ee745612fc9a7/687474703a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e6c617465783f2535437468657461)意味着更强的正则项**，对于相同的[![img](https://camo.githubusercontent.com/2ee9f22d00819650d1da57b8e92428653b1baaeb0a5efbd4f95ee745612fc9a7/687474703a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e6c617465783f2535437468657461)](https://camo.githubusercontent.com/2ee9f22d00819650d1da57b8e92428653b1baaeb0a5efbd4f95ee745612fc9a7/687474703a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e6c617465783f2535437468657461)，[![img](https://camo.githubusercontent.com/ff9b452cab8da34e38960130e11bdb43c17681b4d715f35495b631be82f83c1d/687474703a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e6c617465783f253543626f6c6425374277253744)](https://camo.githubusercontent.com/ff9b452cab8da34e38960130e11bdb43c17681b4d715f35495b631be82f83c1d/687474703a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e6c617465783f253543626f6c6425374277253744)中元素越多则单个元素的值会越小。

### 柔性限制/实际应用



上文说的硬性限制在实际使用时比较麻烦，实际上常用的函数是

[![img](https://camo.githubusercontent.com/aca2480d2e63e33df193ee69d1869f501c3cc3e4123bee06c3510ca892890098/687474703a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e6c617465783f6d696e2535437370616365253543656c6c28253543626f6c64253742772537442c62292b253543667261632537422535436c616d62646125374425374232253744253743253743253543626f6c642537427725374425374325374325354532)](https://camo.githubusercontent.com/aca2480d2e63e33df193ee69d1869f501c3cc3e4123bee06c3510ca892890098/687474703a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e6c617465783f6d696e2535437370616365253543656c6c28253543626f6c64253742772537442c62292b253543667261632537422535436c616d62646125374425374232253744253743253743253543626f6c642537427725374425374325374325354532)

可以通过拉格朗日乘子证明对于每个[![img](https://camo.githubusercontent.com/2ee9f22d00819650d1da57b8e92428653b1baaeb0a5efbd4f95ee745612fc9a7/687474703a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e6c617465783f2535437468657461)](https://camo.githubusercontent.com/2ee9f22d00819650d1da57b8e92428653b1baaeb0a5efbd4f95ee745612fc9a7/687474703a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e6c617465783f2535437468657461)都可以找到[![img](https://camo.githubusercontent.com/2ec5723e26bb86479707648ec0b1b1f9183674b99e31bcce5cbe2835d1e7f9e1/687474703a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e6c617465783f2535436c616d626461)](https://camo.githubusercontent.com/2ec5723e26bb86479707648ec0b1b1f9183674b99e31bcce5cbe2835d1e7f9e1/687474703a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e6c617465783f2535436c616d626461)使得硬性限制的目标函数等价于上式。

其中[![img](https://camo.githubusercontent.com/db89c02a9c53b868fba4f1c0c9e4e7d986adeb498d5324dcb072189d24ad3904/687474703a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e6c617465783f253543667261632537422535436c616d62646125374425374232253744253743253743253543626f6c642537427725374425374325374325354532)](https://camo.githubusercontent.com/db89c02a9c53b868fba4f1c0c9e4e7d986adeb498d5324dcb072189d24ad3904/687474703a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e6c617465783f253543667261632537422535436c616d62646125374425374232253744253743253743253543626f6c642537427725374425374325374325354532)这一项被称为罚(penalty)，[![img](https://camo.githubusercontent.com/2ec5723e26bb86479707648ec0b1b1f9183674b99e31bcce5cbe2835d1e7f9e1/687474703a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e6c617465783f2535436c616d626461)](https://camo.githubusercontent.com/2ec5723e26bb86479707648ec0b1b1f9183674b99e31bcce5cbe2835d1e7f9e1/687474703a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e6c617465783f2535436c616d626461)**是超参数，控制了正则项的重要程度**。

当[![img](https://camo.githubusercontent.com/bc056ceedf61585d3b6a3f773ef4edf48c1311f60f335d60d30ba6060270c40c/687474703a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e6c617465783f2535436c616d6264613d30)](https://camo.githubusercontent.com/bc056ceedf61585d3b6a3f773ef4edf48c1311f60f335d60d30ba6060270c40c/687474703a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e6c617465783f2535436c616d6264613d30)时无作用，[![img](https://camo.githubusercontent.com/61cf8e5887264e92d223f8e3044131588ab79b6276c3c73935362d73f6fa9cc3/687474703a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e6c617465783f2535436c616d62646125354372696768746172726f77253543696e667479)](https://camo.githubusercontent.com/61cf8e5887264e92d223f8e3044131588ab79b6276c3c73935362d73f6fa9cc3/687474703a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e6c617465783f2535436c616d62646125354372696768746172726f77253543696e667479)时最优解[![img](https://camo.githubusercontent.com/5deec64969e5a58e2d4259232575940bfbbdf093a0472cea94e2a5305d3739ba/687474703a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e6c617465783f253543626f6c64253742772537442535452a25354372696768746172726f7730)](https://camo.githubusercontent.com/5deec64969e5a58e2d4259232575940bfbbdf093a0472cea94e2a5305d3739ba/687474703a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e6c617465783f253543626f6c64253742772537442535452a25354372696768746172726f7730)，也就是说[![img](https://camo.githubusercontent.com/2ec5723e26bb86479707648ec0b1b1f9183674b99e31bcce5cbe2835d1e7f9e1/687474703a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e6c617465783f2535436c616d626461)](https://camo.githubusercontent.com/2ec5723e26bb86479707648ec0b1b1f9183674b99e31bcce5cbe2835d1e7f9e1/687474703a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e6c617465783f2535436c616d626461)越大模型复杂度就被控制的越低。

[![12-01](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/12/12-01.JPG)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/12/12-01.JPG)

以[![img](https://camo.githubusercontent.com/ff9b452cab8da34e38960130e11bdb43c17681b4d715f35495b631be82f83c1d/687474703a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e6c617465783f253543626f6c6425374277253744)](https://camo.githubusercontent.com/ff9b452cab8da34e38960130e11bdb43c17681b4d715f35495b631be82f83c1d/687474703a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e6c617465783f253543626f6c6425374277253744)中只有两个参数为例，其中绿色的部分是原本损失函数函数值的“等高线”，黄色部分可以看作是正则项对应函数值的“等高线” ，使用权重衰减后需要优化的损失函数相当于图中两组等高线叠加。原本最优解位于绿色中心，现在这一位置在对于正则项有很高的损失，而正则项最小值位于原点，因此现在的最终优化解会更靠近原点，而当所有参数都更靠近原点时模型的规模也就更小。

（我们用参数值的范围来衡量模型的大小）

### 参数更新

#### 计算梯度 



[![img](https://camo.githubusercontent.com/816e5c55c47c5414a4ddafeae7feaccbd6446371d3719dcd4bb12b53196747da/687474703a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e6c617465783f253543667261632537422535437061727469616c2537422537442537442537422535437061727469616c253742253543626f6c642537427725374425374425374428253543656c6c28253543626f6c64253742772537442c62292b253543667261632537422535436c616d62646125374425374232253744253743253743253543626f6c642537427725374425374325374325354532293d253543667261632537422535437061727469616c253742253543656c6c28253543626f6c64253742772537442c62292537442537442537422537422535437061727469616c253742253543626f6c64253742772537442537442537442537442b2535436c616d626461253543626f6c6425374277253744)](https://camo.githubusercontent.com/816e5c55c47c5414a4ddafeae7feaccbd6446371d3719dcd4bb12b53196747da/687474703a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e6c617465783f253543667261632537422535437061727469616c2537422537442537442537422535437061727469616c253742253543626f6c642537427725374425374425374428253543656c6c28253543626f6c64253742772537442c62292b253543667261632537422535436c616d62646125374425374232253744253743253743253543626f6c642537427725374425374325374325354532293d253543667261632537422535437061727469616c253742253543656c6c28253543626f6c64253742772537442c62292537442537442537422537422535437061727469616c253742253543626f6c64253742772537442537442537442537442b2535436c616d626461253543626f6c6425374277253744)、

#### 更新参数



将上式结果带入更新参数公式整理可得

[![img](https://camo.githubusercontent.com/221e250a978f50f162533c2a1f5bc34e70bfd1751255e95b1c11eaf4bfdec1a1/687474703a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e6c617465783f253543626f6c64253742772537445f253742742b312537443d28312d2535436574612535436c616d62646129253543626f6c64253742772537445f253742742537442d253543657461253543667261632537422535437061727469616c253742253543656c6c28253543626f6c64253742772537445f742c625f74292537442537442537422537422535437061727469616c253742253543626f6c64253742772537445f25374274253744253744253744253744)](https://camo.githubusercontent.com/221e250a978f50f162533c2a1f5bc34e70bfd1751255e95b1c11eaf4bfdec1a1/687474703a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e6c617465783f253543626f6c64253742772537445f253742742b312537443d28312d2535436574612535436c616d62646129253543626f6c64253742772537445f253742742537442d253543657461253543667261632537422535437061727469616c253742253543656c6c28253543626f6c64253742772537445f742c625f74292537442537442537422537422535437061727469616c253742253543626f6c64253742772537445f25374274253744253744253744253744)

注意到这个公式中后一项与原来更新参数的公式没有区别，仅仅是在前一项[![img](https://camo.githubusercontent.com/779be9c1d70105fed3cfa9549c24c591ab988cf098abfce4af42c0d069c64c18/687474703a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e6c617465783f253543626f6c64253742772537445f25374274253744)](https://camo.githubusercontent.com/779be9c1d70105fed3cfa9549c24c591ab988cf098abfce4af42c0d069c64c18/687474703a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e6c617465783f253543626f6c64253742772537445f25374274253744) 上加了一个系数[![img](https://camo.githubusercontent.com/36e26287fd68d846aa72c6befe07fbaf4130e11a2a7a5a45c8eb4fe552993054/687474703a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e6c617465783f28312d2535436574612535436c616d62646129)](https://camo.githubusercontent.com/36e26287fd68d846aa72c6befe07fbaf4130e11a2a7a5a45c8eb4fe552993054/687474703a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e6c617465783f28312d2535436574612535436c616d62646129)。通常[![img](https://camo.githubusercontent.com/4c3bff3b6dcd9ffc7a7122b114f60f2f3687b910cebc2f00f751b30a4071f11d/687474703a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e6c617465783f2535436574612535436c616d62646125334331)](https://camo.githubusercontent.com/4c3bff3b6dcd9ffc7a7122b114f60f2f3687b910cebc2f00f751b30a4071f11d/687474703a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e6c617465783f2535436574612535436c616d62646125334331) ，也就是说由于引入了[![img](https://camo.githubusercontent.com/2ec5723e26bb86479707648ec0b1b1f9183674b99e31bcce5cbe2835d1e7f9e1/687474703a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e6c617465783f2535436c616d626461)](https://camo.githubusercontent.com/2ec5723e26bb86479707648ec0b1b1f9183674b99e31bcce5cbe2835d1e7f9e1/687474703a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e6c617465783f2535436c616d626461)，每次更新参数前先给待更新参数乘上一个小于1的权重再更新，权重衰退由此得名。

（由于lambda的引入，我们每次的w都会先进行衰退再减去学习率*步长）

### 总结

- 权重衰退通过L2正则项使得模型参数不会过大，从而控制模型复杂度
- 正则项权重（[![img](https://camo.githubusercontent.com/2ec5723e26bb86479707648ec0b1b1f9183674b99e31bcce5cbe2835d1e7f9e1/687474703a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e6c617465783f2535436c616d626461)](https://camo.githubusercontent.com/2ec5723e26bb86479707648ec0b1b1f9183674b99e31bcce5cbe2835d1e7f9e1/687474703a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e6c617465783f2535436c616d626461)）是控制模型复杂度的超参数

### 代码

生成数据

![image-20240703012706882](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20240703012706882.png)

```
import torch
from torch import nn
from d2l import torch as d2l

n_train, n_test, num_inputs, batch_size = 20, 100, 200, 5
true_w, true_b = torch.ones((num_inputs, 1)) * 0.01, 0.05
train_data = d2l.synthetic_data(true_w, true_b, n_train)
train_iter = d2l.load_array(train_data, batch_size)
test_data = d2l.synthetic_data(true_w, true_b, n_test)
test_iter = d2l.load_array(test_data, batch_size, is_train=False)
```

初始化模型参数

```
def init_params():
	w = torch.normal(0, 1, size=(num_inputs, 1),											requires_grad=True)
	b = torch.zeros(1,requires_grad=True)
	return [w,b]
```

定义L2范数

```
def L2_penalty():
	return torch.sum(w.pow(2))/2
```

定义训练代码实现

```
def train(lambd):
    w, b = init_params()
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    num_epochs, lr = 100, 0.003
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log', xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            # 增加了L2范数惩罚项，
            # 广播机制使l2_penalty(w)成为一个长度为batch_size的向量
            l = loss(net(X), y) + lambd * l2_penalty(w)
            l.sum().backward()
            d2l.sgd([w, b], lr, batch_size)
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1, (d2l.evaluate_loss(net, train_iter, loss),d2l.evaluate_loss(net, test_iter, loss)))
    print('w的L2范数是：', torch.norm(w).item())
	
```

简介实现

```
def train_concise(wd):
    net = nn.Sequential(nn.Linear(num_inputs, 1))
    for param in net.parameters():
        param.data.normal_()
    loss = nn.MSELoss(reduction='none')
    num_epochs, lr = 100, 0.003
    # 偏置参数没有衰减
    trainer = torch.optim.SGD([
        {"params":net[0].weight,'weight_decay': wd},
        {"params":net[0].bias}], lr=lr)
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.mean().backward()
            trainer.step()
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1,
                         (d2l.evaluate_loss(net, train_iter, loss),
                          d2l.evaluate_loss(net, test_iter, loss)))
    print('w的L2范数：', net[0].weight.norm().item())
```







## 丢弃法

### 丢弃法动机、实现及原则

#### 动机

- 一个好的模型需要对输入数据的扰动鲁棒（健壮性）

#### [![13-02](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/13/13-02.jpg)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/13/13-02.jpg) [![13-03](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/13/13-03.jpg)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/13/13-03.jpg)

#### 如何实现模型的这一能力

- 使用有噪音的数据。
- 丢弃法：在层之间加入噪音。

#### 加入噪音的原则

无偏差的加入噪音

[![13-01](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/13/13-01.png)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/13/13-01.png)

- 例如模型的功能是识别猫猫，加入噪音可以是输入模糊的猫猫图片，但尽量不要是狗狗的图片。

### 丢弃法内容

- 丢弃法对每个元素作如下扰动

[![13-04](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/13/13-04.png)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/13/13-04.png)

- 能够满足加入噪音的期望相同原则

[![13-05](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/13/13-05.png)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/13/13-05.png)

### 丢弃法使用

#### 丢弃法的使用位置

通常将丢弃法作用在隐藏全连接层的输出上

[![13-06](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/13/13-06.png)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/13/13-06.png)

随机选中某些神经元将其输出置位0，因此模型不会过分依赖某些神经元

[![13-07](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/13/13-07.png)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/13/13-07.png)

#### 训练中的丢弃法

- 正则项（丢弃法）仅在训练中使用：影响模型参数的更新，预测的时候便不再使用

### 总结

- 丢弃法将一些输出项随机置0来控制模型复杂度
- 常作用在多层感知机的隐藏层输出上
- 丢弃概率是控制模型复杂度的超参数（常取0.9，0.5，0.1）

### 代码部分

#### Drpout部分

```
import torch
from torch import nn
from d2l import torch as d2l

# X为dropout层的输入，dropout为设置的丢弃概率
def dropout_layer (X,dropout)：  
    assert 0<=dropout<=1        #丢弃概率介于0，1之间
    if dropout == 1:
       return torch.zeros_like(x) #若丢弃概率为1，则X的全部项均被置0
    if dropout == 0:
       return X                   #若丢弃概率为0，不对X作丢弃操作，直接返回X
       
    mask=(torch.randn(X.shape)>dropout).float() 
    return mask*X/(1-dropout) 
    #将mask与X相乘实现丢弃操作，并除以(1-dropout)，
    这里不使用选中X中元素置0的原因是相乘操作相比选中操作更快
```

#### 在神经网络中使用丢弃法

```
num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256

dropout1, dropout2 = 0.2, 0.5

class Net(nn.Module)
    # 确认是在训练阶段
    def _init_(self,num_inputs,num_outputs,
    num_outputs,num_hiddens1,num_hiddens2,is_training=True):
       super(Net,self)._init_()
       self.num_inputs=num_inputs
       self.training=is_training
       self.lin1=nn.Linear(num_inputs,num_hiddens1)
       self.lin2=nn.Linear(num_hiddens1,num_hiddens2)
       self.lin2=nn.Linear(num_hiddens2,num_outputs)
       self.relu=nn.ReLU()
    
    def forward(self,X):
      H1=self.relu(self.lin1(X.reshape((-1,self.num_inputs))))
       if self.training == True:  #丢弃法仅在训练中使用
           H1=dropout_layer(H1,dropout1)
       H2=self.relu(self.lin2(H1))
       if self.training == True: #丢弃法仅在训练中使用
           H2=dropout_layer(H2,dropout2)
       out=self.lin3(H2)  #output层不再使用丢弃法
       return out
```

简洁实现

```
net = nn.Sequential(nn.Flatten(),
        nn.Linear(784, 256),
        nn.ReLU(),
        # 在第一个全连接层之后添加一个dropout层
        nn.Dropout(dropout1),
        nn.Linear(256, 256),
        nn.ReLU(),
        # 在第二个全连接层之后添加一个dropout层
        nn.Dropout(dropout2),
        nn.Linear(256, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights);
```







## 数值稳定性 + 模型初始化和激活函数

### 数值稳定性

数值稳定性是深度学习中比较重要的点，特别是当神经网络变得很深的时候，数值通常很容易变得不稳定。

#### 神经网络的梯度

[![image](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/14/14-01.png)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/14/14-01.png)

**考虑d层神经网络**

- t表示层数，[![h^{t-1}](https://camo.githubusercontent.com/a71cebb102ebd87361c6309af0486f84cf349a20dec37eb3f2cae5bc0b90b22a/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e696d6167653f685e7b742d317d)](https://camo.githubusercontent.com/a71cebb102ebd87361c6309af0486f84cf349a20dec37eb3f2cae5bc0b90b22a/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e696d6167653f685e7b742d317d)表示第*t-1*层的输出，经过一个[![f_{t}](https://camo.githubusercontent.com/ea7eda19f58ecfb0aa08ee4f8aa706c973834f1d814cfd6cc862932921b2ea4b/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e696d6167653f665f7b747d)](https://camo.githubusercontent.com/ea7eda19f58ecfb0aa08ee4f8aa706c973834f1d814cfd6cc862932921b2ea4b/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e696d6167653f665f7b747d)函数后，得到第*t*层的输出。
- 最终输出y的表示：输入x经过若干层(*d*层)的函数作用，最后被损失函数作用得到输出y。

**计算损失函数\*L\*关于第\*t\*层参数[![W_{t} ](https://camo.githubusercontent.com/cd456da509cd823ff05c8a8147686bebe7f70b0479d17469c343f39a9cc2a8d7/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e696d6167653f575f7b747d2673706163653b)](https://camo.githubusercontent.com/cd456da509cd823ff05c8a8147686bebe7f70b0479d17469c343f39a9cc2a8d7/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e696d6167653f575f7b747d2673706163653b)的梯度**

- 由链导法则得到上图中乘积公式
- 需要进行d-t次**矩阵乘法**（为什么是矩阵乘法？答：由于所有的*h*都是一些**向量**，导数中分子分母均为向量，所以求导得到的是矩阵，维数为[分子维度]x[分母维度]，可以参考第6节[视频](https://www.bilibili.com/video/BV1eZ4y1w7PY)和[笔记](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/notes/06-矩阵计算.md)）。这也是导致数值稳定性问题的**主要因素**，由于做了太多次的矩阵乘法。

#### 数值稳定性的常见两个问题

**梯度爆炸**

假设梯度都是一些比1大的数比如1.5，做100次乘积之后得到[![4\times 10^{17}](https://camo.githubusercontent.com/cb03a67eca0aab939a881ca6e61c07bfa0c7a014080a832c6ef22dba43c45436/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e696d6167653f345c74696d65732673706163653b31305e7b31377d)](https://camo.githubusercontent.com/cb03a67eca0aab939a881ca6e61c07bfa0c7a014080a832c6ef22dba43c45436/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e696d6167653f345c74696d65732673706163653b31305e7b31377d)，这个数字很容易带来一些浮点数上限的问题（需了解更多请参考计算机系统-计算机中浮点数的存储方式）。

**梯度消失**

假设梯度都是一些比1小的数比如0.8，做100次乘积之后得到[![2\times10^{-10}](https://camo.githubusercontent.com/a231a5a2891622e8dc7f0d3a45b8e4c5fc159338ba446402166557ea948d16e4/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e696d6167653f325c74696d657331305e7b2d31307d)](https://camo.githubusercontent.com/a231a5a2891622e8dc7f0d3a45b8e4c5fc159338ba446402166557ea948d16e4/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e696d6167653f325c74696d657331305e7b2d31307d)，也可能会带来浮点数下溢的问题。

#### 例子：MLP

此处我们着重探讨[1.1节](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/notes/14-数值稳定性.md#11-神经网络的梯度)中所述的求梯度时所做的d-t次矩阵乘法，并以一个实例MLP来探讨其结果的具体形式。

[![image](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/14/14-02.png)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/14/14-02.png)

- 第一行公式，定义[![h^{t}](https://camo.githubusercontent.com/be807542e1881c650372a6b6555cb672468be104bb488370e96db8ac22417766/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e696d6167653f685e7b747d)](https://camo.githubusercontent.com/be807542e1881c650372a6b6555cb672468be104bb488370e96db8ac22417766/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e696d6167653f685e7b747d)和[![h^{t-1}](https://camo.githubusercontent.com/a71cebb102ebd87361c6309af0486f84cf349a20dec37eb3f2cae5bc0b90b22a/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e696d6167653f685e7b742d317d)](https://camo.githubusercontent.com/a71cebb102ebd87361c6309af0486f84cf349a20dec37eb3f2cae5bc0b90b22a/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e696d6167653f685e7b742d317d)(均为向量)的函数关系[![f_{t}](https://camo.githubusercontent.com/ea7eda19f58ecfb0aa08ee4f8aa706c973834f1d814cfd6cc862932921b2ea4b/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e696d6167653f665f7b747d)](https://camo.githubusercontent.com/ea7eda19f58ecfb0aa08ee4f8aa706c973834f1d814cfd6cc862932921b2ea4b/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e696d6167653f665f7b747d)，第t层的权重矩阵作用于t-1层的输出[![h^{t-1}](https://camo.githubusercontent.com/a71cebb102ebd87361c6309af0486f84cf349a20dec37eb3f2cae5bc0b90b22a/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e696d6167653f685e7b742d317d)](https://camo.githubusercontent.com/a71cebb102ebd87361c6309af0486f84cf349a20dec37eb3f2cae5bc0b90b22a/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e696d6167653f685e7b742d317d)后经过激活函数[![\sigma ](https://camo.githubusercontent.com/ebf39e7aad643a5167823309290a6aafec495508d69747797eb052e48131fded/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e696d6167653f5c7369676d612673706163653b)](https://camo.githubusercontent.com/ebf39e7aad643a5167823309290a6aafec495508d69747797eb052e48131fded/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e696d6167653f5c7369676d612673706163653b)得到[![h^{t}](https://camo.githubusercontent.com/be807542e1881c650372a6b6555cb672468be104bb488370e96db8ac22417766/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e696d6167653f685e7b747d)](https://camo.githubusercontent.com/be807542e1881c650372a6b6555cb672468be104bb488370e96db8ac22417766/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e696d6167653f685e7b747d)，注意激活函数[![\sigma ](https://camo.githubusercontent.com/ebf39e7aad643a5167823309290a6aafec495508d69747797eb052e48131fded/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e696d6167653f5c7369676d612673706163653b)](https://camo.githubusercontent.com/ebf39e7aad643a5167823309290a6aafec495508d69747797eb052e48131fded/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e696d6167653f5c7369676d612673706163653b)逐元素计算。

- 第二行公式：这里用到链导法则，激活函数[![\sigma ](https://camo.githubusercontent.com/ebf39e7aad643a5167823309290a6aafec495508d69747797eb052e48131fded/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e696d6167653f5c7369676d612673706163653b)](https://camo.githubusercontent.com/ebf39e7aad643a5167823309290a6aafec495508d69747797eb052e48131fded/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e696d6167653f5c7369676d612673706163653b)先对内部向量逐元素求导，然后把求导后这个向量变成对角矩阵（可以理解为链导法则中内部向量[![W_{t}h_{t-1}](https://camo.githubusercontent.com/2e6701f04d89ef9b5335964cbec64476b9ddadb5b6ca6bc64d73b2b8009f06c9/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e696d6167653f575f7b747d685f7b742d317d)](https://camo.githubusercontent.com/2e6701f04d89ef9b5335964cbec64476b9ddadb5b6ca6bc64d73b2b8009f06c9/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e696d6167653f575f7b747d685f7b742d317d)对自身进行求导，变成一个nxn的对角矩阵，更多请参考[邱锡鹏 《神经网络与深度学习》](https://nndl.github.io/nndl-book.pdf)[1](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/notes/14-数值稳定性.md#user-content-fn-图片1-2bb18940390e507adda6b4ee943dcd6a)）

  [![image](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/14/14-03.png)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/14/14-03.png)

- 视频中**勘误说明**：链导法则中 [![\frac{\partial W^{t}h^{t-1}}{\partial h^{t-1}}= W^{t}](https://camo.githubusercontent.com/71fa9f16c839f40f6fe979e6f72bc88ddf24c3d304e576cf6463c3ad09e6a010/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e696d6167653f5c667261637b5c7061727469616c2673706163653b575e7b747d685e7b742d317d7d7b5c7061727469616c2673706163653b685e7b742d317d7d3d2673706163653b575e7b747d)](https://camo.githubusercontent.com/71fa9f16c839f40f6fe979e6f72bc88ddf24c3d304e576cf6463c3ad09e6a010/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e696d6167653f5c667261637b5c7061727469616c2673706163653b575e7b747d685e7b742d317d7d7b5c7061727469616c2673706163653b685e7b742d317d7d3d2673706163653b575e7b747d) 而不是[![\left (W^{t} \right )^{T}](https://camo.githubusercontent.com/83dc04c534c36c75a02d9516553d3555aef6df231859509223a9953a04942948/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e696d6167653f5c6c6566742673706163653b28575e7b747d2673706163653b2673706163653b5c72696768742673706163653b295e7b547d)](https://camo.githubusercontent.com/83dc04c534c36c75a02d9516553d3555aef6df231859509223a9953a04942948/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e696d6167653f5c6c6566742673706163653b28575e7b747d2673706163653b2673706163653b5c72696768742673706163653b295e7b547d)（这点由分子分母维度也容易推出），故最终求导结果包含[![W^{t}](https://camo.githubusercontent.com/90fa60edb85fc018d18ff5c3e21e28f3c075a514b1c31cba204dba5995760657/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e696d6167653f575e7b747d)](https://camo.githubusercontent.com/90fa60edb85fc018d18ff5c3e21e28f3c075a514b1c31cba204dba5995760657/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e696d6167653f575e7b747d)，而不是其转置。

#### 梯度爆炸

##### 使用ReLU作为激活函数



[![image](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/14/14-04.png)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/14/14-04.png)

由于激活函数Relu求导后或者是1或者是0，变为对角矩阵的斜对角线元素后，与[![W^{i}](https://camo.githubusercontent.com/b672ed6c4894aead93fb0506d22544a527353e71b36e572f7dd042825883dbb3/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e696d6167653f575e7b697d)](https://camo.githubusercontent.com/b672ed6c4894aead93fb0506d22544a527353e71b36e572f7dd042825883dbb3/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e696d6167653f575e7b697d)做乘积，斜对角线为1的部分会使得W中元素保留，最终该连乘式中有一些元素来自[![\prod\left ( W^{i} \right ) ](https://camo.githubusercontent.com/4eb293144f80b51159f7c69a62a047aad97e5d9bbe61e83f4fe6234323f63360/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e696d6167653f5c70726f645c6c6566742673706163653b282673706163653b575e7b697d2673706163653b5c72696768742673706163653b292673706163653b)](https://camo.githubusercontent.com/4eb293144f80b51159f7c69a62a047aad97e5d9bbe61e83f4fe6234323f63360/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e696d6167653f5c70726f645c6c6566742673706163653b282673706163653b575e7b697d2673706163653b5c72696768742673706163653b292673706163653b)，如果大部分[![W^{i}](https://camo.githubusercontent.com/b672ed6c4894aead93fb0506d22544a527353e71b36e572f7dd042825883dbb3/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e696d6167653f575e7b697d)](https://camo.githubusercontent.com/b672ed6c4894aead93fb0506d22544a527353e71b36e572f7dd042825883dbb3/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e696d6167653f575e7b697d)中 值都大于1，且层数比较大，那么连乘之后可能导致梯度爆炸的问题。

##### 梯度爆炸问题

- 值超出值域（infinity）
  - 对于16位浮点数尤为严重（数值区间 [6e-5 , 6e4]），GPU用16位浮点数更快
- 对学习率敏感
  - 如果学习率太大→大参数值→更大的梯度，如此循环几次，容易导致梯度爆炸
  - 如果学习率太小→训练无进展
  - 我们可能需要在训练过程中不断调整学习率

#### 梯度消失

##### 使用Sigmoid作为激活函数

[![image](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/14/14-05.png)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/14/14-05.png)

- 蓝色曲线为函数值
- 黄色曲线为梯度，注意到当输入x值取±6时，此时梯度已经变得很小，由图也可以看出，当输入值稍大或稍小都很容易引起小梯度。

[![image](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/14/14-06.png)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/14/14-06.png)

所以最终连乘式中[![\prod diag\left ( \sigma ^{'}\left ( W^{i}h^{i-1} \right ) \right ) ](https://camo.githubusercontent.com/ed117f76d7cda611a8d81fcae67f6bf383bce937de9b079c5ccaaf30bdb6c5b8/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e696d6167653f5c70726f642673706163653b646961675c6c6566742673706163653b282673706163653b5c7369676d612673706163653b5e7b277d5c6c6566742673706163653b282673706163653b575e7b697d685e7b692d317d2673706163653b5c72696768742673706163653b292673706163653b5c72696768742673706163653b292673706163653b)](https://camo.githubusercontent.com/ed117f76d7cda611a8d81fcae67f6bf383bce937de9b079c5ccaaf30bdb6c5b8/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e696d6167653f5c70726f642673706163653b646961675c6c6566742673706163653b282673706163653b5c7369676d612673706163653b5e7b277d5c6c6566742673706163653b282673706163653b575e7b697d685e7b692d317d2673706163653b5c72696768742673706163653b292673706163653b5c72696768742673706163653b292673706163653b)项乘出来会很小，导致整个梯度很小，产生梯度消失问题。

##### 梯度消失的问题

- 梯度值变为0
  - 对16位浮点数尤为严重
- 训练没有进展
  - 不管如何选择学习率，由于梯度已经为0了，学习率x梯度=0
- 对于底部层尤为严重
  - 仅仅顶部层训练得较好。第*t*层导数包含d-t个矩阵乘积，越往底层走，t越小，乘得越多，梯度消失越严重，所以底部层效果更差。
  - 无法让神经网络更深。只能把顶部层训练得比较好，底部层跑不动，这和给一个浅的神经网络没有什么区别。

### 模型初始化和激活函数

#### 让训练更加稳定

我们的一个核心目标是如何让训练更稳定，梯度值不要太大也不要太小

- 目标：让梯度值在合理的范围内
  - 例如 [1e-6, 1e3]
- 常用方法：
  - 将乘法变加法：
    - ResNet（跳跃连接，如果很多层，加入加法进去）
    - LSTM（引入记忆细胞，更新门，遗忘门，通过门权重求和，控制下一步是否更新）
  - 归一化：
    - 梯度归一化（归一化均值，方差）
    - 梯度裁剪(clipping)：比如大于/小于一个固定的阈值，就让梯度等于这个阈值，将梯度限制在一个范围中。（可以缓解梯度爆炸）
  - 合理的权重初始和激活函数：本节课讲述重点

**下面我们重点探讨最后一种方法：合理的权重初始和激活函数**

#### 基本假设：让每层的均值/方差是一个常数

- **将每层的输出和梯度都看做随机变量**

  比如第i层有100维，就将输出和梯度分别看成100个随机变量

- **让它们的均值和方差都保持一致**

  我们的目标，这样不管神经网络多深，最后一层总与第一层差不多，从而不会梯度爆炸和消失

根据我们的假设，可以列出如下方程式：

[![image](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/14/14-07.png)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/14/14-07.png)

#### 权重初始化

- 在合理值区间里随机初始参数

- 训练

  开始

  的时候更容易有数值不稳定

  - 远离最优解的地方损失函数表面可能很复杂
  - 最优解附近表面会比较平

- 使用N(0, 0.01)分布来初始可能对小网络没问题，但不能保证深度神经网络

#### 例子：MLP

下面我们以MLP为例，考虑需要什么条件，才能满足[2.2节](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/notes/14-数值稳定性.md#22-基本假设：让每层的均值/方差是一个常数)的假设。

##### 模型假设

- 每一层**权重**中的变量均为**独立同分布**，并设出均值、方差。
- 每一层**输入**的变量**独立于**该层**权重**变量。同时**输入变量**之间**独立同分布**。
- 假设没有激活函数(先简化分析，之后会考虑有激活函数的情况)，可以求得该层输出的期望为0。

[![image](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/14/14-08.png)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/14/14-08.png)

此处用到了一个重要性质：

[![image](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/14/14-09.png)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/14/14-09.png)

更多均值、方差运算可以参考[期望、方差、协方差及相关系数的基本运算](https://blog.csdn.net/MissXy_/article/details/80705828)

##### 正向方差

[![image](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/14/14-10.png)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/14/14-10.png)

- 第二行的计算中仍然用到了[2.4.1节](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/notes/241模型假设)的期望的重要性质：如果两个变量独立，它们乘积的均值=均值的乘积，再结合w的期望为0(注意w和h独立，w之间独立同分布)，即有第二行末项期望为0。
- 最后一行由于wi,j独立同分布，方差相同，加上做了hj独立同分布的假设，所以可以写成 **[t-1层输出维度] x [t层权重方差] x [t-1层输出方差]** 的形式
- 此时，我们回过头来看我们的终极目标[2.2节](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/notes/14-数值稳定性.md#22-基本假设：让每层的均值/方差是一个常数)的假设，每层输出期望为0我们已经可以满足(2.4.1节已经推导出)，而方差相同这一目标，通过上图的推导，我们发现需要[![ n_{t-1}\gamma _{t}=1 ](https://camo.githubusercontent.com/f78bf82d006b08b156283f8f36616c89d1f31532215546b92819e2cc772c773a/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e696d6167653f2673706163653b6e5f7b742d317d5c67616d6d612673706163653b5f7b747d3d312673706163653b)](https://camo.githubusercontent.com/f78bf82d006b08b156283f8f36616c89d1f31532215546b92819e2cc772c773a/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e696d6167653f2673706163653b6e5f7b742d317d5c67616d6d612673706163653b5f7b747d3d312673706163653b)。

##### 反向均值和方差

[![image](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/14/14-11.png)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/14/14-11.png)

反向的情况和正向的类似，不过此时我们需要满足的式子变为[![ n_{t}\gamma _{t}=1 ](https://camo.githubusercontent.com/9b5cf0aa698dc23314756d854eac930e1469588685dffa3b9fbefeec54b963a6/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e696d6167653f2673706163653b6e5f7b747d5c67616d6d612673706163653b5f7b747d3d312673706163653b)](https://camo.githubusercontent.com/9b5cf0aa698dc23314756d854eac930e1469588685dffa3b9fbefeec54b963a6/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e696d6167653f2673706163653b6e5f7b747d5c67616d6d612673706163653b5f7b747d3d312673706163653b)。

##### Xavier初始

- 上述推导带来的问题：难以同时满足[![ n_{t-1}\gamma _{t}=1 ](https://camo.githubusercontent.com/f78bf82d006b08b156283f8f36616c89d1f31532215546b92819e2cc772c773a/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e696d6167653f2673706163653b6e5f7b742d317d5c67616d6d612673706163653b5f7b747d3d312673706163653b)](https://camo.githubusercontent.com/f78bf82d006b08b156283f8f36616c89d1f31532215546b92819e2cc772c773a/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e696d6167653f2673706163653b6e5f7b742d317d5c67616d6d612673706163653b5f7b747d3d312673706163653b)和[![ n_{t}\gamma _{t}=1 ](https://camo.githubusercontent.com/9b5cf0aa698dc23314756d854eac930e1469588685dffa3b9fbefeec54b963a6/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e696d6167653f2673706163653b6e5f7b747d5c67616d6d612673706163653b5f7b747d3d312673706163653b)](https://camo.githubusercontent.com/9b5cf0aa698dc23314756d854eac930e1469588685dffa3b9fbefeec54b963a6/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e696d6167653f2673706163653b6e5f7b747d5c67616d6d612673706163653b5f7b747d3d312673706163653b)。（需要每层输出的维度都相同）

- 采用Xavier折中解决，不能同时满足上面两式，转而满足 [**上面两式做加法后除以2**] 得到的式子，用两种分布进行初始化（每层方差、均值满足推导式）。

  [![image](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/14/14-12.png)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/14/14-12.png)

- 如果能确定每层输入、输出维度大小，则能确定该层权重的方差大小。

- 权重初始化方式：正态分布、均匀分布，均值/方差满足Xavier的假设。

##### 假设线性的激活函数

真实情况下，我们并不会用线性的激活函数（这样相当于没有进行激活），这里为了简化问题，假设激活函数是线性的。

- **正向**

[![image](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/14/14-13.png)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/14/14-13.png)

上述推导表明，为了使得前向传播的均值为0，方差固定的话，激活函数必须f(x)=x，这种恒等映射。

- **反向**

[![image](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/14/14-14.png)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/14/14-14.png)

PPT上的推导似乎有点问题（上图中第二行方程），笔者重新进行了下述推导，读者也可自行推导验证：

[![image](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/14/14-15.png)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/14/14-15.png)

**通过正向和反向的推导，我们可以得出的【结论】是：当激活函数为f(x)=x，这种恒等映射更有利于维持神经网络的稳定性。**

##### 检查常用激活函数

[![image](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/14/14-16.png)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/14/14-16.png)

对于常用激活函数：tanh，relu满足在零点附近有f(x)=x，而sigmoid函数在零点附近不满足要求，可以对sigmoid函数进行调整（根据Taylor展开式，调整其过原点）

### 总结

- 当数值过大或者过小时，会导致数值问题。
- 常发生在深度模型中，因为其会对n个数累乘。
- 合理的权重初始值(如Xavier)和激活函数的选取(如relu, tanh, 调整后的sigmoid)可以提升数值稳定性。





## 实战

```
import hashlib  #这个模块可以用于生成文件哈希值
import os  #处理文件和目录
import tarfile  #处理.tar和.tar.gz文件
import zipfile  #用于处理.zip文件
import requests  

DATA_HUB = dict()
DATA_URL = ###

# cache_dir为缓存目录，默认值为.../data
def download(name,cache_dir=os.path.join('..','data')):
	# 检查名字是否存在
	assert name in DATA_HUB,f"{name} 不存在于 {DATA_HUB}"
	# 获取数据集的 URL 和 SHA1 哈希值
	url,sha1_hash = DATA_HUB[name]
	# 创建缓存目录
	os.makedirs(cash_dir,exist_ok=True)
	fname = os.path.join(cach_dir,url.split('/')[-1])
	if os.path.exists(fname):
		sha1 = hashlib.sha1()
			with open(fname,'rb') as f:
                while True:
                    data = f.read(1048576):
                    if not data:
                        break
                    sha1.update(data)
            if sha1.hexdigest() == sha1_hash:
            	return fname
    print(f'正在从{url}下载{fname}...')
    r = requests.get(url, stream=True, verify=True)
    with open(fname, 'wb') as f:
        f.write(r.content)
    return fname

def download_extract(name, folder=None):  #@save
    """下载并解压zip/tar文件"""
    fname = download(name)
    base_dir = os.path.dirname(fname)
    data_dir, ext = os.path.splitext(fname)
    if ext == '.zip':
        fp = zipfile.ZipFile(fname, 'r')
    elif ext in ('.tar', '.gz'):
        fp = tarfile.open(fname, 'r')
    else:
        assert False, '只有zip/tar文件可以被解压缩'
    fp.extractall(base_dir)
    return os.path.join(base_dir, folder) if folder else data_dir

def download_all():  #@save
    """下载DATA_HUB中的所有文件"""
    for name in DATA_HUB:
        download(name)
```

数据展示-数据处理（不需要的数值删除，缺失值替换，处理离散值我们用one-hot编码来替换它们）-从pandas格式中提取NUMPY格式，并将其转化成张量形式

对于数据预处理部分：

![image-20240703193142034](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20240703193142034.png)

```
import numpy as np
import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l

#读取数据
train_data = pd.read.csv((download('kaggle_house_train'))
test_data = pd.read_csv(download('kaggle_house_test'))

#读取数据集形状
print(train_data.shape)
print(test_data.shape)

# 读取数据集特征及对应标签
print(train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])

由于ID与数据集本身无太大关联（不携带预测信息）
all_features = pd.concat((train_data.iloc[:,1:-1],test_data[:,1:]))

## 数据预处理
# 将所有缺失的值替换为相应特征的平均值。通过将特征重新缩放到零均值和单位方差来标准化数据
# 若无法获得测试数据，则可根据训练数据计算均值和标准差
numeric_features = all_features.dtype[all_features.dtype!='object'].index
all_features[numeric_features] = all_features[numeric_features].apply(lambd x:(x-x.mean())/x.std())
# 在标准化数据之后，所有均值消失，因此我们可以将缺失值设置为0
all_features[numeric_features] = all_features[numeric_features].fillna(0)


# 处理离散值 若特征仅有两个值1则一个看作0，另一个看作1
# 使用 pd.get_dummies 函数进行独热编码（One-Hot Encoding）和处理缺失值的目的是将类别型特征转换为适合机器学习算法的数值格式。
# “Dummy_na=True”将“na”（缺失值）视为有效的特征值，并为其创建指示符特征
all_features = pd.get_dummies(all_features,dummy_na=True)
all_features.shape


## 通过values属性，我们可以 [从pandas格式中提取NumPy格式，并将其转换为张量表示]用于训练。
n_train= train.data.shape[0]
train_featrue = torch.tensor(all_features[:n_train].values,dtype=torch.float32)
test_featrue = torch.tensor(all_features[n_train:].values,dtype=torch.float32)
train_labels = torch.tensor(train_data.SalePrices.values.reshape(-1:1),dtype=torch.float32)
```

训练

![image-20240703195033844](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20240703195033844.png)

```
#带损失平方的线性模型（base）
loss = nn.MSELoss
in_features = train_features.shape[1]

def get_nn():
	net = nn.Sequential(nn.linear(infeatures,1))
	return net
	
	
# 使用相对误差（由于数据值相差较大）
def log_rmse(net, features, labels):
    # 使用 torch.clamp 函数将预测值限制在 [1, ∞) 之间，避免对数计算中的负值或零
    clipped_preds = torch.clamp(net(features), 1,float('inf'))
    # 计算 log(RMSE)
    rmse = torch.sqrt(nn.MSELoss()(torch.log(clipped_preds), torch.log(labels)))
    return rmse.item()
	
	
# 训练函数借助ADAM优化器
def train(net, train_features, train_labels, test_features, test_labels, num_epochs, weight_decay, learning_rate, batch_size):
	# 用于保存每个 epoch 的训练和测试损失。
    train_ls, test_ls = [], []
    train_iter = d2l.load_array((train_features, train_labels), batch_size)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()  # 假设您使用 MSE 作为损失函数
    
    for epoch in range(num_epochs):
        net.train()  # 设置网络为训练模式
        for X, y in train_iter:
            optimizer.zero_grad()
            l = loss_fn(net(X), y)
            l.backward()
            optimizer.step()
        net.eval()  # 设置网络为评估模式
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls
    
    
# K折交叉验证
def get_k_fold_data(k,i,X,y):
	assert k>1
	flod_size = X.shape[0] // k
	X_train,y_train=None,None
	for j in range(k):
		idx = slice(j*fold_size,(j+1)*fold_size)
		X_part,y_part = X[idx,:],y[idx,:]
		if j==i:
			X_valid,y_valid=X_part,y_part
		elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
        #将当前折的数据与已有的训练集数据拼接起来。
        X_train = torch.cat([X_train,X_part],0)
        y_train = torch.cat([y_train, y_part], 0)
    return X_train, y_train, X_valid, y_valid

def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay,batch_size):
	train_l_sum,valid_l_sum=0,0
	for i in range(k):
		data =  get_k_fold_data(k, i, X_train, y_train)
		net = get_net()
		train_ls, valid_ls = train(net, *data, num_epochs, learning_rate, weight_decay, batch_size)
		train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
    if i == 0:
            d2l.plot(list(range(1, num_epochs + 1)), [train_ls, valid_ls],
                     xlabel='epoch', ylabel='rmse', xlim=[1, num_epochs],
                     legend=['train', 'valid'], yscale='log')
        print(f'折{i + 1}，训练log rmse{float(train_ls[-1]):f}, '
              f'验证log rmse{float(valid_ls[-1]):f}')
    return train_l_sum / k, valid_l_sum / k
```

提交预测









# 深度学习计算

## PyTorch 神经网络基础

### 层和块

在之前的内容中，我们认识了一些神经网络，比如：线性回归，Softmax回归，多层感知机；他们有的是整个模型，有的是一层神经网络，有的甚至只是一个单元，他们的功能以及复杂程度也各不相同，但他们都有着如下三个特征：

- 接受一些输入
- 产生对应的输出
- 由一组可调整参数描述

对于一些复杂的网络，研究讨论比层大但比整个模型小的部分很有意义，因为复杂的网络中经常有重复出现的部分，每个部分也常常有自己的功能。考虑到上面的三个特征，这就使得我们思考是否可以对这些部分进行一个抽象，这就得到了块的概念：块指单个层，多个层组成的部分，或者整个模型本身。使用块对整个模型进行描述就简便许多，这一过程是递归的，块的内部还可以划分为多个块，直至满足需要为止。

PyTorch帮我们实现了块的大部分所需功能，包括自动求导，我们只需从nn.Module继承并改写其中的一部分就能得到我们需要的块以及模型。

**`nn.Sequential`**

**定义了一种特殊的`Module`**， 即在PyTorch中表示一个块的类， 它维护了一个由`Module`组成的有序列表。 

#### **自定义块**

**每个块必须提供的基本功能：**

1. 将输入数据作为其前向传播函数的参数。
2. 通过前向传播函数来生成输出。请注意，输出的形状可能与输入的形状不同。例如，我们上面模型中的第一个全连接的层接收一个20维的输入，但是返回一个维度为256的输出。
3. 计算其输出关于输入的梯度，可通过其反向传播函数进行访问。通常这是自动发生的。
4. 存储和访问前向传播计算所需的参数。
5. 根据需要初始化模型参数。

**实例

```
import torch
from torch import nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_layer = nn.Linear(20, 256)
        self.out_layer = nn.Linear(256, 10)

    def forward(self, X):
        X = F.relu(self.hidden_layer(X))
        return self.out_layer(X)

# 示例：如何使用该模型
# 假设输入数据有 20 个特征
model = MLP()
X = torch.rand(5, 20)  # 生成 5 个样本，每个样本有 20 个特征
output = model(X)
print(output)

```

##### 顺序块

现在我们可以更仔细地看看`Sequential`类是如何工作的， 回想一下`Sequential`的设计是为了把其他模块串起来。 为了构建我们自己的简化的`MySequential`， 我们只需要定义两个关键函数：

1. 一种将块逐个追加到列表中的函数；
2. 一种前向传播函数，用于将输入按追加块的顺序传递给块组成的“链条”。

```
class mySequential(nn.Module):
	def __init__(self,**args):
		super().__init__
		for idx,module in enumerate(args):
			self._modules[str(idx)] = module
	
	def forward(self,X):
		for block in self._module.values:
			X = block(X)
		return X
```

##### **在前向传播函数中执行代码**

`Sequential`类使模型构造变得简单， 允许我们组合新的架构，而不必定义自己的类。 然而，并不是所有的架构都是简单的顺序架构。 当需要更强的灵活性时，我们需要定义自己的块。 例如，我们可能希望在前向传播函数中执行Python的控制流。 此外，我们可能希望执行任意的数学运算， 而不是简单地依赖预定义的神经网络层。

到目前为止， 我们网络中的所有操作都对网络的激活值及网络的参数起作用。 然而，有时我们可能希望合并既不是上一层的结果也不是可更新参数的项， 我们称之为*常数参数*（constant parameter）。 例如，我们需要一个计算函数 𝑓(𝑥,𝑤)=𝑐⋅𝑤⊤𝑥的层， 其中𝑥是输入， 𝑤是参数， 𝑐是某个在优化过程中没有更新的指定常量。 因此我们实现了一个`FixedHiddenMLP`类，如下所示：

```
class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # 不计算梯度的随机权重参数。因此其在训练期间保持不变
        self.rand_weight = torch.rand((20, 20), requires_grad=False)
        self.linear = nn.Linear(20, 20)

    def forward(self, X):
        X = self.linear(X)
        # 使用创建的常量参数以及relu和mm函数
        X = F.relu(torch.mm(X, self.rand_weight) + 1)
        # 复用全连接层。这相当于两个全连接层共享参数
        X = self.linear(X)
        # 控制流
        while X.abs().sum() > 1:
            X /= 2
        return X.sum()
```

### 参数管理

在选择了架构并设置了超参数后，我们就进入了训练阶段。此时，我们的目标是找到使损失函数最小化的模型参数值。经过训练后，我们将需要使用这些参数来做出未来的预测。此外，有时我们希望提取参数，以便在其他环境中复用它们，将模型保存下来，以便它可以在其他软件中执行，或者为了获得科学的理解而进行检查。

#### 参数访问

```
# 我们先构建一个单隐藏层的多层感知机
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
## 参数访问+索引访问+利用state_dict
net[2].state_dict()
```

#### 目标参数

```
# 提取偏置
net[2].bias.data
net[2].bias
#
tensor([-0.1032])
tensor([-0.1032], requires_grad=True)
```

#### 一次性访问所有参数

```
# 利用named_parameters()
print(*[(name,param.shape) for name,param in net[0].named_parameters()])
print(*[(name,param.shape) for name,param in net.named_parameters()])
# 另一种采用state_dict，这是一个Ordereddict
netp[2].state_dict()['2.bias;].data
```

#### 从嵌套块中收集参数

```
def block1():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                         nn.Linear(8, 4), nn.ReLU())

def block2():
    net = nn.Sequential()
    for i in range(4):
    	net.add_module(f'blcok{i}',block1())
    return net

rgnet = nn.Sequential(block2(),nn.Linear(4,1))
rgnet(X)
```

#### 参数初始化

##### 内置初始化

```
def init_normal(m):
    if type(m) == nn.Linear:
    	#用正态分布初始化
    	nn.init.normal(m.weight,mean=0,std=0.01)
    	#常数
    	nn.init.constant_(m.weight, 1)
    	#xavier
    	nn.init.xavier_uniform_(m.weight)
    	
    	nn.init.zeros(m.bias)
    	
net.apply(init_normal)
net[0].weight.data[0], net[0].bias.data[0]
```

##### 自定义初始化

![image-20240704152743711](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20240704152743711.png)

#### 参数绑定

有时我们希望在多个层间共享参数： 我们可以定义一个稠密层，然后使用它的参数来设置另一个层的参数。

```
# 我们需要给共享层一个名称，以便可以引用它的参数
shared = nn.Linear(8, 8)
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                    shared, nn.ReLU(),
                    shared, nn.ReLU(),
                    nn.Linear(8, 1))
net(X)
# 检查参数是否相同
print(net[2].weight.data[0] == net[4].weight.data[0])
net[2].weight.data[0, 0] = 100
# 确保它们实际上是同一个对象，而不只是有相同的值
print(net[2].weight.data[0] == net[4].weight.data[0])
```

### 延后初始化

有时在建立网络时，我们不会指定网络的输入输出维度，也就不能确定网络的参数形状，深度学习框架支持延后初始化，即当第一次将数据传入模型时自动的得到所有的维度，然后初始化所有的参数。

PyTorch也支持这一点，比如nn.LazyLinear，但本门课程中并未介绍。

### 自定义层

深度学习成功背后的一个因素是神经网络的灵活性：我们可以用创造性的方式组合不同的层，从而设计出适用于各种任务的架构。例如，研究人员发明了专门用于处理图像、文本、序列数据和执行动态规划的层。同样的，对于层而言，深度学习框架并不能满足我们所有的需求，然而，层本身也具有极大的灵活性，我们可以自定义想要的层。

### 读写文件

到目前为止，我们讨论了如何处理数据，以及如何构建、训练和测试深度学习模型。然而，有时我们希望保存训练的模型，以备将来在各种环境中使用（比如在部署中进行预测）。此外，当运行一个耗时较长的训练过程时，最佳的做法是定期保存中间结果，以确保在服务器电源被不小心断掉时，我们不会损失几天的计算结果。







# 卷积神经网络

## 卷积层

### 平移不变性

![image-20240704182222900](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20240704182222900.png)

### 局部性

![image-20240704183058348](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20240704183058348.png)

### 代码

互相关运算

```
import torch
from torch import nn

def corr2d(X,K):    #X为输入，K为核矩阵
    h,w=K.shape    #h得到K的行数，w得到K的列数
    Y=torch.zeros((X.shape[0]-h+1,X.shape[1]-w+1))  #用0初始化输出矩阵Y
    for i in range(Y.shape[0]):   #卷积运算
        for j in range(Y.shape[1]):
          Y[i,j]=(X[i:i+h,j:j+w]*K).sum()
    return Y
```

实现二维卷积层

```
#实现二维卷积层
class Conv2d(nn.Module):
    def _init_(self,kernel_size):
        super()._init_()
        self.weight=nn.Parameter(torch.rand(kerner_size))
        self.bias=nn.Parameter(torch.zeros(1))
    def forward(self,x):
        return corr2d(x,self.weight)+self.bias 
```

建立一个边缘检测

```
X=torch.ones((6,8))
X[:,2:6]=0
X
```

```
>>> tensor([[1., 1., 0., 0., 0., 0., 1., 1.],
        [1., 1., 0., 0., 0., 0., 1., 1.],
        [1., 1., 0., 0., 0., 0., 1., 1.],
        [1., 1., 0., 0., 0., 0., 1., 1.],
        [1., 1., 0., 0., 0., 0., 1., 1.],
        [1., 1., 0., 0., 0., 0., 1., 1.]])
```

```
K=torch.tensor([[-1,1]])  #这个K只能检测垂直边缘
Y=corr2d(X,K)
Y
```

```
>>> tensor([[ 0., -1.,  0.,  0.,  0.,  1.,  0.],
            [ 0., -1.,  0.,  0.,  0.,  1.,  0.],
            [ 0., -1.,  0.,  0.,  0.,  1.,  0.],
            [ 0., -1.,  0.,  0.,  0.,  1.,  0.],
            [ 0., -1.,  0.,  0.,  0.,  1.,  0.],
            [ 0., -1.,  0.,  0.,  0.,  1.,  0.]])
```

检测垂直的边缘

```
corr2d(X.t(),K)
```

```
>>> tensor([[0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.]])
```

学习由X生成Y的卷积核

```
# 输入，输出，卷积核，偏置
conv2d = nn.Conv2d(1, 1, kernel_size=(1, 2), bias=False)

# 通道数，批量大小，高度，宽度
X = X.reshape((1, 1, 6, 8))
Y = Y.reshape((1, 1, 6, 7))

for i in range(10):
    Y_hat = conv2d(X)
    l = (Y_hat-Y)**2//2
    conv2d.zero_grad()
    l.sum().backward()
    conv2d.weight.data[:]-=6e-2*conv2d.weight.grad
    if (i+1) % 2 == 0:
    	print(f'batch {i+1} ,loss {l.sum():3f}}
```

```
>>> batch 2, loss 3.852
    batch 4, loss 1.126
    batch 6, loss 0.386
    batch 8, loss 0.145
    batch 10, loss 0.057
```

## 卷积层里的填充和步幅

###  填充 

**填充**(Padding)指的是在输入周围添加额外的行/列

[![image](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/20/20-01.png)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/20/20-01.png)

**维度变化**：

[![image](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/20/20-02.png)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/20/20-02.png)

**两种不同的卷积方式**： ①Valid 卷积：不进行填充，卷积运算过后得到的矩阵形状为(n-f+1)×(n-f+1)。

②Same 卷积：先对矩阵进行填充，然后再进行卷积运算，使得运算前后矩阵大小不变。

[![image](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/20/20-03.png)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/20/20-03.png)

### 步幅  

**想法来源：如果按照原来的操作(卷积步长为1)，那么给定输入大小为224x224，在使用5x5卷积核的情况下，需要55层**才能将输出降低到4x4，也就是说，需要大量的计算才能得到维度较小的输出。

**步幅**是指行/列的滑动步长

[![image](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/20/20-04.png)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/20/20-04.png)

**维度变化**:

[![image](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/20/20-05.png)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/20/20-05.png)

注意：第三点可以当做结论来记(Same卷积或Valid卷积(且s≥k时))。一般来说，如果n是偶数，s取2，池化层做Valid卷积(不填充)且k=2，此时输出维度直接可以写成n/2 x n/2。如果怕搞混，直接记第一个公式每次现推也可。

### 总结 

- 填充和步幅是卷积层的**超参数**
- **填充**(padding)在输入周围添加额外的行/列，来控制输出形状的减少量
- **步幅**(stride)是每次滑动核窗口时的行/列的步长，可以成倍地减少输出形状

### 代码

#### 填充和步幅

**导入包，定义comp_conv2d函数 (进行卷积操作, 输出后两维，便于观察高宽的维度变化)**

```
import torch
from torch import nn

def comp_conv2d(conv2d, X):
	# 维度的前面加入批量大小数(batch_size)和输入通道数(channel_in)
    X = X.reshape((1, 1) + X.shape) 
    Y = conv2d(X)                    
    return Y.reshape(Y.shape[2:])  #去掉前面的两维后(原来四维) 进行输出
```

#### padding

**在所有侧边填充1个像素(padding=1, 即(1,1))**

```
conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1) #输入输出通道数为1, 卷积核大小3x3, 填充为1(上下左右各填充一行)
X = torch.rand(size=(8, 8))         
comp_conv2d(conv2d, X).shape
```

```
>>> torch.Size([8, 8])
```

**填充不同的高度和宽度(padding=(2,1))**

```
conv2d = nn.Conv2d(1, 1, kernel_size=(5, 3), padding=(2, 1))
comp_conv2d(conv2d, X).shape
```

```
>>> torch.Size([8, 8])
```

#### stride

**将高度和宽度的步幅设置为2**

```
conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=2)
comp_conv2d(conv2d, X).shape
```

```
>>> torch.Size([4, 4])
```

**一个稍微复杂的例子**

```
conv2d = nn.Conv2d(1, 1, kernel_size=(3, 5), padding=(0, 1), stride=(3, 4))
comp_conv2d(conv2d, X).shape
```

```
>>> torch.Size([2, 2])
```

## 多输入多输出通道

### 多个输入通道

- 彩色图像可能有RGB三个通道

- 转换为灰度会丢失信息

  [![image](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/21/21-01.png)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/21/21-01.png)

- 每个通道都有一个卷积和，结果是所有通道卷积结果的和

[![image](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/21/21-02.png)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/21/21-02.png)

- 输入**X**:[![c_{i}\times n_{h}\times n_{w}](https://camo.githubusercontent.com/ede4ec360206a8830f9e4238a35c436d5881462fce077e695ae3bfd683ccbe56/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e696d6167653f635f7b697d5c74696d65732673706163653b6e5f7b687d5c74696d65732673706163653b6e5f7b777d)](https://camo.githubusercontent.com/ede4ec360206a8830f9e4238a35c436d5881462fce077e695ae3bfd683ccbe56/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e696d6167653f635f7b697d5c74696d65732673706163653b6e5f7b687d5c74696d65732673706163653b6e5f7b777d)
- 核**W**：[![c_{i}\times k_{h}\times k_{w}](https://camo.githubusercontent.com/e4217dc5acd1775e3830925e71415d8059621f1b5616a2ae8e53d9264303f872/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e696d6167653f635f7b697d5c74696d65732673706163653b6b5f7b687d5c74696d65732673706163653b6b5f7b777d)](https://camo.githubusercontent.com/e4217dc5acd1775e3830925e71415d8059621f1b5616a2ae8e53d9264303f872/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e696d6167653f635f7b697d5c74696d65732673706163653b6b5f7b687d5c74696d65732673706163653b6b5f7b777d)
- 输出**Y**:[![m_{h}\times m_{w}](https://camo.githubusercontent.com/0704054ba0d6b123fe7d3508371099665253aecec2c758f3cba4005d24305367/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e696d6167653f6d5f7b687d5c74696d65732673706163653b6d5f7b777d)](https://camo.githubusercontent.com/0704054ba0d6b123fe7d3508371099665253aecec2c758f3cba4005d24305367/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e696d6167653f6d5f7b687d5c74696d65732673706163653b6d5f7b777d)

[![Y=\sum _{i=0}^{c_{i}}X_{i,:,:}\bigstar W_{i,:,:}](https://camo.githubusercontent.com/11eddda729f3b5b08fb8c797e2fe70e66dcd6a4758c676ad943dc41a773fc910/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e696d6167653f593d5c73756d2673706163653b5f7b693d307d5e7b635f7b697d7d585f7b692c3a2c3a7d5c626967737461722673706163653b575f7b692c3a2c3a7d)](https://camo.githubusercontent.com/11eddda729f3b5b08fb8c797e2fe70e66dcd6a4758c676ad943dc41a773fc910/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e696d6167653f593d5c73756d2673706163653b5f7b693d307d5e7b635f7b697d7d585f7b692c3a2c3a7d5c626967737461722673706163653b575f7b692c3a2c3a7d)

多个输入通道：

```
import torch
from d2l import torch as d2l

def corr2d_multi_in(X, K):
    return sum(d2l.corr2d(x, k) for x, k in zip(X, K))
```

### 多个输出通道

- 无论有多少输入通道，到目前位置我们植绒到单输出通道
- 我们可以有多个三维卷积核，每个核生成一个输出通道
- 输入**X**:[![c_{i}\times k_{h}\times k_{w}](https://camo.githubusercontent.com/e4217dc5acd1775e3830925e71415d8059621f1b5616a2ae8e53d9264303f872/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e696d6167653f635f7b697d5c74696d65732673706163653b6b5f7b687d5c74696d65732673706163653b6b5f7b777d)](https://camo.githubusercontent.com/e4217dc5acd1775e3830925e71415d8059621f1b5616a2ae8e53d9264303f872/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e696d6167653f635f7b697d5c74696d65732673706163653b6b5f7b687d5c74696d65732673706163653b6b5f7b777d)
- 核**W**：[![c_{o}\times c_{i}\times k_{h}\times k_{w}](https://camo.githubusercontent.com/db922e672cf0c939ff058271b413d0b0c81b0da02a111bf6714bea10c20ea491/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e696d6167653f635f7b6f7d5c74696d65732673706163653b635f7b697d5c74696d65732673706163653b6b5f7b687d5c74696d65732673706163653b6b5f7b777d)](https://camo.githubusercontent.com/db922e672cf0c939ff058271b413d0b0c81b0da02a111bf6714bea10c20ea491/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e696d6167653f635f7b6f7d5c74696d65732673706163653b635f7b697d5c74696d65732673706163653b6b5f7b687d5c74696d65732673706163653b6b5f7b777d)
- 输出**Y**：[![c_{o}\times m_{h}\times m_{w}](https://camo.githubusercontent.com/d7f7c4abac6c1ae3b751fc15b6f42d2592d0ebe0146a2c088f305eade884b6d2/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e696d6167653f635f7b6f7d5c74696d65732673706163653b6d5f7b687d5c74696d65732673706163653b6d5f7b777d)](https://camo.githubusercontent.com/d7f7c4abac6c1ae3b751fc15b6f42d2592d0ebe0146a2c088f305eade884b6d2/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e696d6167653f635f7b6f7d5c74696d65732673706163653b6d5f7b687d5c74696d65732673706163653b6d5f7b777d)

[![Y_{i,:,:}=X\bigstar W_{i,:,:}\qquad for \quad i=1,...,c_{o}](https://camo.githubusercontent.com/ccb530df0e5be9267635cc5817021485ff677fd9b6be34b18bd5a249489a7da1/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e696d6167653f595f7b692c3a2c3a7d3d585c626967737461722673706163653b575f7b692c3a2c3a7d5c71717561642673706163653b666f722673706163653b5c717561642673706163653b693d312c2e2e2e2c635f7b6f7d)](https://camo.githubusercontent.com/ccb530df0e5be9267635cc5817021485ff677fd9b6be34b18bd5a249489a7da1/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e696d6167653f595f7b692c3a2c3a7d3d585c626967737461722673706163653b575f7b692c3a2c3a7d5c71717561642673706163653b666f722673706163653b5c717561642673706163653b693d312c2e2e2e2c635f7b6f7d)

多个输出通道：

```
def corr2d_multi_in_out(X, K):
    return torch.stack([corr2d_multi_in(X, k) for k in K], 0)
```

### 多个输入和输出通道

- 每个通道可以识别特定的模式

[![image](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/21/21-03.png)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/21/21-03.png)

- 输入通道核识别并组合输入中的模式

### 1X1卷积层

[![image](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/21/21-04.png)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/21/21-04.png)

### 二维卷积层

[![image](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/21/21-05.png)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/21/21-05.png)

### 总结

- 输出通道数是卷积层的超参数
- 每个输入通道有独立的二维卷积和，所有通道结果相加得到一个输出通道结果
- 每个输出通道有独立的三维卷积核

### 代码

多输入互相关运算

```
import torch
from d2l import torch as d2l

def corr2d_multi_in(X,K):
	return sum(cd2l.orr2d(x,k) for x,k in zip(X,K))
```

多通道互相关运算

 

```
def corr2d_multi_out(X,K):
	return torch.stack([d2l.torch(X,k) for k in K],0)
K = torch.stack((K,K+1,K+2),0)
```

1*1卷积

```
def corr2d_multi_in_out_1x1(X, K):
    c_i, h, w = X.shape
    c_o = K.shape[0]
    X = X.reshape((c_i, h * w))
    K = K.reshape((c_o, c_i))
    # 全连接层中的矩阵乘法
    Y = torch.matmul(K, X)
    return Y.reshape((c_o, h, w))
```





## 池化层

双重目的：降低卷积层对**位置**的敏感性，同时降低对**空间**降采样表示的敏感性。

### 最大汇聚层和平均汇聚层

#### 二维最大池化

返回滑动窗口中的最大值：保留最大信息特征

#### 平均池化层

将最大操作替换为平均，磨平/柔和操作

### 填充和步幅

池化层也具有填充和步幅，跟卷积相似，但是没有kernel的可学习参数

输出通道数=输入通道数，将通道融合交给了卷积层

### 代码

实现池化层的正向传播

```
import torch
from torch import nn
from d2l import torch as d2l

def pool2d(X, pool_size, mode='max'):
    p_h, p_w = pool_size
    Y = torch.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = X[i: i + p_h, j: j + p_w].max()
            elif mode == 'avg':
                Y[i, j] = X[i: i + p_h, j: j + p_w].mean()
    return Y
```





## 经典卷积神经网络LeNet

### LeNet卷积神经网络

#### 手写数字识别

- LeNet网络最早是为了应用于手写数字的识别应用。
- 应用背景：
  - 邮政局希望可以自动读出信件上的邮政编码
  - 人们希望可以用支票自动取钱
- 该模型在80年代末的银行被真正的部署

[![image](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/23/23-01.png)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/23/23-01.png)

#### MNIST

- LeNet所使用的数据集
- 50，000个训练数据
- 10，000个测试数据
- 图像大小为28*28
- 10类

[![image](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/23/23-02.png)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/23/23-02.png)

#### LeNet的具体模型



[![image](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/23/23-03.png)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/23/23-03.png)

#### 总结

- LeNet是早期成功的神经网络
- 先使用卷积层来学习图片空间信息
- 然后使用全连接层来转换到类别空间

### 代码部分

#### 定义网络结构和准备工作

- 导入所需的库

```
#导入所需的库
import torch
from torch import nn
from d2l import torch as d2l
```

- 定义网络结构（具体可参考上文“具体模型”的图）

```
#定义网络结构
net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),
    nn.AvgPool2d(2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),nn.Flatten(),
    nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.Sigmoid(),
    nn.Linear(84, 10))
```

- 查看每一层数据的变化情况

```
#把每一层数据的shape给打印出来
X = torch.rand(size=(1, 1, 28, 28), dtype=torch.float32)#创建符合要求的张量
for layer in net:
    X = layer(X)#通过每一层
    print(layer.__class__.__name__,'output shape: \t',X.shape)#打印
```

#### 模型训练

- 下载数据集

```
batch_size = 256 #批量大小
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)#下载或加载数据集，得到训练和测试集的迭代对象
```

- 使用GPU计算模型在数据集上的精度

```
def evaluate_accuracy_gpu(net, data_iter, device=None): #@save
    """使用GPU计算模型在数据集上的精度"""
    if isinstance(net, nn.Module):
        net.eval()  # 设置为评估模式
        if not device:
            device = next(iter(net.parameters())).device
    # 正确预测的数量，总预测的数量
    metric = d2l.Accumulator(2)#创建一个累加器，包含2个要累加的元素
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                # BERT微调所需的（之后将介绍）
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(d2l.accuracy(net(X), y), y.numel())#把每一组数据预测结果正确的个数和长度累加
    return metric[0] / metric[1]
```

- 训练函数

```
def train_ch6(net, train_iter, test_iter, num_epochs, lr, device): 
    """用GPU训练模型(在第六章定义)"""
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)#对linear类型的层用xavier初始化
    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])#动画需要
    timer, num_batches = d2l.Timer(), len(train_iter)
    for epoch in range(num_epochs):
        # 训练损失之和，训练准确率之和，范例数
        metric = d2l.Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()#梯度清零
            X, y = X.to(device), y.to(device)
            y_hat = net(X)#正向传播
            l = loss(y_hat, y)#计算损失
            l.backward()#反向传播
            optimizer.step()#梯度下降
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])#训练损失之和，训练准确率之和，范例数
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, train_acc, None))
        test_acc = evaluate_accuracy_gpu(net, test_iter)#评估测试集的精度
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')
```

- 运行

```
lr, num_epochs = 0.9, 10
train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
```







# 现代卷积神经网络

## AlexNet

### 历史

#### 2000 流行的机器学习方法——SVM，核方法

- 核方法替代了之前的神经网络网络方法，SVM对于调参不敏感，现在也有一些应用
- 本质上是特征提取，具体的方法是选择核函数来计算，把特征映射到高纬空间，使得他们线性可分
- 经过核函数计算之后，原问题可以转化为凸优化问题，这是2006年左右的研究热点
- 核方法有很多漂亮的定理，有很好的数学解释性
- 2010年左右，深度学习才兴起

#### 2000计算机视觉主要方法——几何学

- 首先还是对图片进行特征抽取
- 希望把计算机视觉问题描述成几何问题，建立（非）凸优化模型，可以得到很多漂亮的定理。
- 可以假设这是一个几何问题，假设这个假设被满足了，可以推出很好的效果

#### 2010计算机视觉的热点问题——特征工程

- 特征工程就是怎么抽取一张图片的特征，因为直接输入一张图片效果非常的差
- 特征描述子：SIFT,SURF

#### 硬件的发展奠定了深度学习的兴起

- 数据的增长，硬件的计算能力奠定了人们对于方法的选择

[![image](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/24/24-01.png)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/24/24-01.png)

#### ImageNet（2010）

[![image](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/24/24-02.png)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/24/24-02.png)

- AlexNet赢得了2012年ImageNet竞赛冠军

- 本质上是一个加强版的LeNet，更深更大

- AlexNet主要改进措施：

  - dropout（正则）
  - ReLu（梯度更大）
  - MaxPooling（取最大值，梯度相对增大）

- 影响：计算机视觉方法论的改变，从人工提取特征过渡到CNN学习特征

  [![image](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/24/24-03.png)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/24/24-03.png)

### AlexNet架构

[![image](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/24/24-04.png)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/24/24-04.png)

[![image](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/24/24-05.png)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/24/24-05.png)

[![image](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/24/24-06.png)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/24/24-06.png)

- 网络代码

```
import torch
from torch import nn
from d2l import torch as d2l

 net=nn.Sequential(
 	nn.Conv2d(1,96,kernel_size=11,stride=4,padding=1),
 	nn.ReLu()，
 	nn.MaxPool2d(pool_size=3,stride=2),
 	
 	nn.Conv2d(96,256,kernel_size=5,stride=1,padding=2),
 	nn.ReLu()，
 	nn.MaxPool2d(pool_size=3,stride=2),
 	
 	nn.Conv2d(256,384,kernel_size=3,stride=1,padding=1),
 	nn.ReLu()，
 	nn.Conv2d(384,384,kernel_size=3,stride=1,padding=1),
 	nn.ReLu()，
 	nn.Conv2d(384,384,kernel_size=3,stride=1,padding=1),
 	nn.ReLu()，
 	nn.MaxPool2d(pool_size=3,stride=2),
 	
 	nn.Linear(6400,4096),nn.ReLu(),nn.dropout(p=0.5),
 	nn.Linear(4096,4096),nn.ReLu(),nn.dropout(p=0.5),
 	nn.Linear(4096,1000)
 )
 
 X=torch.rand(1,1,224,224)
```

- 更多细节
  - 激活函数从sigmoid变成Relu，减缓梯度消失
  - 隐藏全连接层后加入了丢弃层（2个4096之后加入了dropout）
  - 数据增强，将一张图片进行变化，选取多个位置、光照之类的。

[![image](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/24/24-07.png)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/24/24-07.png)

- 复杂度对比
  - 参数个数增加，每次更新数据增加

[![image](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/24/24-08.png)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/24/24-08.png)

### 总结

- AlexNet 是更大更深的LeNet，10x参数个数，260x计算复杂度
- 新加入了dropout，relu，maxpooling，数据增强
- 标志着新一轮神经网络热潮开始了

## 使用块的网络VGG

### NiNVGG块

直到现在更深更大的模型也是我们努力的方向，在当时AlexNet比LeNet更深更大得到了更好的精度，大家也希望把网络做的更深更大。选择之一是使用更多的全连接层，但全连接层的成本很高；第二个选择是使用更多的卷积层，但**缺乏好的指导思想**来说明在哪加，加多少。最终**VGG采取了将卷积层组合成块**，再把**卷积块组合到一起**的思路。

VGG块可以看作是AlexNet思路的拓展，AlexNet中将三个相同的卷积层放在一起再加上一个池化层，而VGG将其拓展成可以使用任意个3x3，不改变输入大小的的卷积层，最后加上一个2x2的最大池化层。

[![25-01](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/25/25-01.PNG)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/25/25-01.PNG)

为什么选择3x3卷积呢？在计算量相同的情况下选用更大的卷积核涉及对网络会越浅，VGG作者经过实验发现用3x3卷积的效果要比5x5好，也就是说神经网络库深且窄的效果会更好。

### VGG架构

多个VGG块后接全连接层，不同次数的重复块得到不同的架构，如VGG-16, VGG-19等，后面的数字取决于网络层数。

可以讲VGG看作是将AlexNet中连续卷积的部分取出加以推广和复制，并删去了AlexNet中不那么规整的前几层。

[![25-02](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/25/25-02.PNG)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/25/25-02.PNG)

VGG较AlexNet相比性能有很大的提升，而代价是处理样本速度的降低和内存占用的增加。

### 总结

- VGG使用可重复使用的卷积块来构建深度卷积网络
- 不同卷积块个数和超参数可以得到不同复杂度的变种

这些思想影响了后面神经网络的设计，在之后的模型中被广泛使用。

### 代码

VGG块

```
import torch
from torch import nn
from d2l import torch as d2l

def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels,
                                kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
    return nn.Sequential(*layers)
    
conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))

def vgg(conv_arch):
    conv_blks = []
    in_channels = 1
    # 卷积层部分
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels

    return nn.Sequential(
        *conv_blks, nn.Flatten(),
        # 全连接层部分
        nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 10))

net = vgg(conv_arch)
```





## NiN-网络中的网络

### 动机

**全连接层的问题**

- **卷积层**需要的**参数较少**
- 而卷积层后的第一个**全连接层**的**参数较多**

[![image](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/26/26-01.png)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/26/26-01.png)

以VGG为例(图示)，全连接层需要先Flatten，输入维度为512x7x7，输出维度为4096，则需要参数个数为512x7x7x4096=102M。

### NiN块

- 核心思想：一个卷积层后面跟两个1x1的卷积层，后两层起到全连接层的作用。

[![image](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/26/26-02.png)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/26/26-02.png)

### NiN架构

- 无全连接层
- 交替使用NiN块和步幅为2的最大池化层
  - 逐步减小高宽和增大通道数
- 最后使用全局平均池化得到输出
  - 其输入通道是类别数

### NiN Networks

[![image](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/26/26-03.png)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/26/26-03.png)

NiN架构如上图右边所示，若干个NiN块(图示中为4个块)+池化层；前3个块后接最大池化层，最后一块连接一个全局平均池化层。

### 总结

- NiN块结构：使用卷积层加两个1x1卷积层
  - 后者对每个像素增加了非线性性
- NiN使用全局平均池化层来替代VGG和AlexNet中的全连接层
  - 不容易过拟合，更少的参数个数

### 代码

```
# 如果在Colab上跑, 或没有安装过d2l包, 需要最开始pip install d2l
!pip install git+https://github.com/d2l-ai/d2l-zh@release  # installing d2l
```

**NiN块**

```
import torch
from torch import nn
from d2l import torch as d2l

# 定义NiN块
def nin_block(in_channels, out_channels, kernel_size, strides, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding),
        nn.ReLU(), nn.Conv2d(out_channels, out_channels, kernel_size=1),
        nn.ReLU(), nn.Conv2d(out_channels, out_channels, kernel_size=1),
        nn.ReLU())
```

**NiN模型**

```
net = nn.Sequential(
    nin_block(1, 96, kernel_size=11, strides=4, padding=0),
    nn.MaxPool2d(3, stride=2),
    nin_block(96, 256, kernel_size=5, strides=1, padding=2),
    nn.MaxPool2d(3, stride=2),
    nin_block(256, 384, kernel_size=3, strides=1, padding=1),
    nn.MaxPool2d(3, stride=2), nn.Dropout(0.5),
    # 标签类别数是10
    nin_block(384, 10, kernel_size=3, strides=1, padding=1),
    nn.AdaptiveAvgPool2d((1, 1)),          #全局平均池化，高宽都变成1
    nn.Flatten())             #消掉最后两个维度, 变成(batch_size, 10)
```

**demo测试，查看每个块的输出情况**

```
X = torch.rand(size=(1, 1, 224, 224))
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape:\t', X.shape)
```

```
>>>
Sequential output shape:	 torch.Size([1, 96, 54, 54])
MaxPool2d output shape:		 torch.Size([1, 96, 26, 26])
Sequential output shape:	 torch.Size([1, 256, 26, 26])
MaxPool2d output shape:		 torch.Size([1, 256, 12, 12])
Sequential output shape:	 torch.Size([1, 384, 12, 12])
MaxPool2d output shape:		 torch.Size([1, 384, 5, 5])
Dropout output shape:		 torch.Size([1, 384, 5, 5])
Sequential output shape:	 torch.Size([1, 10, 5, 5])
AdaptiveAvgPool2d output shape:	 torch.Size([1, 10, 1, 1])
Flatten output shape:		 torch.Size([1, 10])
```

**训练模型**

```
lr, num_epochs, batch_size = 0.1, 10, 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
```

```
>>> <Figure size 252x180 with 1 Axes>
```





## GoogLeNet

### 含并行连结的网络

- GoogLeNet吸收了NiN中串联网络的思想，并在此基础上做了改进。我们往往不确定到底选取什么样的层效果更好，到底是3X3卷积层还是5X5的卷积层，诸如此类的问题是GooLeNet选择了另一种思路“小学生才做选择，我全都要”，这也使得GooLeNet成为了第一个模型中超过1000个层的模型。
- 选择各种卷积层的通道数量（输出图片的高宽是不会改变的）

### Inception块

- 在GoogLeNet中，基本的卷积块被称为*Inception块*（Inception block）

  [![截屏2022-01-23 上午10.11.18](https://github.com/kinza99/DeepLearning-MuLi-Notes/raw/main/imgs/27/27-1.png)](https://github.com/kinza99/DeepLearning-MuLi-Notes/blob/main/imgs/27/27-1.png)

- Inception块由四条并行路径组成。 前三条路径使用窗口大小为1×1、3×3和5×5的卷积层，从不同空间大小中提取信息。 中间的两条路径在输入上执行1×1卷积，以减少通道数，从而降低模型的复杂性。 第四条路径使用3×3最大汇聚层，然后使用1×1卷积层来改变通道数。 这四条路径都使用合适的填充来使输入与输出的高和宽一致，最后我们将每条线路的输出在通道维度上连结，并构成Inception块的输出。在Inception块中，通常调整的超参数是每层输出通道数。

- 我们发现路径的不同组合方式。单纯的卷积层；通道小的卷积+通道大的卷积；池化层+卷积层。

### GoogleNet模型

- GoogLeNet一共使用9个Inception块和全局平均汇聚层的堆叠来生成其估计值。Inception块之间的最大汇聚层可降低维度。 第一个模块类似于AlexNet和LeNet，Inception块的组合从VGG继承，全局平均汇聚层避免了在最后使用全连接层。[![截屏2022-01-23 上午10.17.11](https://github.com/kinza99/DeepLearning-MuLi-Notes/raw/main/imgs/27/27-2.png)](https://github.com/kinza99/DeepLearning-MuLi-Notes/blob/main/imgs/27/27-2.png)
- 第一个模块是7×7卷积层+池化。

![image-20240713230335077](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20240713230335077.png)

- 第二个模块使用两个卷积层：第一个卷积层是1×1卷积层；第二个卷积层使用s将通道数量增加三倍的3×3卷积层。 这对应于Inception块中的第二条路径。

![image-20240713230408146](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20240713230408146.png)

- 第三个模块串联两个完整的Inception块。 第一个Inception块的输出通道数为64+128+32+32=25664+128+32+32=256，四个路径之间的输出通道数量比为64:128:32:32=2:4:1:1。 第二个和第三个路径首先将输入通道的数量分别减少到96/192=1/296/192=1/2和16/192=1/1216/192=1/12，然后连接第二个卷积层。第二个Inception块的输出通道数增加到128+192+96+64=480，四个路径之间的输出通道数量比为128:192:96:64=4:6:3:2。 第二条和第三条路径首先将输入通道的数量分别减少到128/256=1/2128/256=1/2和32/256=1/832/256=1/8。

![image-20240713230710702](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20240713230710702.png)

- 第四模块更加复杂， 它串联了5个Inception块，其输出通道数分别是192+208+48+64=512、160+224+64+64=512、128+256+64+64=512、112+288+64+64=528和256+320+128+128=832256+320+128+128=832。 这些路径的通道数分配和第三模块中的类似，首先是含3×3卷积层的第二条路径输出最多通道，其次是仅含1×1卷积层的第一条路径，之后是含5×5卷积层的第三条路径和含3×3最大汇聚层的第四条路径。 其中第二、第三条路径都会先按比例减小通道数。 这些比例在各个Inception块中都略有不同。

### 总结

- Inception块相当于一个有4条路径的子网络。它通过不同窗口形状的卷积层和最大汇聚层来并行抽取信息，并使用1×1卷积层减少每像素级别上的通道维数从而降低模型复杂度。
- GoogLeNet将多个设计精细的Inception块与其他层（卷积层、全连接层）串联起来。其中Inception块的通道数分配之比是在ImageNet数据集上通过大量的实验得来的。
- GoogLeNet和它的后继者们一度是ImageNet上最有效的模型之一：它以较低的计算复杂度提供了类似的测试精度。





## 批量归一化

深层神经网络的训练，尤其是使网络在较短时间内收敛是十分困难的，**批量归一化[batch normalization]**是一种流行且有效的技术，能加速深层网络的收敛速度，目前仍被广泛使用。

### 训练深层网络时的问题-收敛困难



[![deep-model](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/28/deep_model.png)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/28/deep_model.png)

深度神经网络在训练时会遇到一些问题：

- 收敛速度慢：
  - 由于训练时先正向传播后反向传播，且每层的梯度一般较小，若网络较深，则反向传播时会出现类似于梯度消失的现象，导致距离数据更近的层梯度较小，收敛慢，而距离输出更近的层梯度较大，收敛快。然而底部的层一般都用于提取较基础的特征信息，上方的层收敛后，由于底部提取基础特征的层仍在变化（由于参数一直在随着每层梯度而不断更新，所以底部提取也在不断发生变化），上方的层一直在不停的重新训练，导致整个网络难以收敛，训练较慢。
- 内部协变量转移：
  - 分布偏移：偏移在视频课程中并未出现，但在《动手学深度学习》这本书中有提到过，在[4.9. 环境和分布偏移](https://zh-v2.d2l.ai/chapter_multilayer-perceptrons/environment.html)部分。偏移指的是训练数据可能和测试数据的分布不同，比如利用来自真实的猫和狗的照片的训练数据训练模型，然后让模型去预测动画中的猫和狗的图片。[![cat-dog-train](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/28/cat-dog-train.svg)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/28/cat-dog-train.svg)[![cat-dog-test](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/28/cat-dog-test.svg)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/28/cat-dog-test.svg)这显然会降低正确率也会对模型的进一步优化带来干扰。一般情况下对于分布偏移我们毫无办法，然而，在一些特定场景中，如果假定一些训练数据和测试数据分布的前提条件，就能对分布偏移进行处理，其中之一就是协变量偏移。
  - 协变量偏移：协变量偏移假设输入的分布可能随时间变化，但标签函数（条件分布$P(y|\bold x)$）没有改变。统计学家称这为*协变量偏移*（covariate shift）并给出了一些解决方案
  - **内部协变量偏移(Internal Covariate Shift)**：每一层的参数在更新过程中，会改变下一层输入的分布，导致网络参数变幻莫测，难以收敛，神经网络层数越多，表现得越明显。
  - 注意：
    - 1：内部协变量偏移这个词与标准的协变量偏移所有区别。
    - 2：能缓解内部协变量偏移仅仅是批量归一化的作者提出的假想，后续论文证实批量归一化实际对内部协变量偏移的缓解帮助不大
    - 3：批量归一化一般只影响模型的收敛速度，不影响精度
- 过拟合：
  - 由于网络深度加深，变得更为复杂，使得网络容易过拟合。

### 批量归一化

**批量归一化(batch normalization)**在 [[Ioffe & Szegedy, 2015\]](https://zh-v2.d2l.ai/chapter_references/zreferences.html#ioffe-szegedy-2015)中被提出，用于解决上述训练深度网络时的这些问题，然而这只是人们的感性理解，关于批量归一化具体是怎样帮助训练这个问题目前仍待进一步研究。

批量归一化尝试将每个训练中的mini-batch小批量数据（即会导致参数更新的数据）在每一层的结果进行归一化，使其更稳定，归一化指的是对于当前小批量中的所有样本，求出期望和方差，然后将每个样本减去期望再除以标准差。

#### 形式化表达-理论

下面的运算均为向量运算，向量中的每个维度代表一个特征，对于每个特征分别进行计算再拼接在一起即为向量运算。

设$ \bold x \in \mathcal{B}$为来自一个小批量$\mathcal{B}$的输入，批量规范化根据下式进行转换 $$ \mathrm{BN}(\mathbf{x}) = \boldsymbol{\gamma} \odot \frac{\mathbf{x} - \hat{\boldsymbol{\mu}}*\mathcal{B}}{\hat{\boldsymbol{\sigma}}*\mathcal{B}} + \boldsymbol{\beta}. $$ 式中$\hat{\boldsymbol{\mu}}*\mathcal{B}$为小批量$\mathcal{B}$样本均值，$\hat{\boldsymbol{\sigma}}*\mathcal{B}$为样本标准差： $$ \begin{split}\begin{aligned} \hat{\boldsymbol{\mu}}*\mathcal{B} &= \frac{1}{|\mathcal{B}|} \sum*{\mathbf{x} \in \mathcal{B}} \mathbf{x},\ \hat{\boldsymbol{\sigma}}*\mathcal{B}^2 &= \frac{1}{|\mathcal{B}|} \sum*{\mathbf{x} \in \mathcal{B}} (\mathbf{x} - \hat{\boldsymbol{\mu}}_{\mathcal{B}})^2 + \epsilon\end{aligned}\end{split} $$ 其中$\epsilon$用于防止分母为0，经过减期望与除以标准差后得到期望为1方差为0的小批量数据。然而，期望和方差为了使小批量有更自由的选择，再将其乘拉伸参数$\boldsymbol {\gamma}$，加偏移参数$\boldsymbol \beta$，这两个参数与$\bold x$同样大小，是模型中的可学习参数，与其他参数一同更新。

由于$\hat{\boldsymbol{\mu}}*\mathcal{B}$和$\hat{\boldsymbol{\sigma}}*\mathcal{B}$为由当前小批量计算的值，实际上是整个分布对应的期望与标准差的估计值，由于小批量的随机选择，$\hat{\boldsymbol{\mu}}*\mathcal{B}$和$\hat{\boldsymbol{\sigma}}*\mathcal{B}$会给模型带来一定的与输入数据有关的噪音，而这些噪音也能对模型进行正则化，防止过拟合。为何这种噪音能加快训练并带来正则化还有待研究，不过已有理论说明了为什么批量规范化最适应$50∼100$范围中的中等批量大小的问题。

训练时不能使用整个数据集，只能一步步的训练和更新；而预测时模型已然固定，可以根据整个数据集精确计算均值和方差。因此，批量归一化对于训练和预测时有两种不同模式。

#### 批量归一化层

批量归一化不再单独的考虑单个样本，需要对整个mini-batch进行，因此需要考虑多种情况。

##### 全连接层

通常，我们将批量规范化层置于全连接层中的仿射变换和激活函数之间。如下： $$ \mathbf{h} = \phi(\mathrm{BN}(\mathbf{W}\mathbf{x} + \mathbf{b})) $$

##### 卷积层

在卷积层中，我们将通道视作每个位置的特征，将每个样本中的每个位置视作一个样本进行计算。每个通道都有着自己的拉伸参数${\gamma}$和偏移参数$\beta$，所有通道加在一起组成了拉伸参数向量$\boldsymbol {\gamma}$和偏移参数向量$\boldsymbol \beta$，若样本数为m，卷积输出为p*q，计算时对m*p*q个向量进行批量归一化运算（即视作有m*p*q个样本）

##### 预测过程中的批量归一化

在训练过程中，我们需要不断地更新模型，方差和均值也就在不断地变化，就必须计算当前小批量数据对应的方差和均值，然而预测时我们的模型已经确定下来，可以用在整个训练数据集上得到的均值和方差来对预测时的结果进行归一化。

### 代码实现

#### 实现细节-从零实现

- 在实际实现时，一般使用指数加权平均来更新小批量的均值和方差，指数加权平均将旧值和当前计算结果不断进行加权平均，最终做到平滑的向更新值靠拢，公式如下：
- 𝑆𝑡={𝑌1,𝑡=1 𝛽𝑆𝑡−1+(1−𝛽)𝑌𝑡,𝑡>1
- 批量归一化的参数可以通过动量梯度下降，RMSProp，Adam等多种优化方法进行训练。

```
import torch
from torch import nn
from d2l import torch as d2l

# 输入 可学习的两个参数 全局的均值 全局的方差（整个数据集的而非小批量的） eps防止除零 momentum更新均值和方差
def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    # 通过is_grad_enabled来判断当前模式是训练模式还是预测模式
    if not torch.is_grad_enabled():
        # 如果是在预测模式下，直接使用传入的移动平均所得的均值和方差
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            # 使用全连接层的情况，计算特征维上的均值和方差
            mean = X.mean(dim=0)
            var = ((X - mean) ** 2).mean(dim=0)
        else:
            # 使用二维卷积层的情况，计算通道维上（axis=1）的均值和方差。
            # 这里我们需要保持X的形状以便后面可以做广播运算
            mean = X.mean(dim=(0, 2, 3), keepdim=True)
            var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
        # 训练模式下，用当前的均值和方差做标准化
        X_hat = (X - mean) / torch.sqrt(var + eps)
        # 更新移动平均的均值和方差
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
    Y = gamma * X_hat + beta  # 缩放和移位
    return Y, moving_mean.data, moving_var.data
```

创建一个正确的BatchNorm的图层

```
class BatchNorm(nn.Module):
    # num_features：完全连接层的输出数量或卷积层的输出通道数。
    # num_dims：2表示完全连接层，4表示卷积层
    def __init__(self, num_features, num_dims):
        super().__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        # 参与求梯度和迭代的拉伸和偏移参数，分别初始化成1和0
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        # 非模型参数的变量初始化为0和1
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)
        
    def forward(self, X):
        # 如果X不在内存上，将moving_mean和moving_var
        # 复制到X所在显存上
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        # 保存更新过的moving_mean和moving_var
        Y, self.moving_mean, self.moving_var = batch_norm(
            X, self.gamma, self.beta, self.moving_mean,
            self.moving_var, eps=1e-5, momentum=0.9)
        return Y
```

应用：应用到LeNet模型上

批量规范化是在卷积层或全连接层之后、相应的激活函数之前应用的

```
net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5), 
    BatchNorm(6, num_dims=4), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    
    nn.Conv2d(6, 16, kernel_size=5), 
    BatchNorm(16, num_dims=4), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2), 
    
    nn.Flatten(),
    nn.Linear(16*4*4, 120), 
    BatchNorm(120, num_dims=2), nn.Sigmoid(),
    
    nn.Linear(120, 84), 
    BatchNorm(84, num_dims=2), nn.Sigmoid(),
    
    nn.Linear(84, 10))
```

#### 实现细节-调包

```
net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5), nn.BatchNorm2d(6), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), nn.BatchNorm2d(16), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(),
    nn.Linear(256, 120), nn.BatchNorm1d(120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.BatchNorm1d(84), nn.Sigmoid(),
    nn.Linear(84, 10))
```

### 吴恩达老师深度学习课程中的批量归一化

吴恩达老师深度学习课程中的批量归一化中的部分内容与本课程有所出入，考虑到批量归一化这部分内容还没有精确的理论解释，目前的认识仅限于直觉，故将两课程中的区别即补充罗列在此作为参考：

- 关于dropout：
  - 本课中提到批量归一化有正则化效果，无需再进行dropout
  - 吴恩达老师课程中提到批量归一化正则化效果较差，不能作为正则化的手段，必要时需要dropout
- 对于线性层（包括其他带有偏置项的层）后的批量归一化，由于归一化时减去了均值，偏置项被消掉，可以省略归一化层之前的偏置项
- 标准化的输入能使梯度下降加快，批归一化能使得每层的输入都被归一化，这也是训练更快的原因之一
- 批量归一化可以使得不同层之间互相的影响减少，从而应对数据偏移，增强鲁棒性。







## 残差网络ResNet

随着我们设计越来越深的网络，深刻理解“新添加的层如何提升神经网络的性能”变得至关重要。更重要的是设计网络的能力，在ResNet这种网络中，添加层会使网络更具表现力

#### 函数类

- 假设有一类特定的神经网络架构F，它包括学习速率和其他超参数设置。 对于所有*f*∈F，存在一些参数集（例如权重和偏置），这些参数可以通过在合适的数据集上进行训练而获得。 现在假设*f*∗是我们真正想要找到的函数，如果是*f*∗∈F，那我们可以轻而易举的训练得到它，但通常我们不会那么幸运。 相反，我们将尝试找到一个函数*f*∗，这是我们在F中的最佳选择。
- 为了得到更近似真正*f*∗的函数我们需要设计一个更强大的架构F'，但是如果先前的框架F不包含于新框架F‘中就可能导致如下图中左侧的最优函数离实际预测函数误差反而随框架边强而增大，这不是我们期望的结果，所以我们选择使用下图中右侧的嵌套函数类以解决这个问题
- 引入方法：对于深度神经网络，如果我们能将新添加的层训练成*恒等映射*（identity function）*f*(**x**)=**x**，新模型和原模型将同样有效。 同时，由于新模型可能得出更优的解来拟合训练数据集，因此添加层似乎更容易降低训练误差。

[![image](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/29/29-01.png)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/29/29-01.png)

[![image](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/29/29-02.png)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/29/29-02.png)

#### 残差块

- 神经网络中的具体实现：假设我们的原始输入为*x*，而希望学出的理想映射为*f*(**x**)，左图虚线框中的部分需要直接拟合出该映射*f*(**x**)，而右图虚线框中的部分则需要拟合出残差映射*f*(**x**)−**x**。而右图正是ResNet的基础架构–*残差块*（residual block）
- 残差块的代码实现：

```
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l


class Residual(nn.Module):
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        # 第一个卷积层
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        # 第二个卷积层
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        # 如果使用1 x 1卷积以使得输入变换成需要的形状
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        # 对应第一个卷积层的批量规范化层
        self.bn1 = nn.BatchNorm2d(num_channels)
        # 对应第二个卷积层的批量规范化层
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        # 第一层：卷积 -> 规范化 -> relu激活
        Y = F.relu(self.bn1(self.conv1(X)))
        # 第二层：卷积 -> 规范化
        Y = self.bn2(self.conv2(Y))
        # 如果要让输入变换成需要的形状
        if self.conv3:
            # 对X使用1 x 1卷积，以使输出成为需要的形状
            X = self.conv3(X)
        # 嵌套模型的实现，即对上一次训练后的模型进行嵌套
        Y += X
        # relu激活并输出
        return F.relu(Y)
```



[![image](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/29/29-03.png)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/29/29-03.png)

#### ResNet模型

- ResNet的前两层跟之前介绍的GoogLeNet中的一样： 在输出通道数为64、步幅为2的7×7卷积层后，接步幅为2的3×3的最大汇聚层。 不同之处在于ResNet每个卷积层后增加了批量规范化层。

```
b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                   nn.BatchNorm2d(64), nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
```

- GoogLeNet在后面接了4个由Inception块组成的模块。 ResNet则使用4个由残差块组成的模块，每个模块使用若干个同样输出通道数的残差块。 第一个模块的通道数同输入通道数一致。 由于之前已经使用了步幅为2的最大汇聚层，所以无须减小高和宽。 之后的每个模块在第一个残差块里将上一个模块的通道数翻倍，并将高和宽减半。

```
def resnet_block(input_channels, num_channels, num_residuals,
                 first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels,
                                use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk
```



- 接着在ResNet加入所有残差块，这里每个模块使用2个残差块。

```
b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
b3 = nn.Sequential(*resnet_block(64, 128, 2))
b4 = nn.Sequential(*resnet_block(128, 256, 2))
b5 = nn.Sequential(*resnet_block(256, 512, 2))
```

- 最后，与GoogLeNet一样，在ResNet中加入全局平均汇聚层，以及全连接层输出。

```
net = nn.Sequential(b1, b2, b3, b4, b5,
                    nn.AdaptiveAvgPool2d((1,1)),
                    nn.Flatten(), nn.Linear(512, 10))
```



- 每个模块有4个卷积层（不包括恒等映射的1×1卷积层）。 加上第一个7×7卷积层和最后一个全连接层，共有18层。 因此，这种模型通常被称为ResNet-18。 通过配置不同的通道数和模块里的残差块数可以得到不同的ResNet模型，例如更深的含152层的ResNet-152。 虽然ResNet的主体架构跟GoogLeNet类似，但ResNet架构更简单，修改也更方便。这些因素都导致了ResNet迅速被广泛使用。

[![image](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/29/29-04.png)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/29/29-04.png)



#### 总结

- 学习嵌套函数（nested function）是训练神经网络的理想情况。在深层神经网络中，学习另一层作为恒等映射（identity function）较容易（尽管这是一个极端情况）。
- 残差映射可以更容易地学习同一函数，例如将权重层中的参数近似为零。
- 利用残差块（residual blocks）可以训练出一个有效的深层神经网络：输入可以通过层间的残余连接更快地向前传播。
- 残差网络（ResNet）对随后的深层神经网络设计产生了深远影响。







# 硬件处理

## CPU和GPU

### CPU：

#### 提升CPU利用率一

- 在计算a+b之前，需要准备数据
- 主内存->L3->L2->L1->寄存器
  - L1访问延时：0.5ms
  - L2访问延时：7ns（14XL1）
  - 主内存访问延时：100ns(200XL1)
- 提升空间和时间的内存本地性
  - 时间：重用数据使它们在缓存里
  - 空间：按序读写数据是的可以预读取

![image-20240719144709106](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20240719144709106.png)

##### 样例分析

- 如果一个矩阵是按行存储，访问一行比访问一列要快
  - CPU一次读取64字节（缓存线）
  - CPU会“聪明的”提前读取下一个（缓存线）

[![image](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/31/31-01.png)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/31/31-01.png)

#### 提升CPU利用率二：

- 高端CPU有几十个核
  - EC2 P3.16xlarge:2 Intel Xeon CPUs,32物理核
- 并行来利用所用核
  - 超线程不一定提升性能，因为他们共享寄存器

##### 样例分析：

- 左边比右边慢（python）

[![image](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/31/31-02.png)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/31/31-02.png)

- 左边调用n次函数，每次调用有开销
- 右边很容易被并行（例如下面的C++实现）

[![image](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/31/31-03.png)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/31/31-03.png)

### CPU vs GPU:

[![image](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/31/31-04.png)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/31/31-04.png)

#### 提升GPU利用率

- 并行
  - 使用数千个线程
- 内存本地性
  - 缓存更小，架构更简单
- 少用控制语句
  - 支持有限
  - 同步开销大

#### CPU/GPU 带宽

[![image](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/31/31-05.png)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/31/31-05.png)

#### 更多的CPUs和GPUs

- CPU:AMD,ARM
- GPU:AMD,Intel,ARM,Qualcomm...

#### CPU/GPU高性能计算编程

- CPU：C++或者任何高性能语言
  - 编译器成熟
- GPU
  - Nvida上用CUDA
    - 编译器和驱动成熟
  - 其他用OpenCL
    - 质量取决于硬件厂商

### 总结

- CPU:可以处理通用计算。性能优化考虑数据读写效率和多线程
- GPU：使用更多的小核和更好的内存带宽，适合能大规模并行的计算任务

## 深度学习硬件

![image-20240719145059087](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20240719145059087.png)

### DSP:数字信号处理（Digital Signal Process）

- 为数字信号处理算法设计：点积、卷积、FFT
- 低功耗，高性能
  - 比移动GPU快5倍，功耗更低
- VLIW：very long instruction word
  - 频率低，核少，但是一条指令可以进行上百次的累加，便于重复
- 缺点：编程和调试困难，编译器良莠不齐（做的人少，工具不是很好用）

### 可编程阵列（FPGA）

- 有大量的可以用来编程的逻辑单元和可配置链接
- 可以配置成计算复杂函数
  - 编程语言：VHDL Verilog
- 通常比通用硬件更高效，但是体积更大不方便
- 缺点：工具链质量良莠不齐，一次编译需要数个小时（烧一次板子，物理上的改变）
- 用途：主要用来模拟，看看效果好不好，如果好可以进一步造芯片

### AI ASIC

- 深度学习热门领域（针对特定领域）
  
  - 大公司都在造自己的芯片（Intel Qualcomm Google Amazon Facebook）
- Google TPU 是标ss志性芯片（听说在Google内部已经盛行 取代GPU了）
  - 能够媲美 Nvidia GPU性能
  - 在Google 大量部署
  - 核心是 systolic array（时间快 便宜）（专门用来针对大矩阵运算）
- systolic array
  - 计算单元（PE）阵列
  - 特别适合做矩阵乘法
  - 设计和制造相对简单（核少）

  ![image-20240719150440300](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20240719150440300.png)
- 矩阵乘法例子：
  - 对于一般的矩阵乘法：通过切开、填充来匹配SA大小
  - 批量输入来降低延迟（避免空等，先出的硬件空闲）
  - 通常有其他硬件单元来处理别的NN操作子，例如激活层
- 缺点：只针对深度学习这方面有用，别的方面效果不大 





## 单机多卡并行



一台机器可以安装多个GPU（一般为1-16个），在训练和预测时可以将一个小批量计算切分到多个GPU上来达到加速目的，常用的切分方案有数据并行，模型并行，通道并行。

### 数据并行

将小批量的数据分为n块，每个GPU拿到完整的参数，对这一块的数据进行前向传播与反向传播，计算梯度。

数据并行通常性能比模型并行更好，因为对数据进行划分使得各个GPU的计算内容更加均匀。

#### 数据并行的大致流程



[![DataParallel](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/33/DataParallel.png)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/33/DataParallel.png)

主要分为五部

- 1：每个GPU读取一个数据块（灰色部分）
- 2：每个GPU读取当前模型的参数（橙色部分）
- 3：每个GPU计算自己拿到数据块的梯度（绿色部分）
- 4：GPU将计算得到的梯度传给内存（CPU）（绿色箭头）
- 5：利用梯度对模型参数进行更新（橙色箭头）

数据并行并行性较好，主要因为当每个GPU拿到的数据量相同时计算量也相似，各个GPU的运算时间相近，性能较好

### 模型并行

将整个模型分为n个部分，每个GPU拿到这个部分的参数和负责上一个部分的GPU的输出作为输入来进行计算，s反向传播同理。

模型并行通常用于模型十分巨大，参数众多，即使在每个mini-batch只有一个样本的情况下单个GPU的显存仍然不够的情况，但并行性较差，可能有时会有GPU处于等待状态。

### 通道并行

通道并行是数据并行和模型并行同时进行

### 总结

- 当一个模型能用单卡计算时，通常使用数据并行扩展到多卡
- 模型并行则用在超大模型上



## 分布式训练

### 分布式计算

- 本质上来说和之前讲的单机多卡并行没有区别。二者之间的区别是分布式计算是通过网络把数据从一台机器搬到另一台机器
- 样本分布在不同磁盘上，允许所有机器进行读取s
- 机器/worker/栈

[![image](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/35/35-01.png)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/35/35-01.png)

### GPU机器架构

![image-20240719175039729](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20240719175039729.png)

- 总的来说，gpu到gpu的通讯是很快的，gpu到cpu慢一点。机器到机器更慢。因而总体性能的关键就是尽量在本地做通讯而少在机器之间做通讯S

##### 样例：计算一个小批量（如何减少跨机器的学习）

- 样本分割：每个worker从参数服务器那里获取模型参数：首先把样本复制到机器的内存，然后把样本分到每个gpu上
- 复制参数到每个gpu上：同样，先把每一次的参数放到内存里，然后再复制到每个gpu上
- 每个gpu计算梯度
- 再主内存上把所有gpu上的梯度加起来（主内存整合）
- 梯度从主内存传回服务器
- 每个服务器对梯度求和，并更新参数，之后进行下一轮的计算

### 关于性能

##### 对于**同步SGD**：

- 这里每个worker都是同步计算一个批量，称为同步SGD
- 假设有n个ggpu，每个gpu每次处理b个样本，那么同步SGD等价于再单gpu运行批量大小为nb的SGD
- 在理想情况下，n个gpu可以得到相对单gpu的n倍加速（同步且可以放入n倍batch）

##### **性能**：

- t1 = 在单gpu上计算b个样本梯度时间
- 假设有m个参数，一个worker每次发送和接受m个参数、梯度
  - t2 = 发送和接受所用时间
- 每个批量的计算时间为max（t1，t2） ->发送接收与计算两个是并行的
  - 选取足够大的b使t1>t2：通信时间大则存在性能问题（GPU在等待数据）
  - 增加b或n导致更大的批量 大小，当值需要更多计算来得到给定的模型精度

##### 性能的权衡

[![image](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/35/35-02.png)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/35/35-02.png)

对于系统性能而言，GPU的批量大小变大，可以更好的利用GPU的性能；

对于训练有效性而言，GPU批量变大需要更多epoch来训练精度，来应对收敛速度变慢。

### 实践时的建议

- 使用一个大数据集
- 需要好的gpu-gpu和机器-机器带宽
- 高效的数据读取和预处理
- 模型需要有好的计算和通讯比，flop/model size越大越好
  - Inception>ResNet>AlexNet
- 使用足够大的批量大小来得到更好的系统性能
- 使用高效的优化算法对应大批量大小

### 总结

- 分布式同步数据并行是多gpu数据并行在多机器上的拓展
- 网络通讯通常是瓶颈
- 需要注意使用特别大的批量大小时的收敛效率
- 更复杂的分布式有异步、模型并行（这里没有介绍）



# 图像处理技术

## 数据增广

### 使用增强数据训练

常见的问题：采集数据得到的训练场景与实际部署场景。-->增强模型泛化性

数据增强是指在一个已有数据集上操作使其有更多的多样性。

​	对语音来说可以加入不同的背景噪音

​	对图片而言可以改变其颜色，形状等。

数据增广往往是在线生成，并且是随机数据生成。通常只在训练时做数据增强而测试时不用。可以将数据增强理解为一个正则项。

### 增强手段

#### 翻转

一些例子：左右翻转，上下翻转（根据模型特性）

要注意不是所有增强策略都总是可行，如建筑图片上下翻转就不太合适，而之前的树叶分类竞赛中的树叶图片就没关系。

[![36-01](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/36/36-01.png)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/36/36-01.png)

#### 切割

从图片中切割一块，变形到固定形状。

**具体随机做法：**

随机取一个高宽比

随机取图片大小（切下部分占原图的百分数）

随机取位置

[![36-02](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/36/36-02.png)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/36/36-02.png)

#### 颜色

改变色调，饱和度，明亮度。

[![36-03](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/36/36-03.png)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/36/36-03.png)

#### 其他

高斯模糊，部分像素变黑，图片变形，锐化等等。

理论上讲Photoshop能做到的都可以用作图片数据增强。

如果测试集中有类似的效果那么相应的数据增广手段会更有效。

[![36-04](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/36/36-04.png)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/36/36-04.png)

### 总结

- 数据增广通过变形数据来获取多样性从而使得模型泛化性能更好
- 常见图片增广包括翻转，切割，变色

### 代码实现

transforms内部分多种翻转函数

```
#垂直水平翻转
torchvision.transforms.RandomHorizontalFlip()
torchvision.transforms.RandomVerticalFlip()

# 随即裁剪：最终裁剪大小，随机保留百分比，随机高宽比
torchvision.transforms.RandomResizedCrop((200,200),scale=(0.1,1),ratio=(0.5,2))

# brightness（亮度）,contrast（对比度）,saturation（饱和度）,hue（色调）
# 随即改变亮度：第一个数值随即改变上下亮度50%
torchvision.transforms.ColorJitter(brightness=0.5,contrast=0,saturation=0,hue=0)
# 随即改变色调：第四个数值随即改变上下亮度50%
torchvision.transforms.ColorJitter(brightness=0,contrast=0,saturation=0,hue=0.5)

# 组合多种图像增广
torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(),color_aug,shape_aug])
```

具体使用方式

```
train_aug = torchvision.transforms.Compose([
	torchvision.transforms.RandomHorizontalFlip(),
	torchvision.transforms.ToTensor()])
	
test_aug = torchvision.transforms.Compose([
	torchvision.transforms.ToTensor()])
```

定义一个辅助函数，用于读取图像和应用图像增广

```
def load_cifar10(is_train,augs,batch_size):
	# 每次读取进去进行一次泛化
	dataset = torchvision.datasets.CIFAR10(
		root="../data".train=is_train,
		transform=augs,download=True)
	dataloader = torch.utils.data.DataLoader(
		dataset,batch_size=batch_size,shuffle=is_train,
		num_worker=4)
	return dataloader
```

### QA知识点

1.理论上，在原始数据足够多的时候可以不使用增广，但是理想上原始数据无法抵达真实场景的多样性。





## 微调

### 背景

由于训练样本数量有限，训练模型的准确性可能无法满足实际要求，解决方法：

- 收集更多的数据，也意味着需要大量的时间和金钱。 
- 使用迁移学习将从*源数据集*学到的知识迁移到*目标数据集*。 例如，尽管ImageNet数据集中的大多数图像与椅子无关，但在此数据集上训练的模型可能会提取更通用的图像特征，这有助于识别边缘、纹理、形状和对象组合。 这些类似的特征也可能有效地识别椅子。

### 步骤

1. ***源模型***（pre-train）：在源数据集（例如ImageNet数据集）上预训练神经网络模型。

2. 创建一个新的神经网络模型，即***目标模型***：

   ​		复制源模型上的所有模型设计及其参数（输出层除外）

   ​		我们假定这些模型参数包含从源数据集中学到的知识，这些知识也将适用于目标数据集。我们还假设源模型的输出层与源数据集的标签密切相关；因此不在目标模型中使用该层。

3. 向目标模型添加输出层，其输出数是目标数据集中的类别数。然后随机初始化该层的模型参数。

4. 在目标数据集（如椅子数据集）上训练目标模型。输出层将从头开始进行训练，而所有其他层的参数将根据源模型的参数进行微调。

[![image](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/37/37-01.png)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/37/37-01.png)

### 技巧

##### 对于训练

**使用更强的正则化，选择更小的学习率，选择更小的迭代。**

对于不同的任务往往只需要改变最后输出层，这一层随机初始化即可。

##### 固定一些层

前面的层更加通用/底层，固定底部一些层的参数，不参与更新；

后面的层与我们的label更加相关，所以对这部分进行修改。

### 总结

- 迁移学习：将从源数据集中学到的知识“迁移”到目标数据集，微调是迁移学习的常见技巧。
- 除输出层外，目标模型从源模型中复制所有模型设计及其参数，并根据目标数据集对这些参数进行微调。但是，目标模型的输出层需要从头开始训练。
- 通常，微调参数使用较小的学习率，而从头开始训练输出层可以使用更大的学习率。

### 代码

定义和初始化模型

```
# pretrained=True：拿走模型及模型训练好的参数
pretrained_net = torchvision.models.resnet18(pretrained=True)

# from pretrained_net.fc we can get in_features/out_features
finetune_net = torchvision.models.resnet18(pretrained=True)
finetune_net.fc = nn.Linear(pretrained_net.fc.in_features,2)

# 只对最后一层做随机初始化
nn.init.xavier_uniform_(finetune_net.fc.weight)
```

微调模型

```
# 提取不包括全连接层权重和偏置的参数
param_1x = [
 	param for name,param in net.named_parameters()
 	if name not in ["fc.weight","fc.bias"]]
 	
# 检查是否有需要应用不同学习率的参数
	if param_group:
	
		params=[
		{'params': param_1x, 'lr': learning_rate * 10},
    	{'params': [param for name, param in net.named_parameters() if name in ["fc.weight", "fc.bias"]], 'lr': learning_rate}
    	]
 		trainer = torch.potim.SGD(
 		params,lr=learning_rate,weight_decay=0.001)
 		
 	else:
		trainer = torch.potim.SGD(net.parameters(),
		lr=learning_rate,weight_decay=0.001)
 		
```





## 物体检测和数据集

#### 物体检测

###### 图片分类和目标检测在任务上的区别

图片分类已知有一个确定目标，任务是**识别该目标属于何种分类**；

目标检测不仅需要检测出图片中所有感兴趣的目标类别，并确定其**位置**，所以目标检测要比图片分类更复杂应用场景更广。

###### 图片分类和目标检测在数据集上的区别

由于目标检测中每一张图片可能存在多个目标，每个目标我们不仅需要分类，还需要确定边缘框以给出目标位置信息，因此**目标检测数据集的标注成本要显著高于图片分类**，也就导致了目标检测数据集较小。

###### 边缘框

用一个尽量小矩形框将目标物体大体框起来，边框的位置信息就可以表示目标位置在图片中的位置信息，常见的边缘框有两种表示方法：

- （左上x，左上y，右下x，右下y）
- （左上x，左上y，宽，高）

1. 目标检测数据集的常见表示：

   每一行表示一个物体：“图片文件名，物体类别，边缘框”

   由于边缘框用4个数值表示，因此对于每一行的那一个物体而言，需要用6个数值表示。

2. 目标检测领域常用数据集：

   COCO（80类物体，330K图片，所有图片共标注1.5M物体）

#### 边缘框实现

**目标的位置**

在图像分类任务中，我们假设图像中只有一个主要物体对象，我们只关注如何识别其类别。 然而，很多时候图像里有多个我们感兴趣的目标，我们不仅想知道它们的类别，还想得到它们在图像中的具体位置。 

在计算机视觉里，我们将这类任务称为***目标检测*（object detection）或*目标识别*（object recognition）。**目标检测在多个领域中被广泛使用。 例如，在无人驾驶里，我们需要通过识别拍摄到的视频图像里的车辆、行人、道路和障碍物的位置来规划行进线路。 机器人也常通过该任务来检测感兴趣的目标。安防领域则需要检测异常目标，如歹徒或者炸弹。

```
%matplotlib inline
import torch
from d2l import torch as d2l

# 下面加载本节将使用的示例图像。可以看到图像左边是一只狗，右边是一只猫。 它们是这张图像里的两个主要目标。

d2l.set_figsize()
img = d2l.plt.imread('../img/catdog.jpg')
d2l.plt.imshow(img);
```

**边界框**

- 在目标检测中，我们通常使用***边界框*（bounding box）来描述对象的空间位置**。 边界框是矩形的，由矩形左上角的以及右下角的*x*和*y*坐标决定。 另一种常用的边界框表示方法是边界框中心的(*x*,*y*)轴坐标以及框的宽度和高度。

- 在这里，我们定义在这两种表示法之间进行转换的函数：`box_corner_to_center`从两角表示法转换为中心宽度表示法

  `box_center_to_corner`反之亦然。 

  输入参数`boxes`可以是长度为4的张量，也可以是形状为（*n*，4）的二维张量，其中*n*是边界框的数量。

```
# 
def box_corner_to_center(boxes):
    """从（左上，右下）转换到（中间，宽度，高度）"""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    boxes = torch.stack((cx, cy, w, h), axis=-1)
    return boxes

#@save
def box_center_to_corner(boxes):
    """从（中间，宽度，高度）转换到（左上，右下）"""
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    boxes = torch.stack((x1, y1, x2, y2), axis=-1)
    return boxes
```

- 我们将根据坐标信息定义图像中狗和猫的边界框。 图像中坐标的原点是图像的左上角，向右的方向为*x*轴的正方向，向下的方向为*y*轴的正方向。

```
# bbox是边界框的英文缩写
dog_bbox, cat_bbox = [60.0, 45.0, 378.0, 516.0], [400.0, 112.0, 655.0, 493.0]
```

- 我们可以将边界框在图中画出，以检查其是否准确。 画之前，我们定义一个辅助函数`bbox_to_rect`。 它将边界框表示成`matplotlib`的边界框格式。

```
#@save
def bbox_to_rect(bbox, color):
    # 将边界框(左上x,左上y,右下x,右下y)格式转换成matplotlib格式：
    # ((左上x,左上y),宽,高)
    return d2l.plt.Rectangle(
        xy=(bbox[0], bbox[1]), width=bbox[2]-bbox[0], height=bbox[3]-bbox[1],
        fill=False, edgecolor=color, linewidth=2)
```

- 在图像上添加边界框之后，我们可以看到两个物体的主要轮廓基本上在两个框内。

```
fig = d2l.plt.imshow(img)
fig.axes.add_patch(bbox_to_rect(dog_bbox, 'blue'))
fig.axes.add_patch(bbox_to_rect(cat_bbox, 'red'));
```

##### 小结

- 目标检测不仅可以识别图像中所有感兴趣的物体，还能识别它们的位置，该位置通常由矩形边界框表示。
- 我们可以在两种常用的边界框表示（中间，宽度，高度）和（左上，右下）坐标之间进行转换。

#### 数据集

目标检测领域没有像MNIST和Fashion-MNIST那样的小数据集。 为了快速测试目标检测模型，我们收集并标记了一个小型数据集。 首先，我们拍摄了一组香蕉的照片，并生成了1000张不同角度和大小的香蕉图像。 然后，我们在一些背景图片的随机位置上放一张香蕉的图像。 最后，我们在图片上为这些香蕉标记了边界框。

1. 下载数据集

- 包含所有图像和CSV标签文件的香蕉检测数据集可以直接从互联网下载。

```
import os
import pandas as pd
from mxnet import gluon, image, np, npx
from d2l import mxnet as d2l

# set_np()初始化numpy的兼容格式
npx.set_np()


d2l.DATA_HUB['banana-detection'] = (
    d2l.DATA_URL + 'banana-detection.zip',
    '5de26c8fce5ccdea9f91267273464dc968d20d72')
```

读取数据集

- 通过`read_data_bananas`函数，我们读取香蕉检测数据集。 该数据集包括一个的CSV文件，内含目标类别标签和位于左上角和右下角的真实边界框坐标。

```
def read_data_bananas(is_train=True):
    """读取香蕉检测数据集中的图像和标签"""
    # 下载并自动解压数据集
    data_dir = d2l.download_extract('banana-detection')
    
    # is_train 参数（决定是读取训练数据还是验证数据）
    csv_fname = os.path.join(data_dir, 'bananas_train' if is_train else 'bananas_val', 'label.csv')
    
    
    csv_data = pd.read_csv(csv_fname)
    csv_data = csv_data.set_index('img_name')
    
    images, targets = [], []
    for img_name, target in csv_data.iterrows():
    	# torchvision.io.read_image 函数加载图像数据
    	# 并自动将其转换为 PyTorch 的 tensor 格式
        images.append(torchvision.io.read_image(
            os.path.join(data_dir, 'bananas_train' if is_train else 'bananas_val', 'images', f'{img_name}')))
        
       # 这里的target包含（类别，左上角x，左上角y，右下角x，右下角y），
       # 其中所有图像都具有相同的香蕉类（索引为0）
        targets.append(list(target))
    
    # 归一化处理
    # unsqueeze(1) 是为了确保标签的维度与模型期望的输入维度匹配
    return images, torch.tensor(targets).unsqueeze(1) / 256
```

- 通过使用`read_data_bananas`函数读取图像和标签，以下`BananasDataset`类别将允许我们创建一个自定义`Dataset`实例来加载香蕉检测数据集。

```
class BananasDataset(torch.utils.data.Dataset):
    """一个用于加载香蕉检测数据集的自定义数据集"""
    def __init__(self, is_train):
        self.features, self.labels = 	
        read_data_bananas(is_train)
        
        print('read ' + str(len(self.features)) + (f' training examples' if is_train else f' validation examples'))

    def __getitem__(self, idx):
        return (self.features[idx].float(), self.labels[idx])

    def __len__(self):
        return len(self.features)
```

- 最后，我们定义`load_data_bananas`函数，来为训练集和测试集返回两个数据加载器实例。对于测试集，无须按随机顺序读取它。

```
def load_data_bananas(batch_size):
    """加载香蕉检测数据集"""
    train_iter = torch.utils.data.DataLoader(
    BananasDataset(is_train=True),batch_size, shuffle=True)
    val_iter = torch.utils.data.DataLoader(
    BananasDataset(is_train=False),batch_size)
    
    return train_iter, val_iter
```

- 让我们读取一个小批量，并打印其中的图像和标签的形状。 图像的小批量的形状为（批量大小、通道数、高度、宽度），看起来很眼熟：它与我们之前图像分类任务中的相同。 标签的小批量的形状为（批量大小，*m*，5），其中*m*是数据集的任何图像中边界框可能出现的最大数量。

```
batch_size, edge_size = 32, 256
train_iter, _ = load_data_bananas(batch_size)
batch = next(iter(train_iter))
batch[0].shape, batch[1].shape
```

#### 小结

- 我们收集的香蕉检测数据集可用于演示目标检测模型。
- 用于目标检测的数据加载与图像分类的数据加载类似。但是，在目标检测中，标签还包含真实边界框的信息，它不出现在图像分类中。





## 锚框

### 定义

#### 锚框（Anchor Boxes）

以每个像素为中心，生成多个缩放比和宽高比（aspect ratio）不同的边界框。 这些边界框被称为*锚框*（anchor box）。

锚框是一种预定义的框，用于作为对象检测算法中的参考点。**这些框有固定的大小和比例，用来预测实际对象可能出现的位置（预设好的）。**锚框的主要目的是提供一个初始的、固定的参考系统，使得检测模型可以学习到在这些参考框的基础上调整来准确预测对象的位置。

- **作用**：在训练对象检测模型时，每个锚框都会尝试匹配图像中最接近的真实对象。如果一个锚框与某个真实对象的重叠程度（通常使用交并比，Intersection over Union, IoU）高于某个阈值，则这个锚框被认为是正样本，模型将试图调整这个锚框的大小和位置来更好地覆盖这个对象。
- **应用**：锚框在许多现代对象检测框架中都有应用，如 Faster R-CNN、SSD（Single Shot MultiBox Detector）等。这些框架使用锚框作为学习目标位置和尺寸的基础。

##### 边界框（Bounding Boxes）

边界框是对象检测任务中的一个术语，指的是围绕识别对象的矩形框。**它们描述了对象在图像中的位置和范围。**（真实存在）

- **参数**：边界框通常由四个坐标定义，(x_min, y_min, x_max, y_max) 或 (x, y, width, height)，其中 (x, y) 代表框的左上角坐标，而 width 和 height 分别代表框的宽度和高度。
- **功能**：在对象检测中，边界框是模型输出的一部分，用来标示模型认为有对象存在的区域。**模型的任务是正确预测对象的类别及其边界框的位置**。
- **评估**：在测试对象检测模型时，预测的边界框会与真实的边界框（标注数据中给出）比较，通过计算它们的 IoU 来评估模型的精确度。高 IoU 值表示预测的边界框与真实框非常接近，即预测准确。

#### 锚框与边界框的关系

虽然锚框和边界框都是用来定位图像中的对象，但它们的功能和用途有所不同。锚框是用来作为对象检测算法中的**参考和起点**，而边界框则是**算法的最终输出**，表示模型检测到的对象的具体位置。锚框通常在算法内部使用，而边界框则是算法对外输出的对象位置信息。

### IoU-交并比

比较相似度

![image-20240721115744680](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20240721115744680.png)

### 赋予锚框标号

- 将每个锚框视作一个训练样本。要么将锚框标注成背景（负类样本），要么关联上一个真实边缘框：会产生大量的负类样本。
- 每次把图片读进来我们都会对图片进行“赋予锚框标号”的操作。

### 使用（NMS）非极大值抑制输出



### 代码

- 分配锚框
- 计算IoU，删除背景锚框
- 确定每个锚框的*类别*（class）和*偏移量*（offset）

#### 锚框

```python
import torch
from d2l import torch as d2l

# 精简输出，控制输出精度
torch.set_printoptions(2)
```

#### 生成多个锚框

假设输入图像的高度为ℎ，宽度为𝑤。 我们以图像的每个像素为中心生成不同形状的锚框：*缩放比*为𝑠∈(0,1]，*宽高比*为𝑟>0。

**缩放比**：代表锚框尺寸与图像尺寸的比例。如果 s=1，则锚框的大小在某个维度上可能与图像的大小相同。

**宽高比**：锚框宽度和高度的比例。一个大于1的 r值表示锚框比图像更宽，而小于1的值则表示锚框比图像更高。通过改变宽高比，可以生成各种形状的锚框，以更好地适应图像中可能的不同对象形状。

![image-20240722143854123](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20240722143854123.png)

请注意，当中心位置给定时，已知宽和高的锚框是确定的。

要生成多个不同形状的锚框，让我们设置许多缩放比（scale）取值𝑠1,…,𝑠𝑛和许多宽高比（aspect ratio）取值𝑟1,…,𝑟𝑚。 当使用这些比例和长宽比的所有组合以每个像素为中心时，输入图像将总共有𝑤ℎ𝑛𝑚个锚框。 尽管这些锚框可能会覆盖所有真实边界框，但计算复杂性很容易过高。 在实践中，**我们只考虑**包含𝑠1或𝑟1的**组合：**

(**(𝑠1,𝑟1),(𝑠1,𝑟2),…,(𝑠1,𝑟𝑚),(𝑠2,𝑟1),(𝑠3,𝑟1),…,(𝑠𝑛,𝑟1).**)

也就是说，以同一像素为中心的锚框的数量是𝑛+𝑚−1。 对于整个输入图像，将共生成𝑤ℎ(𝑛+𝑚−1)个锚框。

上述生成锚框的方法在下面的`multibox_prior`函数中实现。 我们指定输入图像、尺寸列表和宽高比列表，然后此函数将返回所有的锚框。

```python
# 生成锚框
def multibox_prior(data, sizes, ratios):
    """生成以每个像素为中心具有不同形状的锚框"""
    
    # data.shape通常返回张量的的维度
    # 二维数据：(channels, height, width)
    in_height, in_width = data.shape[-2:]
    
    device, num_sizes, num_ratios = data.device, len(sizes), len(ratios)
    # 生成的锚框数量
    boxes_per_pixel = (num_sizes + num_ratios - 1)
    size_tensor = torch.tensor(sizes, device=device)
    ratio_tensor = torch.tensor(ratios, device=device)

    # 为了将锚点移动到像素的中心，需要设置偏移量。
    # 因为一个像素的高为1且宽为1，我们选择偏移我们的中心0.5
    offset_h, offset_w = 0.5, 0.5
    steps_h = 1.0 / in_height  # 在y轴上缩放步长
    steps_w = 1.0 / in_width  # 在x轴上缩放步长

    # 生成锚框的所有中心点
    # torch.arange(in_height）会生成一个从0到in_height-1的整数序列
    #  + offset_h 是为了将起始位置从边缘移到中心
    center_h = (torch.arange(in_height, device=device) + offset_h) * steps_h
    center_w = (torch.arange(in_width, device=device) + offset_w) * steps_w
    
    # torch.meshgrid 生成一个网格，这个网格上的每一点都是一个中心点的坐标
    shift_y, shift_x = torch.meshgrid(center_h, center_w, indexing='ij')
    # 坐标拉平为一维数组
    shift_y, shift_x = shift_y.reshape(-1), shift_x.reshape(-1)

    # 生成“boxes_per_pixel”个高和宽，
    # 之后用于创建锚框的四角坐标(xmin,xmax,ymin,ymax)
    w = torch.cat((size_tensor * torch.sqrt(ratio_tensor[0]),
                   sizes[0] * torch.sqrt(ratio_tensor[1:])))\
                   * in_height / in_width  # 处理矩形输入
    h = torch.cat((size_tensor / torch.sqrt(ratio_tensor[0]),
                   sizes[0] / torch.sqrt(ratio_tensor[1:])))
    
    # 除以2来获得半高和半宽
    anchor_manipulations = torch.stack((-w, -h, w, h)).T.repeat(in_height * in_width, 1) / 2

    # 每个中心点都将有“boxes_per_pixel”个锚框，
    # 所以生成含所有锚框中心的网格，重复了“boxes_per_pixel”次
    out_grid = torch.stack([shift_x, shift_y, shift_x, shift_y],dim=1).repeat_interleave(boxes_per_pixel, dim=0)
    output = out_grid + anchor_manipulations
    return output.unsqueeze(0)
```

##### **显示以图像中以某个像素为中心的所有锚框**

```python
def show_bboxes(axes, bboxes, labels=None, colors=None):
    """显示所有边界框"""
    def _make_list(obj, default_values=None):
        if obj is None:
            obj = default_values
        elif not isinstance(obj, (list, tuple)):
            obj = [obj]
        return obj

    labels = _make_list(labels)
    colors = _make_list(colors, ['b', 'g', 'r', 'm', 'c'])
    for i, bbox in enumerate(bboxes):
        color = colors[i % len(colors)]
        rect = d2l.bbox_to_rect(bbox.detach().numpy(), color)
        axes.add_patch(rect)
        if labels and len(labels) > i:
            text_color = 'k' if color == 'w' else 'w'
            axes.text(rect.xy[0], rect.xy[1], labels[i],
                      va='center', ha='center', fontsize=9, color=text_color,
                      bbox=dict(facecolor=color, lw=0))
```

#### 交并比IoU

```
def box_iou(boxes1, boxes2):
# 其中每个边界框用四个坐标表示（通常是 xmin, ymin, xmax, ymax）。
    """计算两个锚框或边界框列表中成对的交并比"""
    box_area = lambda boxes: ((boxes[:, 2] - boxes[:, 0]) *
                              (boxes[:, 3] - boxes[:, 1]))
    # boxes1,boxes2,areas1,areas2的形状:
    # boxes1：(boxes1的数量,4),
    # boxes2：(boxes2的数量,4),
    # areas1：(boxes1的数量,),
    # areas2：(boxes2的数量,)
    areas1 = box_area(boxes1)
    areas2 = box_area(boxes2)
    
    # inter_upperlefts,inter_lowerrights,inters的形状:
    # (boxes1的数量,boxes2的数量,2)
    inter_upperlefts = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    inter_lowerrights = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    
    # 如果交集不存在（即宽或高为负值），通过clamp(min=0) 确保其值为零
    inters = (inter_lowerrights - inter_upperlefts).clamp(min=0)
    
    # inter_areasandunion_areas的形状:(boxes1的数量,boxes2的数量)
    inter_areas = inters[:, :, 0] * inters[:, :, 1]
    union_areas = areas1[:, None] + areas2 - inter_areas
    return inter_areas / union_areas
```

#### 将真实边界框分配给锚框

我们需要每个锚框的*类别*（class）和*偏移量*（offset）标签，其中前者是与锚框相关的对象的类别，后者是真实边界框相对于锚框的偏移量。

要标记任何生成的锚框，我们可以参考分配到的最接近此锚框的真实边界框的位置和类别标签。 下文将介绍一个算法，它能够把最接近的真实边界框分配给锚框。

```
# 真实边界框集合 ground_truth，锚框集合 anchors，运算设备 device，以及一个 IoU 阈值 iou_threshold
def assign_anchor_to_bbox(ground_truth, anchors, device, iou_threshold=0.5):
    """将最接近的真实边界框分配给锚框"""
    # num_anchors 和 num_gt_boxes 分别是锚框和真实边界框的数量。
    num_anchors, num_gt_boxes = anchors.shape[0], ground_truth.shape[0]
    
    # 位于第i行和第j列的元素x_ij是锚框i和真实边界框j的IoU
    jaccard = box_iou(anchors, ground_truth)
    
    # 对于每个锚框，分配的真实边界框的张量
    # 初始化：torch.full 是一个用来创建具有指定形状和填充单一值的张量的函数。
    # torch.full(size, fill_value, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
    # size (tuple 或 list)：张量形状，(3, 4) 表示一个3行4列的矩阵。
    # fill_value (scalar)：用于填充张量的值。
    anchors_bbox_map = torch.full((num_anchors,), -1, dtype=torch.long,device=device)
    
    
    # 根据阈值，决定是否分配真实边界框
    max_ious, indices = torch.max(jaccard, dim=1)
    
    # anc_i 是所有 IoU 超过阈值的锚框的索引列表。
	# box_j 是这些锚框对应的最匹配真实边界框的索引。
    # torch.nonzero 是一个 PyTorch 函数，用于返回一个张量中所有非零元素的索引。
    anc_i = torch.nonzero(max_ious >= iou_threshold).reshape(-1)
    box_j = indices[max_ious >= iou_threshold]
    anchors_bbox_map[anc_i] = box_j
    col_discard = torch.full((num_anchors,), -1)
    row_discard = torch.full((num_gt_boxes,), -1)
   
   for _ in range(num_gt_boxes):
        max_idx = torch.argmax(jaccard)
        box_idx = (max_idx % num_gt_boxes).long()
        anc_idx = (max_idx / num_gt_boxes).long()
        anchors_bbox_map[anc_idx] = box_idx
        jaccard[:, box_idx] = col_discard
        jaccard[anc_idx, :] = row_discard
    
    return anchors_bbox_map
```

#### 标记类别和偏移量

类别：与真实边界框相同；如果一个锚框没有被分配真实边界框，我们只需将锚框的类别标记为*背景*（background）。 背景类别的锚框通常被称为*负类*锚框，其余的被称为*正类*锚框。

偏移量：锚框𝐴的偏移量将根据𝐵和𝐴中心坐标的相对位置以及这两个框的相对大小进行标记。给定框𝐴和𝐵，中心坐标分别为(𝑥𝑎,𝑦𝑎)和(𝑥𝑏,𝑦𝑏)，宽度分别为𝑤𝑎和𝑤𝑏，高度分别为ℎ𝑎和ℎ𝑏，可以将𝐴的偏移量标记为：
	![image-20240722155710056](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20240722155710056.png)

```
def offset_boxes(anchors, assigned_bb, eps=1e-6):
    """对锚框偏移量的转换"""
    c_anc = d2l.box_corner_to_center(anchors)
    c_assigned_bb = d2l.box_corner_to_center(assigned_bb)
    offset_xy = 10 * (c_assigned_bb[:, :2] - c_anc[:, :2]) / c_anc[:, 2:]
    offset_wh = 5 * torch.log(eps + c_assigned_bb[:, 2:] / c_anc[:, 2:])
    offset = torch.cat([offset_xy, offset_wh], axis=1)
    return offset
    
def multibox_target(anchors, labels):
    """使用真实边界框标记锚框"""
    batch_size, anchors = labels.shape[0], anchors.squeeze(0)
    batch_offset, batch_mask, batch_class_labels = [], [], []
    device, num_anchors = anchors.device, anchors.shape[0]
    for i in range(batch_size):
        label = labels[i, :, :]
        anchors_bbox_map = assign_anchor_to_bbox(
            label[:, 1:], anchors, device)
        bbox_mask = ((anchors_bbox_map >= 0).float().unsqueeze(-1)).repeat(
            1, 4)
        # 将类标签和分配的边界框坐标初始化为零
        class_labels = torch.zeros(num_anchors, dtype=torch.long,
                                   device=device)
        assigned_bb = torch.zeros((num_anchors, 4), dtype=torch.float32,
                                  device=device)
        # 使用真实边界框来标记锚框的类别。
        # 如果一个锚框没有被分配，标记其为背景（值为零）
        indices_true = torch.nonzero(anchors_bbox_map >= 0)
        bb_idx = anchors_bbox_map[indices_true]
        class_labels[indices_true] = label[bb_idx, 0].long() + 1
        assigned_bb[indices_true] = label[bb_idx, 1:]
        # 偏移量转换
        offset = offset_boxes(anchors, assigned_bb) * bbox_mask
        batch_offset.append(offset.reshape(-1))
        batch_mask.append(bbox_mask.reshape(-1))
        batch_class_labels.append(class_labels)
    bbox_offset = torch.stack(batch_offset)
    bbox_mask = torch.stack(batch_mask)
    class_labels = torch.stack(batch_class_labels)
    return (bbox_offset, bbox_mask, class_labels)
```

#### 使用非极大值抑制预测边界框

```
def offset_inverse(anchors, offset_preds):
    """根据带有预测偏移量的锚框来预测边界框"""
    anc = d2l.box_corner_to_center(anchors)
    pred_bbox_xy = (offset_preds[:, :2] * anc[:, 2:] / 10) + anc[:, :2]
    pred_bbox_wh = torch.exp(offset_preds[:, 2:] / 5) * anc[:, 2:]
    pred_bbox = torch.cat((pred_bbox_xy, pred_bbox_wh), axis=1)
    predicted_bbox = d2l.box_center_to_corner(pred_bbox)
    return predicted_bbox
```

##### **以下`nms`函数按降序对置信度进行排序并返回其索引**

```
def nms(boxes, scores, iou_threshold):
    """对预测边界框的置信度进行排序"""
    B = torch.argsort(scores, dim=-1, descending=True)
    keep = []  # 保留预测边界框的指标
    while B.numel() > 0:
        i = B[0]
        keep.append(i)
        if B.numel() == 1: break
        iou = box_iou(boxes[i, :].reshape(-1, 4),
                      boxes[B[1:], :].reshape(-1, 4)).reshape(-1)
        inds = torch.nonzero(iou <= iou_threshold).reshape(-1)
        B = B[inds + 1]
    return torch.tensor(keep, device=boxes.device)
```

**将非极大值抑制应用于预测边界框**

```
def multibox_detection(cls_probs, offset_preds, anchors, nms_threshold=0.5,pos_threshold=0.009999999):
    """使用非极大值抑制来预测边界框"""
    device, batch_size = cls_probs.device, cls_probs.shape[0]
    anchors = anchors.squeeze(0)
    num_classes, num_anchors = cls_probs.shape[1], cls_probs.shape[2]
    out = []
    for i in range(batch_size):
        cls_prob, offset_pred = cls_probs[i], offset_preds[i].reshape(-1, 4)
        conf, class_id = torch.max(cls_prob[1:], 0)
        predicted_bb = offset_inverse(anchors, offset_pred)
        keep = nms(predicted_bb, conf, nms_threshold)

        # 找到所有的non_keep索引，并将类设置为背景
        all_idx = torch.arange(num_anchors, dtype=torch.long, device=device)
        combined = torch.cat((keep, all_idx))
        uniques, counts = combined.unique(return_counts=True)
        non_keep = uniques[counts == 1]
        all_id_sorted = torch.cat((keep, non_keep))
        class_id[non_keep] = -1
        class_id = class_id[all_id_sorted]
        conf, predicted_bb = conf[all_id_sorted], predicted_bb[all_id_sorted]
        # pos_threshold是一个用于非背景预测的阈值
        below_min_idx = (conf < pos_threshold)
        class_id[below_min_idx] = -1
        conf[below_min_idx] = 1 - conf[below_min_idx]
        pred_info = torch.cat((class_id.unsqueeze(1),
                               conf.unsqueeze(1),
                               predicted_bb), dim=1)
        out.append(pred_info)
    return torch.stack(out)
```





## 物体检测算法：RCNN,SSD,YOLO

### 区域卷积神经网络

#### R-CNN

- 使用启发式搜索算法来选择锚框
- 使用预训练模型来对每个锚框抽取特征（每个锚框当作一个图片，用CNN）
- 训练一个SVM来类别分类（神经网络之前，category prediction）
- 训练一个线性回归模型来预测边缘框偏移（bounding box prediction）
- **兴趣区域（Rol）池化层**（有利于尺寸标准化；并批量处理）
  - 给定一个锚框，**均匀分割**（如果没法均匀分割，取整）成 n x m 块，**输出每块的最大值**（max pooling）
  - 不管锚框多大，总是输出nm个值
  - 目的：每个锚框都可以变成想要的形状

#### Fast RCNN

- 使用CNN对**整张图片做卷积层，然后利用Rol池化层抽取特征**（快的关键）

- 使用Rol池化层对每个锚框（将在原图片中搜索到的锚框，映射到CNN得到的结果上），生成固定长度的特征

  [![image](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/44/44-01.png)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/44/44-01.png)

#### Faster RCNN

- 使用一个 **区域提案网络（RPN）来替代启发式搜索获得更好的锚框**

  RPN 是 Faster R-CNN 的核心，用于实时生成高质量的区域提案。它是一个全卷积网络，可以同时预测边界框位置和对象存在的分数。

  RPN 使用了一组预定义的锚框（anchors），这些锚框具有不同的尺寸和宽高比。RPN 评估每个锚框是否包含对象，并调整其位置（通过边界框回归）以更好地匹配潜在对象。

- 如下图所示，将CNN结果输入到卷积层，然后用锚框去圈区域，这些锚框有好有坏，然后进行预测，binary 预测是预测这个锚框的好坏，即有没有有效的圈住物体，bounding box prediction预测是对锚框进行一些改进，最后用NMS（非极大值抑制）对锚框进行合并。

- 具体来说，区域提议网络的计算步骤如下：

  1. 使用填充为1的3×3的卷积层（保持输入输出的宽高不变）变换卷积神经网络的输出，并将输出通道数记为c。这样，卷积神经网络为图像抽取的特征图中的每个单元均得到一个长度为c的新特征。
  2. 以特征图的每个像素为中心，生成多个不同大小和宽高比的锚框并标注它们。
  3. 使用锚框中心单元长度为c的特征，分别预测该锚框的二元类别（含目标还是背景）和边界框。
  4. 使用非极大值抑制，从预测类别为目标的预测边界框中移除相似的结果。最终输出的预测边界框即是兴趣区域汇聚层所需的提议区域。

[![image](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/44/44-02.png)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/44/44-02.png)

#### Mask RCNN

像素级的对象分割

- 如果有**像素级别的标号**，使用FCN（fully convolutional network）利用这些信息。可以提升CNN的性能

- **Rol align**。解决了 Faster R-CNN 中 RoI Pooling 层的对齐问题。RoI Pooling 在抽样过程中对输入进行了量化（即四舍五入），这可能会导致边界框和实际特征之间的轻微错位。RoIAlign 通过使用**双线性插值**来精确地计算输入特征图上任意位置的值，从而保证了特征的精确对齐。

- 双线性插值：

  双线性插值的基本思想是**在两个方向上进行线性插值**。假设我们需要在一个给定的图像上找到一个非整数坐标（x, y）处的像素值。**周围的四个最近的整数像素点（通常是这个点的四个角落）将被用于计算这个点的像素值。**

  设这四个点为：

  - Q11=(x1,y1)
  - Q12=(x1,y2)
  - Q21=(x2,y1)
  - Q22=(x2,y2)

  并且对应的像素值为 f(Q11),f(Q12),f(Q21),f(Q22)

  为了在点 (x,y)处进行插值，双线性插值通过以下步骤进行：

  1. **水平方向的线性插值**：

     - 对 y1行进行插值得到

       R1=f(Q11)×(x2−x)+f(Q21)×(x−x1)

     - 对 y2行进行插值得到 ：

       R2=f(Q12)×(x2−x)+f(Q22)×(x−x1)

  2. **垂直方向的线性插值**：

     - 使用 R1和 R2对 y 进行插值

       P=R1×(y2−y)+R2×(y−y1)

  这里的 P就是点 (x,y)(x, y)(x,y) 处的插值结果。

[![image](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/44/44-03.png)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/44/44-03.png)

#### 总结

- R-CNN是最早也是最有名的一类基于锚框和CNN的目标检测算法
- Fast/Faster RCNN 持续提升性能
- Faster RCNN和Mask RCNN是在要求高精度场景下常用的算法（但是速度是最慢的）

### 单发多框检测（SSD single shot detection）

- 生成锚框
  - 对每个像素，生成多个以它为中心的锚框
  - 给定 n 个大小$s_1,...,s_n$和m个高宽比，生成n+m-1个锚框，其大小和高宽比分别为：$(s_1,r_1),(s_2,r_1)...,(s_n,r_1),(s_1,r_2),...,(s_1,r_m)$
- SSD模型
  - 对多个分辨率下(从基础网络中的多个层上提取特征图，用以检测不同尺寸的对象。每一层的特征图都会被用来预测对象的类别和位置)的卷积特征，生成锚框，预测
  - 分辨率：在基础神经网络中，每一层的输出具有不同的空间分辨率，越接近输入层的分辨率越高（更细粒度的信息），越接近输出层的分辨率越低；
  - 特征图的使用：低层特征图，具有较高分辨率，适合用来检测小物体；反之。
  - 一个基础网络，抽取特征（不是所有层都会被用到），然后用多个卷积层来减半高宽
  - 在每段都生成锚框
    - 底部段拟合小物体
    - 顶部段拟合大物体
  - 对每个锚框预测类别和边缘框

[![image](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/44/44-04.png)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/44/44-04.png)

#### 总结

- 速度快，精度很低。这么多年，作者没有持续的提升，但是启发了后面的一系列工作，实现上相对比较简单。
- SSD通过单神经网络来检测模型（single shot）
- 以像素为中心的产生多个锚框
- 在多个段的输出上进行多尺度的检测

### YOLO（you only look once）

- YOLO将图片均分为 S X S 个锚框
- 每个锚框预测 B 个边缘框（防止多个物体出现在一个锚框里面）
- 后续版本 v2 v3 v4 有持续改进
- 非锚框算法

### SSD代码实现

##### 模型

回想一下在 :numref:`sec_multiscale-object-detection`中，通过深度神经网络分层表示图像的多尺度目标检测的设计。 由于接近 :numref:`fig_ssd`顶部的多尺度特征图较小，但具有较大的感受野，它们适合检测较少但较大的物体。 简而言之，**通过多尺度特征块，单发多框检测生成不同大小的锚框**，并通过**预测边界框的类别和偏移量来检测大小不同的目标**，因此这是一个多尺度目标检测模型。

![image-20240724162125469](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20240724162125469.png)

在下面，我们将介绍 :numref:`fig_ssd`中不同块的实施细节。 首先，我们将讨论如何实施类别和边界框预测。

##### **类别预测层**

具体来说，类别预测层使用一个保持输入高和宽的卷积层。 这样一来，输出和输入在特征图宽和高上的空间坐标一一对应。 考虑输出和输入同一空间坐标（𝑥、𝑦）：输出特征图上（𝑥、𝑦）坐标的通道里包含了以输入特征图（𝑥、𝑦）坐标为中心生成的所有锚框的类别预测。 因此输出通道数为𝑎(𝑞+1)，其中索引为**𝑖(𝑞+1)+𝑗（0≤𝑗≤𝑞）的通道代表了索引为𝑖的锚框有关类别索引为𝑗的预测**。

- 保留空间信息

- 将通道对应为锚框信息

下面，我们定义了这样一个类别预测层，通过参数`num_anchors`和`num_classes`分别指定了𝑎和𝑞。 该图层使用填充为1的3×3的卷积层（保持高宽不变）。此卷积层的输入和输出的宽度和高度保持不变。

```
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

# num_inputs是输入的通道数
def cls_predictors(
	num_inputs,num_anchors,num_classes):
	return nn.Conv2d(
	num_inputs,(num_classes+1)*num_anchors,
	kernel_size=3,padding=1) 
```

##### **边界框预测层**

边界框预测层的设计与类别预测层的设计类似。 唯一不同的是，这里需要为每个锚框预测4个偏移量，而不是𝑞+1个类别。

###### 为什么是 `num_anchors * 4`？

在目标检测中，当预测边界框（bounding boxes）时，每个锚框需要预测4个值：边界框的中心点 (x, y)、宽度和高度。因此，如果你有`num_anchors`个锚框，每个都需要预测4个值，那么总共就是 `num_anchors * 4` 个输出。

```
def bbox_predictors(num_inputs,num_anchors):
	return nn.Conv2d(num_inputs, num_anchors * 4, kernel_size=3, padding=1)
```

##### **连结多尺度的预测**

正如我们所提到的，单发多框检测使用多尺度特征图来生成锚框并预测其类别和偏移量。 在不同的尺度下，特征图的形状或以同一单元为中心的锚框的数量可能会有所不同。 因此，不同尺度下预测输出的形状可能会有所不同。

**本质就是由于锚框不一致，导致尺寸不一致**

**解决方法**：将通道维移到最后一维。由于不同尺度下批量大小不变，我们将预测结果转变成二维的（批量大小，高×宽×通道数）的格式。

```
# 原理解析:在调用 permute(0,2,3,1) 的例子中，我们来逐个解释每个参数的含义：
# 0: 这表示原始张量的第一个维度（batch_size）将保持在第一个位置。
# 2: 这表示原始张量的第三个维度（height）将移动到新顺序的第二个位置。

# start_dim=1 指的是从张量的第二个维度（即索引为1的维度，对于排列后的 (batch_size, height, width, channels) 是 height）开始展平直到最后一个维度。

def flatten_pred(pred):
	return torch.flatten(pred.permute(0,2,3,1),start_dim=1)
	
def concat_pred(preds):
	return torch.cat([flatten_pred(p) for p in preds],dim=1)
```

##### **高和宽减半块**

为了在多个尺度下检测目标，我们在下面定义了高和宽减半块`down_sample_blk`，该模块将输入特征图的高度和宽度减半。 事实上，该块应用了在 :numref:`subsec_vgg-blocks`中的VGG模块设计。 更具体地说，每个高和宽减半块由两个填充为1的3×3的卷积层、以及步幅为2的2×2最大汇聚层组成。 我们知道，填充为1的3×3卷积层不改变特征图的形状。但是，其后的2×2的最大汇聚层将输入特征图的高度和宽度减少了一半。 对于此高和宽减半块的输入和输出特征图，因为1×2+(3−1)+(3−1)=6，所以输出中的每个单元在输入上都有一个6×6的感受野。因此，高和宽减半块会扩大每个单元在其输出特征图中的感受野。

**就是利用VGG里面的max_pooling去使得高宽减半**

```
def down_sample_blk(in_channels,out_channels):
	blk=[]
	for _ in range(2):
		blk.append(nn.Conv2d(in_channels,
	out_channels,kernel_size=2,padding=1))
		blk.append(nn.BatchNorm2d(
					out_channels))
        blk.append(nn.ReLU())
        in_channels = out_channels
    blk.append(nn.MaxPool2d(2))
    return nn.sequential(*blk)
```

##### 基本网络块

基本网络块用于从输入图像中抽取特征。 为了计算简洁，我们构造了一个小的基础网络，该网络串联3个高和宽减半块，并逐步将通道数翻倍。

**融入上面讲的高宽减半，这次加进去通道翻倍**

```
def base_net():
	blk=[]
	# 每一层的卷积核的数量,也就是每一层的输入通道和输入通道
	num_filters = [3, 16, 32, 64]
	for i in range(len(num_filters)-1):
		blk.append(down_sample_blk(num_filters[i],num_filters[i+1]))
		return nn.sequential(*blk)
```

##### 完整的模型

[**完整的单发多框检测模型由五个模块组成**]。每个块生成的特征图**既用于生成锚框**，又用于**预测这些锚框的类别和偏移量**。

在这五个模块中，

第一个是基本网络块，

第二个到第四个是高和宽减半块，

最后一个模块使用全局最大池将高度和宽度都降到1。

从技术上讲，第二到第五个区块都是 :numref:`fig_ssd`中的多尺度特征块。

```
# 这里可供选择不同的网络快
def get_blk(i):
    if i == 0:
        blk = base_net()
    elif i == 1:
        blk = down_sample_blk(64, 128)
    elif i == 4:
        blk = nn.AdaptiveMaxPool2d((1,1))
    else:
    	# 这个地方的通道不再进行改变，而是进行空间上的压缩
        blk = down_sample_blk(128, 128)
    return blk
```

**为每个块定义前向传播**

输出包括：CNN特征图`Y`；

在当前尺度下根据`Y`生成的锚框；

预测的这些锚框的类别和偏移量（基于`Y`）

```
def blk_forward(X,blk,size,ratio,cls_predictor,bbox_predictor):
	Y = blk(X)multibox_prior(Y, sizes=size, ratios=ratio)
	anchor = d2l.multibox_prior(Y, sizes=size, ratios=ratio)
	cls_preds = cls_predictor(Y)
    bbox_preds = bbox_predictor(Y)
    return (Y, anchors, cls_preds, bbox_preds)
```

现在，我们就可以按如下方式[**定义完整的模型**]`TinySSD`了。

##### TinySSD

```python
class TinySSD(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super(TinySSD, self).__init__(**kwargs)
        self.num_classes = num_classes
        idx_to_in_channels = [64, 128, 128, 128, 128]
        for i in range(5):
            # 即赋值语句self.blk_i=get_blk(i)
            
            '''
在Python中，setattr() 函数用于设置对象的属性。其通常格式为 setattr(object, name, value)，其中 object 是需要设置属性的对象，name 是要设置的属性名称（作为字符串），而 value 是属性值。
			'''
			
            setattr(self, f'blk_{i}', get_blk(i))
            setattr(self, f'cls_{i}', cls_predictor(idx_to_in_channels[i],                                     num_anchors, num_classes))
            setattr(self, f'bbox_{i}', bbox_predictor(idx_to_in_channels[i],num_anchors))

    def forward(self, X):
        anchors, cls_preds, bbox_preds = [None] * 5, [None] * 5, [None] * 5
        for i in range(5):
            # getattr(self,'blk_%d'%i)即访问self.blk_i
            X, anchors[i], cls_preds[i], bbox_preds[i] = 	blk_forward(X, getattr(self, f'blk_{i}'), sizes[i], ratios[i],getattr(self, f'cls_{i}'), getattr(self, f'bbox_{i}'))
        anchors = torch.cat(anchors, dim=1)
        cls_preds = concat_preds(cls_preds)
        
        '''Class predictions are reshaped to ensure the final output has a dimension of [batch_size, num_predictions, num_classes + 1]'''
        cls_preds = cls_preds.reshape(
            cls_preds.shape[0], -1, self.num_classes + 1)
        bbox_preds = concat_preds(bbox_preds)
        return anchors, cls_preds, bbox_preds
```

##### 训练模型

- ###### 读取数据集和初始化

- ###### **定义损失函数和评价函数**

- ###### **训练模型**

- ###### **预测目标**

- ###### **筛选所有置信度不低于0.9的边界框，做为最终输出**

```
###### 读取数据集和初始化
# 初始化内容包括批量大小/设备/网络类型/优化算法
batch_size = 32
train_iter, _ = d2l.load_data_bananas(batch_size)
device, net = d2l.try_gpu(), TinySSD(num_classes=1)
trainer = torch.optim.SGD(net.parameters(), lr=0.2, weight_decay=5e-4)

###### **定义损失函数和评价函数**
# 锚框类别损失；锚框偏移量的损失
cls_loss = nn.CrossEntrophyLoss(reduction = 'none')
bbox_loss = nn.L1Loss(reduction = 'none')

def calc_loss(cls_preds,cls_labels,bbox_preds, bbox_labels, bbox_masks):
	batch_size,num_classes = cls_preds.shape[0], cls_preds.shape[2]
	bbox = bbox_loss(bbox_preds * bbox_masks,
                     bbox_labels * bbox_masks).mean(dim=1)
    return cls + bbox

def cls_eval(cls_preds, cls_labels):
    # 由于类别预测结果放在最后一维，argmax需要指定最后一维。
    return float((cls_preds.argmax(dim=-1).type(
        cls_labels.dtype) == cls_labels).sum())

def bbox_eval(bbox_preds, bbox_labels, bbox_masks):
    return float((torch.abs((bbox_labels - bbox_preds) * bbox_masks)).sum())
    
    
###### **训练模型**
num_epochs, timer = 20, d2l.Timer()
animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                        legend=['class error', 'bbox mae'])
net = net.to(device)
for epoch in range(num_epochs):
    # 训练精确度的和，训练精确度的和中的示例数
    # 绝对误差的和，绝对误差的和中的示例数
    metric = d2l.Accumulator(4)
    net.train()
    for features, target in train_iter:
        timer.start()
        trainer.zero_grad()
        X, Y = features.to(device), target.to(device)
        
        # 生成多尺度的锚框，为每个锚框预测类别和偏移量
        anchors, cls_preds, bbox_preds = net(X)
        
        # 为每个锚框标注类别和偏移量
        bbox_labels, bbox_masks, cls_labels = d2l.multibox_target(anchors, Y)
        
        # 根据类别和偏移量的预测和标注值计算损失函数
        l = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks)
        l.mean().backward()
        trainer.step()
        metric.add(cls_eval(cls_preds, cls_labels), cls_labels.numel(),bbox_eval(bbox_preds, bbox_labels, bbox_masks),bbox_labels.numel())
    cls_err, bbox_mae = 1 - metric[0] / metric[1], metric[2] / metric[3]
    animator.add(epoch + 1, (cls_err, bbox_mae))
print(f'class err {cls_err:.2e}, bbox mae {bbox_mae:.2e}')
print(f'{len(train_iter.dataset) / timer.stop():.1f} examples/sec on '
      f'{str(device)}')


###### **预测目标**
X = torchvision.io.read_image('../img/banana.jpg').		unsqueeze(0).float()
img = X.squeeze(0).permute(1, 2, 0).long()

def predict(X):
    net.eval()
    anchors, cls_preds, bbox_preds = net(X.to(device))
    cls_probs = F.softmax(cls_preds, dim=2).permute(0, 2, 1)
    output = d2l.multibox_detection(cls_probs, bbox_preds, anchors)
    idx = [i for i, row in enumerate(output[0]) if row[0] != -1]
    return output[0, idx]
    
output = predict(X)

###### **筛选所有置信度不低于0.9的边界框，做为最终输出**
def display(img, output, threshold):
    d2l.set_figsize((5, 5))
    fig = d2l.plt.imshow(img)
    for row in output:
        score = float(row[1])
        if score < threshold:
            continue
        h, w = img.shape[0:2]
        bbox = [row[2:6] * torch.tensor((w, h, w, h), device=row.device)]
        d2l.show_bboxes(fig.axes, bbox, '%.2f' % score, 'w')

display(img, output.cpu(), threshold=0.9)
```



## 语义分割

### 语义分割

有时只能实现框选的目标检测还是太粗糙了，无法得到更精细的信息。语义分割将图片中的每个像素分类到对应的类别。

分割这一概念在计算机视觉中由来已久。最早的图片分割对给定图片使用聚类等方法把语义上比较像的像素放在一起，但通常不会告诉我们这些像素到底是什么。而语义分割可以告诉我们每个像素对应的label是什么。

这也意味着我们需要对图片的每一个像素都做label，使得语义分割成为了一个比较精细且大的任务。语义分割的数据集成本也较高，往往规模小像素高。常用的数据集之一是Pascal VOC2012。

[![46-01](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/46/46-01.jpg)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/46/46-01.jpg)

### 应用



背景虚化：传统的背景替换往往采用绿幕。在没有绿幕的情况下传统相机可以通过光圈来实现背景虚化，对于手机等设备而言背景虚化通常使用的都是语义分割或结合图像景深信息。

路面分割：如无人驾驶时用于实时识别周围物体，实现找路的功能。

### 实例分割

语义分割只关心像素属于哪一类，而实例分割则更进一步，如图片里有两只狗，则需要得出哪个像素属于哪一只狗。可以将其理解为目标检测的进化版本。

**语义分割对于每一个像素生成一个标号**



[![46-02](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/46/46-02.png)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/46/46-02.png)

### 语义分割和数据集

最重要的语义分割数据集之一是**Pascal VOC2012**

一般比较大的数据集不会全部读到内存中，但是以下为了操作方便

```
#@save
import os
import torch
import torchvision
from d2l import torch as d2l

d2l.DATA_HUB['voc2012'] = (d2l.DATA_URL + 'VOCtrainval_11-May-2012.tar','4e443f8a2eca6b1dac8a6c57641b67dd40621a49')
voc_dir = d2l.download_extract('voc2012', 'VOCdevkit/VOC2012')

def read_voc_images(voc_dir, is_train=True):
    """读取所有VOC图像并标注"""
    txt_fname = os.path.join(voc_dir, 'ImageSets', 'Segmentation',
                             'train.txt' if is_train else 'val.txt')
    mode = torchvision.io.image.ImageReadMode.RGB
    with open(txt_fname, 'r') as f:
        images = f.read().split()
    features, labels = [], []
    for i, fname in enumerate(images):
        features.append(torchvision.io.read_image(os.path.join(
            voc_dir, 'JPEGImages', f'{fname}.jpg')))
        labels.append(torchvision.io.read_image(os.path.join(
            voc_dir, 'SegmentationClass' ,f'{fname}.png'), mode))
    return features, labels

train_features, train_labels = read_voc_images(voc_dir, True)
```

**列举RGB颜色和类名**，**颜色和类名一一对应并完成映射**

```
#@save
VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]

#@save
VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']
               
               
def voc_colormap2label():
	voc_colormap2label = torch.zeros(256**3,dtype=torch.long)
	for i,colormap in enumerate(VOC_CLASSES):
		colormap2label[  (colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i
	return colormap2label
	
def voc_label_indices(colormap, colormap2label):
# 将输入的 colormap 从 [C, H, W] 格式转换为 [H, W, C] 格式
	colormap = colormap.permute(1, 2, 0).numpy().astype('int32')
 	idx = ((colormap[:,:,0]*256+colormap[:,:,1])*256+colormap[:,:,2])
 	return colormap2label[idx]
	
```

在语义分割中，语义像素与类别完全一一映射，如果缩放图像，需要将预测的像素类别重新映射回原始尺寸的输入图像。 为了避免这个问题，我们将图像裁剪为固定尺寸，而不是再缩放。 具体来说，我们[**使用图像增广中的随机裁剪，裁剪输入图像和标签的相同区域**]。

```
def voc_rand_crop(feature,label,height,width):
	# 获取裁剪区域的参数
	react = torchvision.transforms.RandomCrop.get_param(features,(height,width))
	 # 特征图像与标签图像
	 # 使用得到的裁剪参数裁剪特征图像
	 # 使用相同的裁剪参数裁剪标签图像
	 feature = torchvision.transforms.functional.crop(feature,*react)
	 label = torchvision.transforms.functional.crop(label,*react)
	 
	 return featrue,label
```

##### **自定义语义分割数据集类**

```
def __init__(self, is_train, crop_size, voc_dir):
        self.transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.crop_size = crop_size
        features, labels = read_voc_images(voc_dir, is_train=is_train)
        self.features = [self.normalize_image(feature)
                         for feature in self.filter(features)]
        self.labels = self.filter(labels)
        self.colormap2label = voc_colormap2label()
        print('read ' + str(len(self.features)) + ' examples')

    def normalize_image(self, img):
        return self.transform(img.float() / 255)

    def filter(self, imgs):
        return [img for img in imgs if (
            img.shape[1] >= self.crop_size[0] and
            img.shape[2] >= self.crop_size[1])]

    def __getitem__(self, idx):
        feature, label = voc_rand_crop(self.features[idx], self.labels[idx],
                                       *self.crop_size)
        return (feature, voc_label_indices(label, self.colormap2label))

    def __len__(self):
        return len(self.features)
```

##### 整合所有组件

```
def load_data_voc(batch_size,crop_size):
	# 加载并解压
	voc_dir = d2l.download_extract('voc2012', os.path.join(
        'VOCdevkit', 'VOC2012'))
    #工作的GPU
    num_workers = d2l.get_dataloader_workers()
    
    train_iter = torch.utils.data.DataLoader(
        VOCSegDataset(True, crop_size, voc_dir), batch_size,shuffle=True, drop_last=True, num_workers=num_workers)
    
    test_iter = torch.utils.data.DataLoader(
        VOCSegDataset(False, crop_size, voc_dir), batch_size,drop_last=True, num_workers=num_workers)
        
    return train_iter, test_iter
```







## 转置卷积

如果输入和输出图像的空间维度相同，在以像素级分类的语义分割中将会很方便。 例如，输出像素所处的通道维可以保有输入像素在同一位置上的分类结果。

我们可以通过转置卷积增加上采样中间层特征图的空间维度。

### 转置卷积

- 转置卷积和卷积的区别：
  - 卷积不会增大输入的高宽，通常要么不变、要么减半
  - 转置卷积则可以用来增大输入高宽
- 转置卷积的具体实现：

[![image](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/47/47-01.png)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/47/47-01.png)

如图所示，input里的每个元素和kernel相乘，最后把对应位置相加，相当于卷积的逆变换

- 为什么称之为“转置：
  - 对于卷积Y=X*W
    - 可以对W构造一个V，使得卷积等价于矩阵乘法Y'=VX'
    - 这里Y',X'是Y,X对应的向量版本
  - 转置卷积等价于Y'=VTX'
  - 如果卷积将输入从（h，w）变成了（h‘，w’）
    - 同样超参数的转置卷积则从（h‘，w’）变成为（h，w）

### 转置卷积是一种卷积

- 重新排列输入和核

  - 当填充为0步幅为1时：
    - 将输入填充k-1（k时核窗口）
    - 将核矩阵上下、左右翻转
    - 然后做正常卷积（填充0、步幅1）

  ![image](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/47/47-02.png)

  - 当填充为p步幅为1时：
    - 将输如填充k-p-1（k是核窗口）
    - 将核矩阵上下、左右翻转
    - 然后做正常卷积（填充0、步幅1）

  ![image](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/47/47-03.png)

  - 当填充为p步幅为s时：
    - 在行和列之间插入s-1行或列
    - 将输如填充k-p-1（k是核窗口）
    - 将核矩阵上下、左右翻转
    - 然后做正常卷积（填充0、步幅1）

  ![image](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/47/47-04.png)

- 形状换算

  - 输入高（宽）为n，核k，填充p，步幅s：
    - 转置卷积：n‘=sn+k-2p-s
    - 卷积：n’=[(n-k-2p+s)/s]向下取整
  - 如果让高宽成倍增加，那么k=2p+s

- 同反卷积的关系

  - 数学上的反卷积是指卷积的逆运算
    - 若Y=conv（X,K），那么X=deconv（Y,K）
  - 反卷积很少用在深度学习中
    - 我们说的反卷积神经网络指用了转置卷积的神经网络

- 总结

  - 转置卷积是一种变化了输入和核的卷积，来得到上采用的目的
  - 不等同于数学上的反卷积操作

### 代码实现

```
# 基本操作
def trans_conv(X,K):
	h,w = K.shape
	Y = torch.zeros(X.shape[0]-h+1,X.shape[1]-w+1)
	for i in range(X.shape[0]):
		for j in range(X.shape[1]):
			Y[i:i+h,j:j+w]+=X[i,j]*K
	return Y
	
# 使用高级API获得相同的结果
# batch,通道，高宽
X, K = X.reshape(1, 1, 2, 2), K.reshape(1, 1, 2, 2)
# 输入通道，输出通道，核大小
tconv = nn.ConvTranspose2d(1, 1, kernel_size=2, bias=False)
# 设置卷积参数
tconv.weight.data = K
#转置卷积调用API运算
tconv(X)
```

##### **填充、步幅和多通道**

常规卷积的填充放在输入，转置卷积的填充放在输出。例如，当将高和宽两侧的填充数指定为1时，转置卷积的输出中将删除第一和最后的行与列。

步幅被指定为中间结果（输出）

对于多个输入和输出通道，转置卷积与常规卷积以相同方式运作。

##### **与矩阵变换的联系**

用矩阵乘法来实现卷积

```
X = torch.arrange(9.0).reshape(3,3)
K = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
Y = d2l.corr2d(X,K)

'''卷积核K重写为包含大量0的稀疏权重矩阵W。 权重矩阵的形状是（4，9），其中非0元素来自卷积核K。'''

def kernel2matrix(K)：
	k,W=torch.zeros(5),torch.zeros((4,9))
	k[:2],k[3:] = K[0,:],K[1:]
	W[0, :5], W[1, 1:6], W[2, 3:8], W[3, 4:] = k, k, k, k
    return W
    
'''逐行连结输入X，获得了一个长度为9的矢量。 然后，W的矩阵乘法和向量化的X给出了一个长度为4的向量。 重塑它之后，可以获得与上面的原始卷积操作所得相同的结果Y：我们刚刚使用矩阵乘法实现了卷积。'''
```

同理可以实现转置卷积。想要通过矩阵相乘来实现它，我们只需要将权重矩阵`W`的形状转置为(9,4)。

```
Z = trans_conv(Y, K)
```







## 全连接卷积神经网络（FCN）

- FCN是深度神经网络来做语义分割的奠定性工作
- 他用**转置卷积层**来替换CNN最后的全连接层（还有池化层），从而实现每个像素的预测。（如果输入的图片是224 x 224，经过CNN变成 7 x 7，经过 transposed conv，可以还原到 224 x 224 x k，k为通道数）
- 保留空间信息

[![image](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/48/48-01.png)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/48/48-01.png)

### 代码

CNN卷积神经网络

1*1卷积层（把通道数改成类别大小）

转置卷积将特征高宽变化成输入图像的尺寸

```
%matplotlib inline
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

# 拿走resnet的框架去掉后面全连接层的部分
pretrained_net = torchvision.models.resnet18(pretrained=True)
net = nn.Sequential(*list(pretrained_net.children())[:-2])
```



```
'''给定高度为320和宽度为480的输入，net的前向传播将输入的高和宽减小至原来的，即10和15。'''
X = torch.rand(size=(1, 3, 320, 480))

'''使用卷积层将输出通道数转换为Pascal VOC2012数据集的类数（21类）。] 最后需要(将特征图的高度和宽度增加32倍)，从而将其变回输入图像的高和宽。'''

# 令 k=2*p+s，使得最终结果只增大s倍
num_classes = 21
net.add_module('final_conv',nn.Con2d(512, num_classes, kernel_size=1))
net.add_module('transpose_conv', nn.ConvTranspose2d(num_classes, num_classes,
    kernel_size=64, padding=16, stride=32))
```

##### 初始化转置卷积层

###### 双线性插值

1. 将输出图像的坐标(𝑥,𝑦)映射到输入图像的坐标(𝑥′,𝑦′)上。 例如，根据输入与输出的尺寸之比来映射。 请注意，映射后的𝑥′和𝑦′是实数。
2. 在输入图像上找到离坐标(𝑥′,𝑦′)最近的4个像素。
3. 输出图像在坐标(𝑥,𝑦)上的像素依据输入图像上这4个像素及其与(𝑥′,𝑦′)的相对距离来计算。

###### 核心步骤

**核心位置**：

- 核的中心位置取决于核大小 (`kernel_size`)。**对于奇数大小的核，中心是明确的；对于偶数大小的核，中心在两个像素之间**。

**核生成**：

- `og` 包含两个通过 `torch.arange` 生成的张量，分别**代表核的行和列**。这些张量用于计算每个位置与核中心的距离。
- `filt` 通过双线性插值公式计算，对每个元素的值进行计算，基于其到中心的距离。**核心思想是，距离中心越近的点权重越大**。

**权重矩阵初始化**：

- 初始化一个形状为 (in_channels, out_channels, kernel_size, kernel_size) 的零张量。**这保证了每个输入通道到每个输出通道都有一个对应的卷积核**。

**填充卷积核**：

- 将计算得到的双线性核 (`filt`) 复制到每一个输入通道对应的输出通道的位置。这样确保了所有的通道都使用相同的双线性插值核，这是因为双线性插值是通道无关的。

```
def bililinear_kernel(in_channels, out_channels, kernel_size):
	# 计算核中心点的位置
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = (torch.arange(kernel_size).reshape(-1,1),torch.arange(kernel_size).reshape(1,-1))    
    filt = (1 - torch.abs(og[0] - center) / factor) * \
           (1 - torch.abs(og[1] - center) / factor)
    weight = torch.zeros((in_channels, out_channels,kernel_size, kernel_size))
    weight[range(in_channels),
    range(out_channels),:,:] = filt
    
    return weight
    
```

全卷积网络[**用双线性插值的上采样初始化转置卷积层。对于1×1卷积层，我们使用Xavier初始化参数。**]

```
W = bilinear_kernel(num_classes, num_classes, 64)
net.transpose_conv.weight.data.copy_(W);
```



## 样式迁移

样式迁移类似于手机相机中的滤镜，指的是给定内容图片和风格图片，合成一张新的图片，使得他的内容与内容图片相似，而风格却是风格图片的样子，如下例：

[![样式迁移示例](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/49/%E6%A0%B7%E5%BC%8F%E8%BF%81%E7%A7%BB%E7%A4%BA%E4%BE%8B.png)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/49/样式迁移示例.png)

### 目的

[如何使用卷积神经网络，自动将一个图像中的风格应用在另一图像之上，即*风格迁移*（style transfer）]

输入图像：内容图像和风格图像。使用神经网络修改内容图像，使其在风格上接近风格图像。

### 方法

如下图所示，可以用一个预训练好的神经网络来实现样式迁移：

- 1：将内容图像作为初始化图像	，之后将这张图片中的像素作为参数进行更新，最终得到合成的图片。**该合成图像是风格迁移过程中唯一需要更新的变量**，即风格迁移所需迭代的模型参数。
- 2：将内容图片，当前的合成图片，样式图片分别输入一个相同的预训练好的神经网络
- 3：假设该神经网络的不同层分别提取与内容和风格相关的信息，可以据此得到一个损失：
  - 将合成图片与**内容**相关的层的输出与**内容**图片对应层的输出进行处理，得到**内容损失**
  - 将合成图片与**风格**相关的层的输出与**风格**图片对应层的输出进行处理，得到**风格损失**
  - 对合成图片本身，统计图片中的高频噪点（即过明或过暗像素点），得到**全变分损失**
  - 将三部分损失加起来得到总的样式迁移的**损失函数**
- 4：利用3定义的损失函数，以合成图片的每个像素点为参数进行梯度下降，得到最终合成的图片

[![CNN实现](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/49/CNN%E5%AE%9E%E7%8E%B0.png)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/49/CNN实现.png)

### 内容损失

神经网络内容相关层的输出的相似度可以直接反应两张图片在内容上的相似度，因此内容损失可以直接将对应层的输出视为内容直接求平方差损失。

### 风格损失：格拉姆矩阵

对于内容层，可以直接将对应层的输出求平方差损失，但是对于风格则略有不同

一般认为，风格层对应的输出的多个通道分别对应着不同类型的信息，如果将输出的形状从$[batch_size=1,channels,h,w]$转化为$[channels,h*w]$就能得到通道数个向量，以$x_1,x_2...x_c$表示，代表不同通道所提取出的不同信息，而风格可以视作这些信息之间的关联，即相似度。

定义格拉姆矩阵$\bold X * \bold X^T \in R^{c \cdot c}$，矩阵的第i行第j列即向量$x_i$与$x_j$的内积，这个矩阵就代表了一张图片的风格。

对于生成图片和风格图片的格拉姆矩阵求平方差损失就能得到所需的风格损失

此外，为了让风格损失不受格拉姆矩阵及向量的大小影响，实际将格拉姆矩阵除以这些大小$channnels*h*w$。

### 全变分损失

用于去除高频噪点（过明过暗像素点） $$ \sum_{i, j} \left|x_{i, j} - x_{i+1, j}\right| + \left|x_{i, j} - x_{i, j+1}\right| $$

### 代码

##### 阅读内容和风格图像

```
import torch
import torchvision
from torch import nn
from d2l import torch as d2l

# 读出图像大小
d2l.set_figsize()
content_img = d2l.Image.open(
				'../img/rainier.jpg')
d2l.plt.imshow(content_img)

style_img = d2l.Image.open(
				'../img/autumn-oak.jpg')
d2l.plt.imshow(style_img)
```

##### 预处理和后处理

预处理函数preprocess：对输入图像在RGB三个通道分别做标准化，并将结果变换成卷积神经网络接受的输入格式。 

​		**维度格式**：对于大多数深度学习框架（如PyTorch和TensorFlow），CNN接受的输入通常是一个四维张量，其形状为 `(N, C, H, W)`：

后处理函数postprocess：将输出图像中的像素值还原回标准化之前的值。 由于图像打印函数要求每个像素的浮点数值在0～1之间，我们对小于0和大于1的值分别取0和1。

- **可视化**：在进行图像处理或分析时，原始图像数据通常会被标准化或归一化以适应网络训练的需要。但为了可视化或展示这些图像，需要将处理后的数据还原到原始的数据范围（通常是0到255的整数），使图像对人眼来说是自然和理解的。
- **保存和使用**：如果要将处理后的图像用于其他应用，如图像编辑软件或其他图像处理流程，通常需要还原到标准的颜色范围内。这样做确保了图像的兼容性和实用性。
- **避免数据失真**：标准化过程可能会改变像素值的分布，对超出[0,1]范围的值进行裁剪（设置为0或1），这有助于保持数据的完整性和图像的质量。

```
rgb_mean = torch.tensor([0.485, 0.456, 0.406])
rgb_std = torch.tensor([0.229, 0.224, 0.225])

def progress(img,image_shape):
	transforms = torchvision.transforms.Compose([
torchvision.transforms.Resize(image_shape),
torchvision.transforms.ToTensor(),
torchvision.transforms.Normalize(mean=rgb_mean, std=rgb_std)])
    return transforms(img).unsqueeze(0)
    
    
def postprocess(img):
	img = img[0].to(rgb_std.device)
	img = torch.clamp(img.permute(1, 2, 0) * rgb_std + rgb_mean, 0, 1)
    
    return torchvision.transforms.
    ToPILImage()(img.permute(2, 0, 1))
```

##### 初始化合成图像

**合成的图像是训练期间唯一需要更新的变量**。因此，我们可以定义一个简单的模型`SynthesizedImage`，并将合成的图像视为模型参数。模型的前向传播只需返回模型参数即可。

```
class SynthesizedImage(nn.Module):
    def __init__(self, img_shape, **kwargs):
        super(SynthesizedImage, self).__init__(**kwargs)
        self.weight = nn.Parameter(torch.rand(*img_shape))

    def forward(self):
        return self.weight
        
'''创建了合成图像的模型实例，并将其初始化为图像X'''     
def get_inits(X,device,il,style_Y):
	gen_img = SynthesizedImage(X).to(device)
	gen_img.weight.data.copy_(X.data)
    trainer = torch.optim.Adam(gen_img.parameters(),lr=lr)
    styles_Y_gram = [gram(Y) for Y in styles_Y]
    return gen_img(), styles_Y_gram, trainer
```

##### 抽取图像特征

为了抽取图像的内容特征和风格特征，我们可以选择VGG网络中某些层的输出。 一般来说，越靠近输入层，越容易抽取图像的细节信息；反之，则越容易抽取图像的全局信息。 为了避免合成图像过多保留内容图像的细节，我们选择VGG较靠近输出的层，即*内容层*，来输出图像的内容特征。 我们还从VGG中选择不同层的输出来匹配局部和全局的风格，这些图层也称为*风格层*。 

使用VGG层抽取特征时，我们只需要用到**从输入层到最靠近输出层的内容层或风格层之间的所有层**。 下面构建一个新的网络`net`，它只保留需要用到的VGG的所有层。(层的裁剪，防止过多运算)

给定输入`X`，如果我们简单地调用前向传播`net(X)`，只能获得最后一层的输出。 由于我们还需要中间层的输出，因此这里我们逐层计算，并保留内容层和风格层的输出。

```
# 我们使用基于ImageNet数据集预训练的VGG-19模型来抽取图像特征
pretrained_net = torchvision.models.vgg19(pretrained=True)

style_layers, content_layers = [0, 5, 10, 19, 28], [25]

net = nn.Sequential(*[pretrained_net.features[i] for i in range(max(content_layers + style_layers) + 1)])

def extract_features(X,content_layers, style_layers):
	contents = []
    styles = []
    for i in range(len(net)):
    	X = net[i](X)
        if i in style_layers:
            styles.append(X)
        if i in content_layers:
            contents.append(X)
    return contents, styles
    
    
'''对内容图像抽取内容特征'''
def get_contents(image_shape, device):
    content_X = preprocess(content_img, image_shape).to(device)
    contents_Y, _ = extract_features(content_X, content_layers, style_layers)
    return content_X, contents_Y

def get_styles(image_shape, device):
    style_X = preprocess(style_img, image_shape).to(device)
    _, styles_Y = extract_features(style_X, content_layers, style_layers)
    return style_X, styles_Y

```

##### 定义损失函数

风格迁移的损失函数。 它由**内容损失、风格损失和全变分损失**3部分组成。

###### 内容损失

```
def content_loss(Y_hat, Y):
    # 我们从动态计算梯度的树中分离目标：
    # 这是一个规定的值，而不是一个变量。
    return torch.square(Y_hat - Y.detach()).mean()
```

###### 风格损失

通过平方误差函数衡量合成图像与风格图像在风格上的差异。 

为了表达风格层输出的风格，我们先通过`extract_features`函数计算风格层的输出。 假设该输出的样本数为1，通道数为𝑐，高和宽分别为ℎ和𝑤，我们可以将此输出转换为矩阵𝑋，其有𝑐行和ℎ𝑤列。 

这个矩阵可以被看作由𝑐个长度为ℎ𝑤的向量𝑥1,…,𝑥𝑐组合而成的。其中向量𝑥𝑖代表了通道𝑖上的风格特征。

###### 格拉姆矩阵（Gram matrix）

![image-20240725231641925](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20240725231641925.png)

###### 格拉姆矩阵归一化

![image-20240725231753335](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20240725231753335.png)

```
def Gram(X):
	num_channels,n = X.shape[1],X.numel// X.shape[1]
	X = X.reshape((num_channels,n))
	return torch.matmul(X, X.T) / (num_channels * n)
def style_loss(Y_hat, gram_Y):
    return 
    torch.square(gram(Y_hat)-gram_Y.detach()).mean()
```

###### 全变分损失

有时候，我们学到的合成图像里面有大量高频噪点，即有特别亮或者特别暗的颗粒像素。 一种常见的去噪方法是*全变分去噪*（total variation denoising）。全变分损失的目的是鼓励生成图像的平滑性，同时保留图像的关键结构。

![image-20240725232459534](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20240725232459534.png)

![image-20240725232605643](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20240725232605643.png)

```
def tv_loss(Y_hat):
    return 0.5 * (torch.abs(Y_hat[:, :, 1:, :] - Y_hat[:, :, :-1, :]).mean() +
                  torch.abs(Y_hat[:, :, :, 1:] - Y_hat[:, :, :, :-1]).mean())

```

损失函数：**风格转移的损失函数是内容损失、风格损失和总变化损失的加权和**

```
content_weight, style_weight, tv_weight = 1, 1e3, 10

def compute_loss(X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram):
    # 分别计算内容损失、风格损失和全变分损失
    contents_l = [content_loss(Y_hat, Y) * content_weight for Y_hat, Y in zip(
        contents_Y_hat, contents_Y)]
    styles_l = [style_loss(Y_hat, Y) * style_weight for Y_hat, Y in zip(
        styles_Y_hat, styles_Y_gram)]
    tv_l = tv_loss(X) * tv_weight
    # 对所有损失求和
    l = sum(10 * styles_l + contents_l + [tv_l])
    return contents_l, styles_l, tv_l, l
```

##### 训练模型

```
def train(X, contents_Y, styles_Y, device, lr, num_epochs, lr_decay_epoch):
    X, styles_Y_gram, trainer = get_inits(X, device, lr, styles_Y)
    scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_decay_epoch, 0.8)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[10, num_epochs],
                            legend=['content', 'style', 'TV'],
                            ncols=2, figsize=(7, 2.5))
    for epoch in range(num_epochs):
        trainer.zero_grad()
        contents_Y_hat, styles_Y_hat = extract_features(
            X, content_layers, style_layers)
        contents_l, styles_l, tv_l, l = compute_loss(
            X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram)
        l.backward()
        trainer.step()
        scheduler.step()
        if (epoch + 1) % 10 == 0:
            animator.axes[1].imshow(postprocess(X))
            animator.add(epoch + 1, [float(sum(contents_l)),
                                     float(sum(styles_l)), float(tv_l)])
    return X
```



# 循环神经网络

## 序列模型

### 序列数据

- 实际中很多数据是有时序的
- 电影的评价随时间变化而变化
  - 拿了奖后评分上升，直到奖项被遗忘
  - 看了很多好电影后，人们的期望变高
  - 季节性：贺岁片，暑期档
  - 导演、演员的负面报道导致评分变低

#### 更多例子

- 音乐、文本、语言和视频都是连续的
  - 标题“狗咬人”远没有“人咬狗”那么令人惊讶
- 大地震发生后，很有可能会有几次较小的余震
- 人的互动是连续的，从网上吵架可以看出
- 预测明天的股价要比填补昨天遗失的股价更困难

### 统计工具

这T个数据是不独立的随机变量，这是序列模型独特的一点。我们对这样的变量做条件概率。

- 在时间t观察到[![X_{t}](https://camo.githubusercontent.com/a6b078b3e9cfbca5273206a921e9d5ca1c7164b9d0df475c706c0853ac54ce62/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e696d6167653f585f7b747d)](https://camo.githubusercontent.com/a6b078b3e9cfbca5273206a921e9d5ca1c7164b9d0df475c706c0853ac54ce62/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e696d6167653f585f7b747d)，那么得到T个不独立的随机变量。请注意，𝑡对于本文中的序列通常是离散的，并在整数或其子集上变化。 [![(X_{1},...X_{t})\sim p(X)](https://camo.githubusercontent.com/0823b3b6d3c561bf006d83ea097956be13243e4df9f6b1bc1822eef85c3f98fc/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e696d6167653f28585f7b317d2c2e2e2e585f7b747d295c73696d2673706163653b70285829)](https://camo.githubusercontent.com/0823b3b6d3c561bf006d83ea097956be13243e4df9f6b1bc1822eef85c3f98fc/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e696d6167653f28585f7b317d2c2e2e2e585f7b747d295c73696d2673706163653b70285829)

- 使用条件概率展开

  ![p(a,b)=p(a)p(b|a)=p(b)p(a|b)](https://camo.githubusercontent.com/ae9f738f89f4a2a6e6da4e9214b40976dd46f21af02a1433bf2f7a5eadf5b5a2/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e696d6167653f7028612c62293d702861297028627c61293d702862297028617c6229)

[![image](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/51/51-01.png)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/51/51-01.png)

### 序列模型

自回归模型：预测某个数据时，不调用其他数据，而采用该数据前面的数据。



[![image](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/51/51-02.png)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/51/51-02.png)

- 对条件概率建模

[![image](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/51/51-03.png)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/51/51-03.png)

#### 方案A:马尔科夫假设



[![image](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/51/51-04.png)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/51/51-04.png)

- 假设当前数据**只跟τ个过去数据点相关**

[![image](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/51/51-05.png)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/51/51-05.png)

![image-20240726113147509](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20240726113147509.png)

#### 方案B:潜变量模型

是保留一些对**过去观测的总结ℎ𝑡**， **并且同时更新预测𝑥𝑡和总结ℎ𝑡**。 这就产生了基于𝑥^𝑡=𝑃(𝑥𝑡∣ℎ𝑡)估计𝑥𝑡， 以及公式ℎ𝑡=𝑔(ℎ𝑡−1,𝑥𝑡−1)更新的模型。

[![image](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/51/51-06.png)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/51/51-06.png)

- 引入潜变量，潜变量是对历史信息的记录；潜变量是由上一个潜变量和上一个时间点推算得出的。

  ![h_{t}](https://camo.githubusercontent.com/7c2a48b1354ae91f91c4bc95fc0bb6b8b1e97687dbeb2542811a438c934dace1/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e696d6167653f685f7b747d)

  来表示过去信息

  ![h_{t}=f(x_{1},...x_{t-1})](https://camo.githubusercontent.com/2097047baee3ee12ab5dd1a6c6da5ddb8e463184e83e46a2e2c58261dc2b1ce8/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e696d6167653f685f7b747d3d6628785f7b317d2c2e2e2e785f7b742d317d29)

  - 这样[![x_{t}=p(x_{t}|h_{t})](https://camo.githubusercontent.com/c75dd808b220d03ff8eeba6a816465c31a9d10f6900f6f1f3e856efd62915ba0/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e696d6167653f785f7b747d3d7028785f7b747d7c685f7b747d29)](https://camo.githubusercontent.com/c75dd808b220d03ff8eeba6a816465c31a9d10f6900f6f1f3e856efd62915ba0/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e696d6167653f785f7b747d3d7028785f7b747d7c685f7b747d29)

[![image](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/51/51-07.png)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/51/51-07.png)

### 总结

- 时序模型中，当前数据跟之前观察到的数据相关
- 自回归模型使用自身过去数据来预测未来
- 马尔科夫模型假设当前只跟最近少数数据相关，从而简化模型
- 潜变量模型使用潜变量来概括历史信息

### 代码

**使用马尔可夫假设来预测模型，**使用正弦函数和一些可加性噪声来生成序列数据， 时间步为1,2,…,1000。

```
T = 1000
time = torch.arange(1,T+1,dtype = torch.float32)
X = torch.sin(0.01*T)+torch.normal(0,0.2,(T,))
```

将这个序列转换为模型的*特征－标签*（feature-label）对。 基于嵌入维度𝜏，我们[**将数据映射为数据对𝑦𝑡=𝑥𝑡 和𝑥𝑡=[𝑥𝑡−𝜏,…,𝑥𝑡−1]。**] 

缺少𝜏个数据样本：丢弃；用0填充。

```
tau = 4
features = torch.zeros((T-tau,tau))
for i in range(tau):
	features[:,i] = X[i:i+T-tau]
labels = x[tau:].reshape((-1,1))
```

```
batch_size, n_train = 16, 600
# 只有前n_train个样本用于训练
train_iter = d2l.load_array((features[:n_train], labels[:n_train]), batch_size, is_train=True)

'''在这里，我们[使用一个相当简单的架构训练模型： 一个拥有两个全连接层的多层感知机]，ReLU激活函数和平方损失。'''
# 初始化网络权重的函数
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)

# 一个简单的多层感知机
def get_net():
    net = nn.Sequential(nn.Linear(4, 10),
                        nn.ReLU(),
                        nn.Linear(10, 1))
    net.apply(init_weights)
    return net

# 平方损失。注意：MSELoss计算平方误差时不带系数1/2
loss = nn.MSELoss(reduction='none')

# 训练模型
def train(net, train_iter, loss, epochs, lr):
    trainer = torch.optim.Adam(net.parameters(), lr)
    for epoch in range(epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.sum().backward()
            trainer.step()
        print(f'epoch {epoch + 1}, '
              f'loss: {d2l.evaluate_loss(net, train_iter, loss):f}')

net = get_net()
train(net, train_iter, loss, 5, 0.01)
```

##### 预测

首先是检查[**模型预测下一个时间步**]的能力， 也就是*单步预测*（one-step-ahead prediction）。

```
onestep_preds = net(features)
d2l.plot([time,time[tau:]],'time','x',legend = ['data','1-step preds'],xlim=[1,1000],figsize=(6,3))
```

通常，对于直到𝑥𝑡的观测序列，其在时间步𝑡+𝑘处的预测输出𝑥^𝑡+𝑘 称为𝑘*步预测*（𝑘-step-ahead-prediction）。 由于我们的观察已经到了𝑥604，它的𝑘步预测是𝑥^604+𝑘。 换句话说，我们必须使用我们自己的预测（而不是原始数据）来[**进行多步预测**]。 让我们看看效果如何。

```
multistep_preds = torch.zeros(T)
multistep_preds[: n_train + tau] = x[: n_train + tau]
for i in range(n_train + tau, T):
    multistep_preds[i] = net(
        multistep_preds[i - tau:i].reshape((1, -1)))
        
d2l.plot([time, time[tau:], time[n_train + tau:]],
         [x.detach().numpy(), onestep_preds.detach().numpy(),
          multistep_preds[n_train + tau:].detach().numpy()], 'time',
         'x', legend=['data', '1-step preds', 'multistep preds'],
         xlim=[1, 1000], figsize=(6, 3))
```

## 文本预处理

数据存在许多种形式，文本是最常见例子之一。 例如，一篇文章可以被简单地看作一串单词序列，甚至是一串字符序列。 本节中，我们将解析文本的常见预处理步骤。 这些步骤通常包括：

1. 将文本作为字符串加载到内存中。
2. 将字符串拆分为词元（如单词和字符）。
3. 建立一个词表，将拆分的词元映射到数字索引。
4. 将文本转换为数字索引序列，方便模型操作。

#### 读取数据集

```
d2l.DATA_HUB["time_machine"] = (d2l.DATA_URL+"timemachine.txt"+'090b5e7e70c295757f55df93cb0a180b9691891a')

def read_time_machine():
	 with open(d2l.download('time_machine'), 'r') as f:
        lines = f.readlines()
    # import re: 导入 Python 的正则表达式库，允许你使用 re.sub() 等函数
	return [re.sub('[^A-Za-z]+,'',line).strip().lower() for line in lines]
	
lines = read_time_machine()
print(f'# 文本总行数: {len(lines)}')
print(lines[0])
print(lines[10])
```

#### 词元化

- 列表中的每一个”line“都是一个文本序列/文本行
- 每个文本行会被拆分成一个词元列表
- 词元列表是文本最基本的单位

```
def tokennize(lines,token="word"):
	if token == "word":
		return [line.split() for line in lines]
	elif token == "char":
		return [list(line) for line in lines]
	else:
        print('错误：未知词元类型：' + token)

tokens = tokenize(lines)
```

#### 词表

**构建一个字典，通常也叫做词表（vocabulary）， 用来将字符串类型的词元映射到从0开始的数字索引中**

将训练集中的所有**文档合并**在一起，对它们的**唯一词元进行统计**， 得到的统计结果称之为***语料*（corpus）**。 

然后根据每个唯一词元的出现频率，为其**分配一个数字索引**。 很少出现的词元通常被移除，这可以降低复杂性。 另外，语料库中不存在或已删除的任何词元都将映射到一个特定的未知词元“<unk>”。 我们可以选择增加一个列表，用于保存那些被保留的词元， 例如：填充词元（“<pad>”）； 序列开始词元（“<bos>”）； 序列结束词元（“<eos>”）。

```python
class Vocab:
	def __init__(self,tokens=None,min_freq=0,reversed_token=None):
	if tokens is None:
        tokens = []
    if reserved_tokens is None:
        reserved_tokens = []
    counter = count_corpus(tokens)
    Counter = count_corpus(tokens)
    self._token_freqs = sorted(Counter.item(),key=lambda x:x[1],reversed=True)
    # 未知词元索引为 0
    self.idx_to_token = ['<unk>'] + reserved_tokens
    self.token_to_idx = {token:idx for idx,token in enumerate(self.idx_to_token)}
    
    for token,freq in self._token_freq:
        if freq < min_freq:
                break
         if token not in self.token_to_idx:
            self.idx_to_token.append(token)
            self.token_to_idx[token] = len(self.idx_to_token) - 1
              
    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]
    
    def to_token(self,indices):
        if not isinstance(indices,(list,tuple))L
			return self.idx_to_token[indices]
		return [self.idx_to_token[index] for index in indices]   
    
    def unk(self):  # 未知词元的索引为0
        return 0

    @property
    def token_freqs(self):
        return self._token_freqs
        
    
def count_corpus(tokens):
    """统计词元的频率"""
    if len(token) == 0 or isinstance(token[0],list):
        tokens = [token for line in tokens for token in line]
    return collection.Counter(tokens)
```

#### 整合所有功能

```
def load_corpus_time_machine(max_tokens=-1):  #@save
    """返回时光机器数据集的词元索引列表和词表"""
    # 读取数据集
    lines = read_time_machine()
    # 词元化
    tokens = tokenize(lines, 'char')
    # 构建词表
    vocab = Vocab(tokens)
    # 因为时光机器数据集中的每个文本行不一定是一个句子或一个段落，
    # 所以将所有文本行展平到一个列表中
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab

corpus, vocab = load_corpus_time_machine()
len(corpus), len(vocab)
```

## 语言模型和数据集

### 代码



## 循环神经网络（RNN）

### 使用潜变量

RNN使用了隐藏层来记录过去发生的所有事件的信息，从而引入时许的特性，并且避免常规序列模型每次都要重新计算前面所有已发生的事件而带来的巨大计算量。[![截屏2022-02-12 下午2.17.32](https://github.com/kinza99/DeepLearning-MuLi-Notes/raw/main/imgs/54/54-01.png)](https://github.com/kinza99/DeepLearning-MuLi-Notes/blob/main/imgs/54/54-01.png)

xt与当前状态的xt-1与ht相关

ht与当前状态的xt-1与ht-1相关

### RNN

- 我们可以观察到x，也有一个我们无法观察到但是存在的h（hidden_variable）
- 流程如下，首先有一个输入序列，对于时刻t，我们用t-1时刻的输入 x~t-1~ 和潜变量h~t-1~ 来计算新的潜变量h~t~。同时，直接使用h~~t~~来计算得到t时刻的输出o~~t~~。注意，计算第一个潜变量只需要输入即可（因为前面并不存在以往的潜变量）。[![截屏2022-02-12 下午2.34.14](https://github.com/kinza99/DeepLearning-MuLi-Notes/raw/main/imgs/54/54-02.png)](https://github.com/kinza99/DeepLearning-MuLi-Notes/blob/main/imgs/54/54-02.png)
- 值得注意的是，RNN本质也是一种MLP，尤其是将h~~t-1~~这一项去掉时就完全退化成了MLP。RNN的核心其实也就是h~~t-1~~这一项，它使得模型可以和前面的信息联系起来，将时序信息储存起来，可以把RNN理解为是包含时序信息的MLP。

### 困惑度

- 为了衡量一个语言模型的好坏，例如分类模型，可以使用平均交叉熵来衡量，就是将预测概率的负对数值求和之后再去平均，即常用的交叉熵损失。但是由于某些历史原因，NLP往往不是用这种方式，而是在这种方式的基础上最后再取指数，即exp，这样得到的结果如果是1，说明完美；如果是无穷大，说明结果很差。

### 梯度裁剪

- 在T个时间步中进行反向传播，会由于产生O(T)长度的梯度乘法链，导致导数数值不稳定(梯度爆炸或梯度消失)，这里使用一个限制θ，通常为5到10，来控制梯度乘法链的长度。使用如下的公式[![截屏2022-02-12 下午3.10.38](https://github.com/kinza99/DeepLearning-MuLi-Notes/raw/main/imgs/54/54-03.png)](https://github.com/kinza99/DeepLearning-MuLi-Notes/blob/main/imgs/54/54-03.png)在这个公式中，如果梯度长度大于θ，梯度g会变为 $$ \frac{\theta}{\Vert g \Vert} g $$ 这样再对g求2范式就变成了θ，所以可以把梯度限制在θ以下。
- g中表示所有的梯度合并在一起 

### 更多的应用RNNs

- 基本的应用如下图[![截屏2022-02-12 下午3.36.29](https://github.com/kinza99/DeepLearning-MuLi-Notes/raw/main/imgs/54/54-04.png)](https://github.com/kinza99/DeepLearning-MuLi-Notes/blob/main/imgs/54/54-04.png)

### QA总结

- 循环神经网络更适合处理smooth的sequence
- 小规模的模型构建在没有硬件支持的情况下更适合去做应用

### 循环神经网络的从零开始实现

RNN本质上跟其他模型最大的区别就是引入了序列模型中的潜变量来预测下一个信息。对于潜变量中的序列选择，我们将从一个比较大的batch中选出minibatch，这个minibatch可以是随机选择也可以是连续分布。

##### 读取数据

```
import math
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

batch_size,num_step = 32,35
train_iter,vocab = d2l.load_data_time_machine(batch_size,num_steps)
```

##### one_hot encode

我们将词元映射为数字索引，再将其通过**独热编码**转变成更具表现力的特征向量：将每个索引映射为相互不同的单位向量： 假设词表中不同词元的数目为𝑁（即`len(vocab)`）， 词元索引的范围为0到𝑁−1。 如果词元的索引是整数𝑖， 那么我们将创建一个长度为𝑁的全0向量， 并将第𝑖处的元素设置为1。 此向量是原始词元的一个独热向量。

```
X = torch.arange(10).reshape((2, 5))
F.one_hot(X.T, 28).shape
```

##### 初始化模型参数

特别的地方在于隐藏单元组的引入，num_hiddens是一个可调的超参数：

```
def get_param(vocab_size, num_hiddens, device):
	num_inputs = num_outputs = vocab_size
	def normal(shape):
		return torch.randn(size=shape,device=device).*0.0.1
		
	W_xh = normal((num_inputs,num_hiddens))
	W_hh = normal((num_hiddens,num_hiddens))
	b_h = torch.zeros(num_hiddens, device=device)
	
	W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)
    
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params
```

首先需要[**一个`init_rnn_state`函数在初始化时返回隐状态**]

```
def int__rnn_state(batch_size,num_hiddens,device):
	return (torch,zeros((batch_size,num_hiddens),device=device),)
```

**定义了如何在一个时间步内计算隐状态和输出。**这里使用tanh函数作为激活函数。

```
def rnn(inputs,state,device):
	# inputs (时间步数量，批量大小，词表大小)
	W_xh, W_hh, W_hq, b_h, b_q = params
	H, = state
    outputs = []
    for X in inputs:
        H = torch.tanh(torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h)
        Y = torch.mm(H, W_hq) + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H,)
```

**创建一个类来包装这些函数**

```
class RNNModelScratch:
	def __init__(self,volcab_size,num_hiddens,device,gget_params,init_state,forward_fn):
	# 对于rnn：规定词表大小，隐变量数量，各种运算推导参数
	self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)
        self.init_state, self.forward_fn = init_state, forward_fn
        
    def __call__(self,X,state):
    	X = F.one_hot(X.T,self.vocab_size).type(torch.float32)
    	
    def begin_state(self, batch_size, device):
    	return self.init_state(batch_size,self.num_hiddens,device)
```

#### 预测

**首先定义预测函数来生成`prefix`之后的新字符**

其中的`prefix`是一个用户提供的包含多个字符的字符串。 在循环遍历`prefix`中的开始字符时， 我们不断地将隐状态传递到下一个时间步，但是不生成任何输出。 这被称为***预热*（warm-up）期**， 因为在此期间模型会自我更新（例如，更新隐状态）， 但不会进行预测。 预热期结束后，隐状态的值通常比刚开始的初始值更适合预测， 从而预测字符并输出它们。**更新参数。**

```
def predict_ch8(prefix,num_preds,net,vocab,device):
	state = net.begin_state(batch_size=1,device=device)
	outputs[vocab[prefix[0]]]
	get_input = lambda torch.tensor([outputs[-1]],device=device).reshape(1,1)
	for y in prefix[1:]:
		_,state = net(get_input(),state)
		# 先转化成索引，有利于后期预测
		outputs.append(vocab[y])
	for _ in range(num_preds):
		y,state = net(get_input(),state)
		outputs.append(y.argmax(dim=1).reshape(1))
	return ''.join([vocab.idx_to_token[i] for i in outputs])
```

#### 梯度裁剪

通过将梯度𝑔投影回给定半径 （例如𝜃）的球来裁剪梯度𝑔。 如下式：

![image-20240727111721132](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20240727111721132.png)

通过这样做，我们知道**梯度范数永远不会超过𝜃**， 并且更新后的梯度完全与𝑔的原始方向对齐。 它还有一个值得拥有的副作用， **即限制任何给定的小批量数据（以及其中任何给定的样本）对参数向量的影响**（意思就是降低异常点的影响）， 这赋予了模型一定程度的稳定性。 梯度裁剪提供了一个快速修复梯度爆炸的方法， 虽然它并不能完全解决问题，但它是众多有效的技术之一。

```
def grad_clipping(net,theta):
	 if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(torch.sum(p.grad**2) for p in params)
    if norm>theta:
		for param in params:
			param.grad[:]*=theta/norm
```

#### 训练

##### **定义一个函数在一个迭代周期内训练模型**

1. 序列数据的不同采样方法（随机采样和顺序分区）将导致隐状态初始化的差异。
2. 我们在更新模型参数之前裁剪梯度。 这样的操作的目的是，即使训练过程中某个点上发生了梯度爆炸，也能保证模型不会发散。
3. 我们用困惑度来评价模型。如 :numref:`subsec_perplexity`所述， 这样的度量确保了不同长度的序列具有可比性。

```
#@save
def train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter):
    """训练网络一个迭代周期（定义见第8章）"""
    state, timer = None, d2l.Timer()
    metric = d2l.Accumulator(2)  # 训练损失之和,词元数量
    for X, Y in train_iter:
        if state is None or use  _random_iter:
            # 在第一次迭代或使用随机抽样时初始化state
            state = net.begin_state(batch_size=X.shape[0], device=device)
        else:
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
                # state对于nn.GRU是个张量
                state.detach_()
            else:
                # state对于nn.LSTM或对于我们从零开始实现的模型是个张量
                for s in state:
                    s.detach_()
        y = Y.T.reshape(-1)
        X, y = X.to(device), y.to(device)
        y_hat, state = net(X, state)
        l = loss(y_hat, y.long()).mean()
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            grad_clipping(net, 1)
            updater.step()
        else:
            l.backward()
            grad_clipping(net, 1)
            # 因为已经调用了mean函数
            updater(batch_size=1)
        metric.add(l * y.numel(), y.numel())
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()
```

[**循环神经网络模型的训练函数既支持从零开始实现， 也可以使用高级API来实现。**]

```
#@save
def train_ch8(net, train_iter, vocab, lr, num_epochs, device,
              use_random_iter=False):
    """训练模型（定义见第8章）"""
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', ylabel='perplexity',
                            legend=['train'], xlim=[10, num_epochs])
    # 初始化
    if isinstance(net, nn.Module):
        updater = torch.optim.SGD(net.parameters(), lr)
    else:
        updater = lambda batch_size: d2l.sgd(net.params, lr, batch_size)
    predict = lambda prefix: predict_ch8(prefix, 50, net, vocab, device)
    # 训练和预测
    for epoch in range(num_epochs):
        ppl, speed = train_epoch_ch8(
            net, train_iter, loss, updater, device, use_random_iter)
        if (epoch + 1) % 10 == 0:
            print(predict('time traveller'))
            animator.add(epoch + 1, [ppl])
    print(f'困惑度 {ppl:.1f}, {speed:.1f} 词元/秒 {str(device)}')
    print(predict('time traveller'))
    print(predict('traveller'))
```

### 循环神经网络的简洁实现

```
num_hiddens = 256
rnn_layer = nn.RNN(len(vocab),num_hiddens)

state = torch.zeros(1,batch_size,num_hiddens)

'''通过一个隐状态和一个输入，我们就可以用更新后的隐状态计算输出。'''
X = torch.rand(size=(num_steps, batch_size, len(vocab)))
Y, state_new = rnn_layer(X, state)
```

[**我们为一个完整的循环神经网络模型定义了一个`RNNModel`类**]。 注意，`rnn_layer`只包含隐藏的循环层，我们还需要创建一个单独的输出层。

```
#@save
class RNNModel(nn.Module):
    """循环神经网络模型"""
    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.num_hiddens = self.rnn.hidden_size
        # 如果RNN是双向的（之后将介绍），num_directions应该是2，否则应该是1
        if not self.rnn.bidirectional:
            self.num_directions = 1
            self.linear = nn.Linear(self.num_hiddens, self.vocab_size)
        else:
            self.num_directions = 2
            self.linear = nn.Linear(self.num_hiddens * 2, self.vocab_size)

    def forward(self, inputs, state):
        X = F.one_hot(inputs.T.long(), self.vocab_size)
        X = X.to(torch.float32)
        Y, state = self.rnn(X, state)
        # 全连接层首先将Y的形状改为(时间步数*批量大小,隐藏单元数)
        # 它的输出形状是(时间步数*批量大小,词表大小)。
        output = self.linear(Y.reshape((-1, Y.shape[-1])))
        return output, state

    def begin_state(self, device, batch_size=1):
        if not isinstance(self.rnn, nn.LSTM):
            # nn.GRU以张量作为隐状态
            return  torch.zeros((self.num_directions * self.rnn.num_layers,
                                 batch_size, self.num_hiddens),
                                device=device)
        else:
            # nn.LSTM以元组作为隐状态
            return (torch.zeros((
                self.num_directions * self.rnn.num_layers,
                batch_size, self.num_hiddens), device=device),
                    torch.zeros((
                        self.num_directions * self.rnn.num_layers,
                        batch_size, self.num_hiddens), device=device))
```



# 现代循环神经网络

## GRU

### 动机：如何关注一个序列



- 不是每个观察值都是同等重要

[![img](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/56/56-01.png)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/56/56-01.png)

比如上图中的序列，若干个猫中出现了一个鼠，那么我们应该重点关注这个鼠，而中间重复出现的猫则减少关注。文本序列同理，通常长文本我们需要关注的是几个关键词，关键句。

- 想只记住相关的观察需要：
  - 能**关注**的机制（**更新门**）：顾名思义，是否需要根据我的输入，更新隐藏状态
  - 能**遗忘**的机制（**重置门**）：更新候选项时，是否要考虑前一隐藏状态。

### 门的概念

- 更新门Zt，重置门Rt的公式大体相同，唯一不同的是学习到的参数。
- 需要注意的是，计算门的方式和原来RNN的实现中计算新的隐状态相似，只是激活函数改成了sigmoid。
- 门本来是电路中的一个概念，0,1代表不同的电平，可以用于控制电路的通断。此处sigmoid将门的数值归一化到0到1之间，是一种"软更新"方式。而从后面的公式上可以看出，本讲课程采用的是低电平有效（越靠近0，门的作用越明显）的方式控制。

[![img](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/56/56-02.png)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/56/56-02.png)

### 候选隐状态

[![img](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/56/56-03.png)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/56/56-03.png)

- 候选隐状态，如果抛开公式中的$R_{t}$遗忘门来说，这个和之前RNN中计算当前步的隐状态没有差别。
- 但是这里引入了遗忘门，如果$R_{t}$无限接近于0，那么此时候选隐状态将不再考虑前一隐状态的影响，也就是和MLP没有区别，起到“遗忘”的作用；
- 反之，如果$R_{t}$无限接近于1，那么与RNN计算隐状态的过程没有差别，不进行遗忘。
- 公式中的⊙表示逐元素乘积。

> 为什么叫候选隐状态？
>
> 在RNN中，这个所谓的候选隐状态就是当前步的隐状态（$R_{t}$无限接近1时）。但是由于引入了更新门，我们需要考虑是直接沿用上一步的隐藏状态，还是像RNN一样使用当前步计算的隐状态。所以这个结合了当前输入计算的隐状态，不能立马变成当前的$H_{t}$，而是需要用更新门和前一隐状态$H_{t-1}$做一个加权，所以它是一个候选项。

### 隐状态

[![img](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/56/56-04.png)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/56/56-04.png)

用更新门对**候选隐状态**和**前一隐状态**做加权，得到当前步**隐状态**的值。

如果$Z_{t}$无限接近于0，更新起作用，候选隐状态“转正”，变为当前隐状态。

如果$Z_{t}$无限接近于1，更新不起作用，当前隐状态还是沿用前一隐状态。

### 总结

[![img](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/56/56-05.png)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/56/56-05.png)

上图四行公式概括了GRU模型。在RNN的基础上，最重要的是引入了**更新门和重置门**，来决定前一隐状态对当前隐状态的影响。以最开始的猫鼠序列的例子来说，如果我的模型一直看到猫，模型可以学习到隐状态不怎么去更新，于是隐状态一直保留了猫的信息，而看到老鼠，隐状态才进行更新。

- 对于一个更具体的例子而言(语言模型)：

“The cat, which already ate ……, __(is/ was) full.”，假设我的句子很长，预测完前面的词后需要预测下一个词is还是was，如果引入这种更新/重置的机制，那我们的模型可以在was这个词之前尽可能去保持隐状态的信息，从而即使阅读了一个很长的定语从句，但我们还是保留了cat这个词的单数信息，从而模型预测下一个词为'was'。

- 一个与RNN的联动在于：

如果更新门完全发挥作用（无限接近于0），重置门不起作用（无限接近于1），此时GRU模型退化为RNN模型。

## LSTM/长短期记忆网络

### 长短期记忆网络：

- 忘记门：将值朝0减少
- 输入门：决定是不是忽略掉输入数据
- 输出门：决定是不是使用隐状态

可以说，长短期记忆网络的设计灵感来自于计算机的逻辑门。 长短期记忆网络引入了*记忆元*（memory cell），或简称为*单元*（cell）。 有些文献认为记忆元是隐状态的一种特殊类型， 它们与隐状态具有相同的形状，其设计目的是用于记录附加的信息。 为了控制记忆元，我们需要许多门。 其中一个门用来从单元中输出条目，我们将其称为*输出门*（output gate）。 另外一个门用来决定何时将数据读入单元，我们将其称为*输入门*（input gate）。 我们还需要一种机制来重置单元的内容，由*遗忘门*（forget gate）来管理， 这种设计的动机与门控循环单元相同， 能够通过专用机制决定什么时候记忆或忽略隐状态中的输入。 让我们看看这在实践中是如何运作的。

#### 门：

输入门：[![I_{t}=\sigma (X_{t}W_{xi}+H_{t-1}W_{hi}+b_{i})](https://camo.githubusercontent.com/5d35b6208b87b36a2fa57656db4d4e55d39d25dc8fc186a457e6c069822d04c3/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e696d6167653f495f7b747d3d5c7369676d612673706163653b28585f7b747d575f7b78697d2b485f7b742d317d575f7b68697d2b625f7b697d29)](https://camo.githubusercontent.com/5d35b6208b87b36a2fa57656db4d4e55d39d25dc8fc186a457e6c069822d04c3/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e696d6167653f495f7b747d3d5c7369676d612673706163653b28585f7b747d575f7b78697d2b485f7b742d317d575f7b68697d2b625f7b697d29)

忘记门：[![F_{t}=\sigma (X_{t}W_{xf}+H_{t-1}W_{hf}+b_{f})](https://camo.githubusercontent.com/edf8b8b9552b05ac6d234ebff09ed29f5839e6bb99ff8b01f612b82454df07b7/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e696d6167653f465f7b747d3d5c7369676d612673706163653b28585f7b747d575f7b78667d2b485f7b742d317d575f7b68667d2b625f7b667d29)](https://camo.githubusercontent.com/edf8b8b9552b05ac6d234ebff09ed29f5839e6bb99ff8b01f612b82454df07b7/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e696d6167653f465f7b747d3d5c7369676d612673706163653b28585f7b747d575f7b78667d2b485f7b742d317d575f7b68667d2b625f7b667d29)

输出门：[![O_{t}=\sigma (X_{t}W_{xo}+H_{t-1}W_{ho}+b_{o})](https://camo.githubusercontent.com/a3dd8360210665449e6f9cea6c7a021e02b9e073b3be3304647ee3aaf52649b0/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e696d6167653f4f5f7b747d3d5c7369676d612673706163653b28585f7b747d575f7b786f7d2b485f7b742d317d575f7b686f7d2b625f7b6f7d29)](https://camo.githubusercontent.com/a3dd8360210665449e6f9cea6c7a021e02b9e073b3be3304647ee3aaf52649b0/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e696d6167653f4f5f7b747d3d5c7369676d612673706163653b28585f7b747d575f7b786f7d2b485f7b742d317d575f7b686f7d2b625f7b6f7d29)

这三个门的算式和普通RNN计算Ht算式相同。

[![image](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/57/57-1.png)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/57/57-1.png)

#### 候选记忆单元

[![\widetilde{C_{t}}=tanh(X_{t}W_{xc}+H_{t-1}W_{hc}+b_{c})](https://camo.githubusercontent.com/f3810cde80830d3ebbd78d3d2f1f446743b756a60d8c50ad08a536894f1e16dd/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e696d6167653f5c7769646574696c64657b435f7b747d7d3d74616e6828585f7b747d575f7b78637d2b485f7b742d317d575f7b68637d2b625f7b637d29)](https://camo.githubusercontent.com/f3810cde80830d3ebbd78d3d2f1f446743b756a60d8c50ad08a536894f1e16dd/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e696d6167653f5c7769646574696c64657b435f7b747d7d3d74616e6828585f7b747d575f7b78637d2b485f7b742d317d575f7b68637d2b625f7b637d29)

相当于在ht-1到ht的预测中又加了一层隐藏单元

[![image](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/57/57-2.png)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/57/57-2.png)

#### 记忆单元

[![C_{t}=F_{t}\odot C_{t-1}+I_{t}\odot \widetilde{C_{t}}](https://camo.githubusercontent.com/2adb443364832f22654c66bd08dabd2395344bb6a810e5478ec7f5a4bb32baf9/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e696d6167653f435f7b747d3d465f7b747d5c6f646f742673706163653b435f7b742d317d2b495f7b747d5c6f646f742673706163653b5c7769646574696c64657b435f7b747d7d)](https://camo.githubusercontent.com/2adb443364832f22654c66bd08dabd2395344bb6a810e5478ec7f5a4bb32baf9/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e696d6167653f435f7b747d3d465f7b747d5c6f646f742673706163653b435f7b742d317d2b495f7b747d5c6f646f742673706163653b5c7769646574696c64657b435f7b747d7d)

如果遗忘门始终为(1)且输入门始终为(0)， 则过去的记忆元 将随时间被保存并传递到当前时间步。 引入这种设计是为了缓解梯度消失问题， 并更好地捕获序列中的长距离依赖关系。

[![image](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/57/57-3.png)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/57/57-3.png)

#### 隐状态

[![H_{t}=O_{t}\odot tanh(C_{t})](https://camo.githubusercontent.com/baa39d3236644373134e79171faadd419e322913e8c6cf72dd1260c793dcb8ea/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e696d6167653f485f7b747d3d4f5f7b747d5c6f646f742673706163653b74616e6828435f7b747d29)](https://camo.githubusercontent.com/baa39d3236644373134e79171faadd419e322913e8c6cf72dd1260c793dcb8ea/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e696d6167653f485f7b747d3d4f5f7b747d5c6f646f742673706163653b74616e6828435f7b747d29)

最后，我们需要定义如何计算隐状态， 这就是输出门发挥作用的地方。 在长短期记忆网络中，它仅仅是记忆元的的门控版本。 这就确保了Ht的值始终在区间((-1, 1))内.

只要输出门接近1，我们就能够有效地将所有记忆信息传递给预测部分， 而对于输出门接近(0)，我们只保留记忆元内的所有信息，而不需要更新隐状态。

[![image](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/57/57-4.png)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/57/57-4.png)

#### 总结

LSTM的计算流程：

[![I_{t}=\sigma (X_{t}W_{xi}+H_{t-1}W_{hi}+b_{i})](https://camo.githubusercontent.com/5d35b6208b87b36a2fa57656db4d4e55d39d25dc8fc186a457e6c069822d04c3/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e696d6167653f495f7b747d3d5c7369676d612673706163653b28585f7b747d575f7b78697d2b485f7b742d317d575f7b68697d2b625f7b697d29)](https://camo.githubusercontent.com/5d35b6208b87b36a2fa57656db4d4e55d39d25dc8fc186a457e6c069822d04c3/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e696d6167653f495f7b747d3d5c7369676d612673706163653b28585f7b747d575f7b78697d2b485f7b742d317d575f7b68697d2b625f7b697d29)

[![F_{t}=\sigma (X_{t}W_{xf}+H_{t-1}W_{hf}+b_{f})](https://camo.githubusercontent.com/edf8b8b9552b05ac6d234ebff09ed29f5839e6bb99ff8b01f612b82454df07b7/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e696d6167653f465f7b747d3d5c7369676d612673706163653b28585f7b747d575f7b78667d2b485f7b742d317d575f7b68667d2b625f7b667d29)](https://camo.githubusercontent.com/edf8b8b9552b05ac6d234ebff09ed29f5839e6bb99ff8b01f612b82454df07b7/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e696d6167653f465f7b747d3d5c7369676d612673706163653b28585f7b747d575f7b78667d2b485f7b742d317d575f7b68667d2b625f7b667d29)

[![O_{t}=\sigma (X_{t}W_{xo}+H_{t-1}W_{ho}+b_{o})](https://camo.githubusercontent.com/a3dd8360210665449e6f9cea6c7a021e02b9e073b3be3304647ee3aaf52649b0/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e696d6167653f4f5f7b747d3d5c7369676d612673706163653b28585f7b747d575f7b786f7d2b485f7b742d317d575f7b686f7d2b625f7b6f7d29)](https://camo.githubusercontent.com/a3dd8360210665449e6f9cea6c7a021e02b9e073b3be3304647ee3aaf52649b0/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e696d6167653f4f5f7b747d3d5c7369676d612673706163653b28585f7b747d575f7b786f7d2b485f7b742d317d575f7b686f7d2b625f7b6f7d29)

[![\widetilde{C_{t}}=tanh(X_{t}W_{xc}+H_{t-1}W_{hc}+b_{c})](https://camo.githubusercontent.com/f3810cde80830d3ebbd78d3d2f1f446743b756a60d8c50ad08a536894f1e16dd/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e696d6167653f5c7769646574696c64657b435f7b747d7d3d74616e6828585f7b747d575f7b78637d2b485f7b742d317d575f7b68637d2b625f7b637d29)](https://camo.githubusercontent.com/f3810cde80830d3ebbd78d3d2f1f446743b756a60d8c50ad08a536894f1e16dd/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e696d6167653f5c7769646574696c64657b435f7b747d7d3d74616e6828585f7b747d575f7b78637d2b485f7b742d317d575f7b68637d2b625f7b637d29)

[![C_{t}=F_{t}\odot C_{t-1}+I_{t}\odot \widetilde{C_{t}}](https://camo.githubusercontent.com/2adb443364832f22654c66bd08dabd2395344bb6a810e5478ec7f5a4bb32baf9/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e696d6167653f435f7b747d3d465f7b747d5c6f646f742673706163653b435f7b742d317d2b495f7b747d5c6f646f742673706163653b5c7769646574696c64657b435f7b747d7d)](https://camo.githubusercontent.com/2adb443364832f22654c66bd08dabd2395344bb6a810e5478ec7f5a4bb32baf9/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e696d6167653f435f7b747d3d465f7b747d5c6f646f742673706163653b435f7b742d317d2b495f7b747d5c6f646f742673706163653b5c7769646574696c64657b435f7b747d7d)

[![H_{t}=O_{t}\odot tanh(C_{t})](https://camo.githubusercontent.com/baa39d3236644373134e79171faadd419e322913e8c6cf72dd1260c793dcb8ea/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e696d6167653f485f7b747d3d4f5f7b747d5c6f646f742673706163653b74616e6828435f7b747d29)](https://camo.githubusercontent.com/baa39d3236644373134e79171faadd419e322913e8c6cf72dd1260c793dcb8ea/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e696d6167653f485f7b747d3d4f5f7b747d5c6f646f742673706163653b74616e6828435f7b747d29)

### 从零实现

加载时光机器数据集

```
import torch
from torch import nn
from d2l import torch as d2l

batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
```



#### 初始化模型参数



```
def get_lstm_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device)*0.01

    def three():
        return (normal((num_inputs, num_hiddens)),
                normal((num_hiddens, num_hiddens)),
                torch.zeros(num_hiddens, device=device))

    W_xi, W_hi, b_i = three()  # 输入门参数
    W_xf, W_hf, b_f = three()  # 遗忘门参数
    W_xo, W_ho, b_o = three()  # 输出门参数
    W_xc, W_hc, b_c = three()  # 候选记忆元参数
    # 输出层参数
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)
    # 附加梯度
    params = [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc,
              b_c, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params
```

#### 定义模型

在初始化函数中， 长短期记忆网络的隐状态需要返回一个*额外*的记忆元， 单元的值为0，形状为（批量大小，隐藏单元数）。 因此，我们得到以下的状态初始化。

```
def init_lstm_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device),
            torch.zeros((batch_size, num_hiddens), device=device))
```



实际模型的定义与我们前面讨论的一样： 提供三个门和一个额外的记忆元。 请注意，只有隐状态才会传递到输出层， 而记忆元(\mathbf{C}_t)不直接参与输出计算。

```
def lstm(inputs, state, params):
    [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c,
     W_hq, b_q] = params
    (H, C) = state
    outputs = []
    for X in inputs:
        I = torch.sigmoid((X @ W_xi) + (H @ W_hi) + b_i)
        F = torch.sigmoid((X @ W_xf) + (H @ W_hf) + b_f)
        O = torch.sigmoid((X @ W_xo) + (H @ W_ho) + b_o)
        C_tilda = torch.tanh((X @ W_xc) + (H @ W_hc) + b_c)
        C = F * C + I * C_tilda
        H = O * torch.tanh(C)
        Y = (H @ W_hq) + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H, C)
```

#### 训练和预测



```
vocab_size, num_hiddens, device = len(vocab), 256, d2l.try_gpu()
num_epochs, lr = 500, 1
model = d2l.RNNModelScratch(len(vocab), num_hiddens, device, get_lstm_params,
                            init_lstm_state, lstm)
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
```

## 深层循环神经网络

### 深层循环神经网络



之前讲的RNN都只有一个隐藏层（序列变长不算是深度），而一个隐藏层的RNN一旦做的很宽就容易出现过拟合。因此我们考虑将网络做的更深而非更宽，每层都只做一点非线性，靠层数叠加得到更加非线性的模型。

浅RNN：输入-隐层-输出

深RNN：输入-隐层-隐层-...-输出

[![58-01](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/58/58-01.png)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/58/58-01.png)

（课程视频中的图片有错误，最后输出层后一时间步是不受前一步影响的，即没有箭头）

### 公式



[![img](https://camo.githubusercontent.com/9b91a12f1badf10aafaa7b2e12217ed91e3a40727e8f37ae90bf14cd9f440213/687474703a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e6c617465783f2535436d6174686266253742482537445f74253545313d665f31282535436d6174686266253742485f253742742d31253744253545312537442c2535436d6174686266253742585f7425374429)](https://camo.githubusercontent.com/9b91a12f1badf10aafaa7b2e12217ed91e3a40727e8f37ae90bf14cd9f440213/687474703a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e6c617465783f2535436d6174686266253742482537445f74253545313d665f31282535436d6174686266253742485f253742742d31253744253545312537442c2535436d6174686266253742585f7425374429)

*第一层的第t步状态是关于第一层第t-1步状态和第t步输入的函数*

[![img](https://camo.githubusercontent.com/9abfa1b6a822c9430f2f0d49337c239c2ec75ad481332296f63ad52f6300f8f8/687474703a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e6c617465783f2535436d6174686266253742482537445f742535456a3d665f6a282535436d6174686266253742485f253742742d312537442535456a2537442c2535436d6174686266253742485f253742742537442535452537426a2d3125374429253744)](https://camo.githubusercontent.com/9abfa1b6a822c9430f2f0d49337c239c2ec75ad481332296f63ad52f6300f8f8/687474703a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e6c617465783f2535436d6174686266253742482537445f742535456a3d665f6a282535436d6174686266253742485f253742742d312537442535456a2537442c2535436d6174686266253742485f253742742537442535452537426a2d3125374429253744)

*第j层的第t步状态是关于当前层上一步步状态和上一层当前步的函数*

[![img](https://camo.githubusercontent.com/589fc00237965860b1afdb8c28ca04cffd0e18718bd273ae110640115e73a01c/687474703a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e6c617465783f2535436d61746862662537424f2537445f743d67282535436d6174686266253742482537445f742535454c29)](https://camo.githubusercontent.com/589fc00237965860b1afdb8c28ca04cffd0e18718bd273ae110640115e73a01c/687474703a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e6c617465783f2535436d61746862662537424f2537445f743d67282535436d6174686266253742482537445f742535454c29)

*由最后一个隐藏层得到输出*

### 总结



- 深度循环神经网络使用多个隐藏层来获得更多的非线性性

将RNN/GRU/LSTM做深都是一个道理，三者只是使用的函数f不同。

## 编码器-解码器架构

### CNN中的解释



考虑一个CNN模型：

[![CNN](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/61/CNN.png)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/61/CNN.png)

整个CNN实际上可以看作一个编码器，解码器两部分。

- 底层的神经网络，也就是编码器将输入编码成能被模型识别的中间表达形式，也就是特征
- 解码器将中间结果解码为输出

### RNN中的解释

对于RNN而言，同样有着类似的划分

[![RNN](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/61/RNN.png)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/61/RNN.png)

- 编码器将输入文本表示为向量
- 解码器将向量表示为输出

### 抽象的编码器-解码器架构

指一个模型被分为两块：

- 一块是编码器，也叫encoder，用于将输入处理为一个中间状态
- 一块是解码器，也叫decoder，用于将中间状态表示为输出
- 解码器也可以有额外的输入提供信息

[![encoder-decoder](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/61/encoder-decoder.png)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/61/encoder-decoder.png)

### 代码

```
from torch import nn

class EncoderDecoder(nn.Mudule):
	def __init__(self,encoder,decoder,**kwargs):
	super(EncoderDecoder,self).__init__(**kwargs)
	self.encoder=encoder
	self.decoder=decoder
	
	def forward(self,enc_X,dec_X,*args):
	enc_outputs = self.encoder(enc_X,*args)
	dec_state=self.decoder(enc_outputs,*args)
	return self.decoder(dec_X,dec_state)
```

## 序列到序列学习

### 应用举例：机器翻译

- 给定一个源语言的句子，自动翻译成目标语言
- 这两个句子可以有不同的长度

### 模型架构：Seq2seq

[![img](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/62/62-01.png)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/62/62-01.png)

- 序列到序列模型由**编码器-解码器**构成。
- **编码器**RNN可以是**双向**，由于输入的句子是完整地，可以正着看，也可以反着看；而**解码器**只能是**单向**，由于预测时，只能正着去预测。
- 编码器，解码器采用**不同的RNN**，此RNN也可以是GRU，LSTM等。

### 编码器-解码器细节

[![img](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/62/62-02.png)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/62/62-02.png)

- 编码器，将长度不定的输入值放入RNN（或者其他模型），将其转化为固定形状的隐状态。

![image-20240727224619101](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20240727224619101.png)

- 解码器的特定设计决定：1. <bos>表示序列开始词元，<eos>表示模型停止预测。2.**编码器**的**最后时间步的隐状态**用作**解码器**的**初始隐状态**（图中箭头的传递）。编码器还加入了输出序列已经生成词元来进行下一部分的预测。3.在其他设计中，编码器最终的隐藏层在每一个时间步都可以作为编码器输入状态的一部分。

#### 实现循环神经网络编码器

使用了嵌入层（embedding layer）来获得输入序列中每个词元的特征向量。  嵌入层的权重是一个矩阵， 其行数等于输入词表的大小（`vocab_size`）， 其列数等于特征向量的维度（`embed_size`）。 对于任意输入词元的索引𝑖， 嵌入层获取权重矩阵的第𝑖行（从0开始）以返回其特征向量。 

```
class seq2seqEncoder(d2l.Encoder):
	def __init__(self, vocab_size, embed_size, num_hiddens, 					num_layers, dropout=0, **kwargs):
        super(Seq2SeqEncoder, self).__init__(**kwargs)
        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers,
                          dropout=dropout)
    def forward(self,X,*args):
    # 在循环神经网络模型中，第一个轴对应于时间步
   		X = self.embedding(X)
        X = X.permute(1,0,2)
        output,state = self.rnn(X)
        return output,state
    # output:（时间步数，批量大小，隐藏单元数）
```

#### 实现循环神经网络编码器

```
class Seq2SeqDecoder(d2l.Decoder):
    """用于序列到序列学习的循环神经网络解码器"""
	def __init__(self, vocab_size, embed_size, num_hiddens, 				num_layers, dropout=0, **kwargs):
        super(Seq2SeqDecoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size+num_hiddens, num_hiddens, 
        					num_layers, dropout=dropout)
        self.dense = nn.Linear(num_hiddens,vocab_size)
	def forward(self,X,*args):
   		X = self.embedding(X).permute(1,0,2)
   		# 广播context，使其具有与X相同的num_steps
   		context = state[-1].reshape(X.shape[0],1,1)
   		X_and_context = torch.cat((X,context),2)
   		output,state = self.rnn(X_and_context,state)
   		output = self.dense(output).permute(1,0,2)
   	return output,state
```

#### 损失函数

`sec_machine_translation`中， 特定的填充词元被添加到序列的末尾， 因此不同长度的序列可以以相同形状的小批量加载。 但是，我们应该将填充词元的预测排除在损失函数的计算之外。

使用下面的`sequence_mask`函数 [**通过零值化屏蔽不相关的项**]， 以便后面任何不相关预测的计算都是与零的乘积，结果都等于零。 

```python
def sequence_mask(X,valid_len,value=0):
	maxlen=X[1]
	mask = torch.arange(maxlen,dtype=torch.float32)[None,:]<valid_len[:,None]
	X[~mask] = value
	
class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    """带遮蔽的softmax交叉熵损失函数"""
    # pred的形状：(batch_size,num_steps,vocab_size)
    # label的形状：(batch_size,num_steps)
    # valid_len的形状：(batch_size,)
    def forward(self, pred, label, valid_len):
        weights = torch.one_like(label)
        weights = sequence(weight,valid_len)
        self.reduction='none'
        # pytorch 需要把预测/输出的维度放到中间
        unweighted_loss = super().forward(pred.permute(0, 2, 1), label)
        weighted_loss = (unweighted_loss * weights).mean(dim=1)
        return weighted_loss
```

### 训练和推理

[![img](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/62/62-03.png)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/62/62-03.png)

- 损失函数的计算只关注解码器的输出层。
- 训练和预测（推理）有区别的，训练时解码器使用目标句子（真值）作为输入，以指导模型训练；而推理时无法提前得知真值，需要一步一步进行预测。

```
def train_seq2seq(net, data_iter, lr, num_epochs, tgt_vocab, device):
    """训练序列到序列模型"""
    # 初始化权重
    def xavier_init_weights(m):
    # 针对全连接层
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
    # 针对RNN
        if type(m) == nn.GRU:
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])

    net.apply(xavier_init_weights)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    # 引入掩码交叉熵
    loss = MaskedSoftmaxCELoss()
    net.train()
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                     xlim=[10, num_epochs])
    for epoch in range(num_epochs):
        timer = d2l.Timer()
        metric = d2l.Accumulator(2)  # 训练损失总和，词元数量
        for batch in data_iter:
            optimizer.zero_grad()
            
            # 输入序列 X，输入序列的有效长度 X_valid_len，目标序列 Y，和目标序列的有效长度 Y_valid_len
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
            # Y.shape[0]是代表有多少个输出语句，则对应我们有多少个bos开始，最后做成一列
            bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0],
                          device=device).reshape(-1, 1)
            # 将开始符号bos和我们的Y目标序列，去除掉最后一个我们需要预测的部分
            dec_input = torch.cat([bos, Y[:, :-1]], 1)  # 强制教学
            # 生成预测值
            Y_hat, _ = net(X, dec_input, X_valid_len)
            # Y_valid_len 非必要部分不算loss
            l = loss(Y_hat, Y, Y_valid_len)
            
            l.sum().backward()	# 损失函数的标量进行“反向传播”
            d2l.grad_clipping(net, 1)
            num_tokens = Y_valid_len.sum()
            optimizer.step()
            with torch.no_grad():
                metric.add(l.sum(), num_tokens)
        if (epoch + 1) % 10 == 0:
            animator.add(epoch + 1, (metric[0] / metric[1],))
    print(f'loss {metric[0] / metric[1]:.3f}, {metric[1] / timer.stop():.1f} '
        f'tokens/sec on {str(device)}')
```

#### 预测

```
def predict_seq2seq(net, src_sentence, src_vocab, tgt_vocab, num_steps,
                    device, save_attention_weights=False):
    """序列到序列模型的预测"""
    # 在预测时将net设置为评估模式
    net.eval()
    src_tokens = src_vocab[src_sentence.lower().split(' ')] + [ src_vocab['<eos>']]
    enc_valid_len = torch.tensor([len(src_tokens)], device=device)
    
    ’‘’
    这句话的作用是将输入的源序列 src_tokens 截断或填充到固定长度 num_steps。
	如果 src_tokens 的长度小于 num_steps，则使用 src_vocab['<pad>']（填充标记）填充至 num_steps。
	如果长度超过 num_steps，则将序列截断到 num_steps 的长度。
	‘’‘
    src_tokens = d2l.truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])
    
    # 添加批量轴
    # torch.unsqueeze 用于在指定的维度上增加一个维度。
    enc_X = torch.unsqueeze(
        torch.tensor(src_tokens, dtype=torch.long, device=device), dim=0)
        
    enc_outputs = net.encoder(enc_X, enc_valid_len)
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)
    # 添加批量轴
    dec_X = torch.unsqueeze(torch.tensor(
        [tgt_vocab['<bos>']], dtype=torch.long, device=device), dim=0)
    output_seq, attention_weight_seq = [], []
    for _ in range(num_steps):
        Y, dec_state = net.decoder(dec_X, dec_state)
        # 我们使用具有预测最高可能性的词元，作为解码器在下一时间步的输入
        dec_X = Y.argmax(dim=2)
        # 减少维度，获取数据并从张量转变成标量
        pred = dec_X.squeeze(dim=0).type(torch.int32).item()
        # 保存注意力权重（稍后讨论）
        if save_attention_weights:
            attention_weight_seq.append(net.decoder.attention_weights)
        # 一旦序列结束词元被预测，输出序列的生成就完成了
        if pred == tgt_vocab['<eos>']:
            break
        output_seq.append(pred)
    return ' '.join(tgt_vocab.to_tokens(output_seq)), attention_weight_seq
```

### 衡量生成序列的好坏：BLEU

#### BLUE值定义：

[![img](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/62/62-04.png)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/62/62-04.png)

```
def bleu(pred_seq, label_seq, k): 
    """计算BLEU"""
    pred_tokens, label_tokens = pred_seq.split(' '), label_seq.split(' ')
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, k + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        # 计算匹配值
        for i in range(len_label - n + 1):
            label_subs[' '.join(label_tokens[i: i + n])] += 1
        # 计算罚值
        for i in range(len_pred - n + 1):
            if label_subs[' '.join(pred_tokens[i: i + n])] > 0:
                num_matches += 1
                label_subs[' '.join(pred_tokens[i: i + n])] -= 1
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score
```

#### 定义式解析

[![img](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/62/62-06.png)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/62/62-06.png)

BLEU值衡量的是精确率，而且对不同n-gram进行集成打分。

- BP惩罚因子：为了惩罚过短的句子，由于过短的句子基数小，精确率容易提升，所以加上一个BP乘子，当预测句子长度<参考句子长度，则BP<1。
- wn的选择：李沐老师课程中是采用了$\frac{1}{2^n}$作为加权因子，n越大，加权因子越小，但由于pn<1，赋予的权重越大，即长匹配具有更高的权重。





## 束搜索

在序列生成问题中，常用的方法是一个个词元地进行生成，但是先前步生成的词元会影响之后词元的概率分布，为此，我们需要使用搜索算法来得到一个较好的序列

### 贪心搜索

贪心搜索即每个时间步都选择具有最高条件概率的词元。 $$ y_{t'} = \operatorname*{argmax}*{y \in \mathcal{Y}} P(y \mid y_1, \ldots, y*{t'-1}, \mathbf{c}) $$ 我们的目标是找到一个最有序列，他的联合概率，也就是每步之间的条件概率的乘积，最大。 $$ \prod_{t'=1}^{T'} P(y_{t'} \mid y_1, \ldots, y_{t'-1}, \mathbf{c}) $$ 然而，贪心搜索很可能搜索到的不是最优解，例如：[![img](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/63/Greedy_or_not.png)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/63/Greedy_or_not.png)

左侧的搜索方式为贪心搜索，每次找到当前条件概率最大的选项进行预测，但是这样可能会导致之后的条件概率较小，从而导致最终的联合概率较小，生成的序列不优。

而右侧的选择方式虽然在第二步选择了较小的选项，但之后在第三步时有了条件概率为0.6选项，最终结果反而更好。

### 穷举搜索

穷举搜索枚举所有可能的输出序列及其概率，然后选择概率最大的作为最终的输出，枚举搜索可以保证得到最优解，但是计算复杂度很高，难以实现

### 束搜索（beam search）

束搜索综合了贪心搜索和穷举搜索，在能接受的计算成本下得到比贪心搜索更好的结果。

束搜索有一个超参数，名为**束宽（beam size）**𝑘，束搜索的具体流程如下：

- 1：在第一时间步选择条件概率最高的k个选项
- 2：对随后的每个时间步，基于上一时间步的k个候选输出序列预测这一时间步的所有可能选项的条件概率，从中取k个最大的
- 3：最后基于每步得到的序列，删去截止符和其后元素，获得最终候选序列集合，取出加权条件概率最大的

加权条件概率公式如下： $$ \frac{1}{L^\alpha} \log P(y_1, \ldots, y_{L}\mid \mathbf{c}) = \frac{1}{L^\alpha} \sum_{t'=1}^L \log P(y_{t'} \mid y_1, \ldots, y_{t'-1}, \mathbf{c}), $$ 式中$\frac{1}{L^\alpha}$用于调整长序列的评估值使得长短序列间的比较公平

束宽k的选择：

- k=1时实际为贪心搜索
- k越小搜索速度越快，但结果越差，k越大则搜索速度越慢，但结果越好

束搜索只在测试时使用





# 注意力机制

## 注意力机制

## 注意力分数

### 理论内容

### 代码与具体操作

在上一节中，我们使用**高斯核**来对查询和键之间的关系建模。我们可以将上一节中的高斯核函数部分视为注意力评分函数，简称评分函数，然后把这个函数的输出结果输入到softmax函数中进行运算。 通过上述步骤，我们将得到与键对应的值的概率分布（即注意力权重）。 最后，注意力汇聚的输出就是基于这些注意力权重的值的加权和。

```
import math
import torch
from torch import nn
from d2l import torch as d2l
```

#### 掩蔽softmax操作

正如上面提到的，**softmax操作用于输出一个概率分布作为注意力权重**。 在某些情况下，并非所有的值都应该被纳入到注意力汇聚中。 例如，为了在 [9.5节](https://zh-v2.d2l.ai/chapter_recurrent-modern/machine-translation-and-dataset.html#sec-machine-translation)中高效处理小批量数据集， 某些文本序列被填充了没有意义的特殊词元。 为了仅将有意义的词元作为值来获取注意力汇聚， 我们可以指定一个**有效序列长度（即词元的个数）， 以便在计算softmax时过滤掉超出指定范围的位置。** 通过这种方式，我们可以在下面的`masked_softmax`函数中 实现这样的***掩蔽softmax操作*（masked softmax operation）**， 其中任何超出有效长度的位置都被掩蔽并置为0。

```python
def masked_softmax(X,valid_lens):
	  if valid_lens == None:
	  	return nn.functional.softmax(X,dim=-1)
	  	
	  else:
      	shape = X.shape
        
	  	if valid_lens.dim() == 1:
            #将valid_lens的每一个元素重复shape[1]次
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
	  	else:
	  		valid_lens = valid_lens.reshape(-1)
	  	X = d2l.sequence_mask(X.reshape(-1,X.shape[-1]),valid_lens,value=1e-6)
        
	  	return nn.functional.softmax(X,dim=-1)
```

#### 加性注意力

- 一般来说，当查询和键是不同长度的矢量时， 我们可以使用加性注意力作为评分函数。
- 我们有三个值，q,k,v
- 将查询和键连结起来后输入到一个多层感知机（MLP）中，感知机包含一个隐藏层，其隐藏单元数是一个超参数ℎ。 通过使用tanh作为激活函数，并且禁用偏置项。
- ![image-20240728141052516](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20240728141052516.png)

```
class AdditiveAttention(nn.module):
	def __init__(self,key_size,query_size,num_hiddens,dropout,**kwargs):
	super( AdditiveAttention,self).__init__(**kwargs)
	self.W_k = nn.Linear(key_size,num_hiddens,bias=false)
	self.W_q = nn.Linear(query_size,num_hiddens,bias=false)
	self.W_v = nn.Linear(num_hiddens,1,bias=false)
	self.dropout = nn.Dropout(dropout)


#valid_lens:有多少key-value pair是有人需要的
	def forward(self, queries, keys, values, valid_lens):
		queries,keys=self.W_q(queries),self.W_k(keys)
		features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)
        # self.w_v仅有一个输出，因此从形状中移除最后那个维度。
         # scores的形状：(batch_size，查询的个数，“键-值”对的个数)
        scores = self.w_v(features).squeeze(-1)
        self.attention_weights = masked_softmax(scores,valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)
```

#### 缩放点积注意力

使用点积可以得到计算效率更高的评分函数， 但是点积操作要求查询和键具有相同的长度*d*。 假设查询和键的所有元素都是独立的随机变量， 并且都满足零均值和单位方差， 那么两个向量的点积的均值为0，方差为*d*。 为确保无论向量长度如何， 点积的方差在不考虑向量长度的情况下仍然是1， 我们将点积除以√d,在下面的缩放点积注意力的实现中，我们使用了暂退法进行模型正则化。

```
#@save
class DotProductAttention(nn.Module):
    """缩放点积注意力"""
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    # queries的形状：(batch_size，查询的个数，d)
    # keys的形状：(batch_size，“键－值”对的个数，d)
    # values的形状：(batch_size，“键－值”对的个数，值的维度)
    # valid_lens的形状:(batch_size，)或者(batch_size，查询的个数)
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        # 设置transpose_b=True为了交换keys的最后两个维度
        scores = torch.bmm(queries, keys.transpose(1,2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)
```

#### 小结

- 将注意力汇聚的输出计算可以作为值的加权平均，选择不同的注意力评分函数会带来不同的注意力汇聚操作。
- 当查询和键是不同长度的矢量时，可以使用可加性注意力评分函数。当它们的长度相同时，使用缩放的“点－积”注意力评分函数的计算效率更高。

## 使用注意力机制的seq2seq

改变上下文变量：在预测词元时，如果不是所有输入词元都相关，模型将仅对齐（或参与）输入序列中与当前预测相关的部分。这是通过将上下文变量视为注意力集中的输出来实现的。

### Bahdanau注意力模型

假设输入序列中有𝑇个词元， 解码时间步𝑡′的上下文变量是注意力集中的输出：

![image-20240728163531954](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20240728163531954.png)

![image-20240728163604947](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20240728163604947.png)



### 自定义注意力解码器注意力

#### **带有注意力机制解码器的基本接口**

```
class AttentionDecoder(d2l.Decoder):
    """带有注意力机制解码器的基本接口"""
    def __init__(self, **kwargs):
        super(AttentionDecoder, self).__init__(**kwargs)

    @property
    def attention_weights(self):
        raise NotImplementedError
```



#### **实现带有Bahdanau注意力的循环神经网络解码器**

- 初始化解码器的状态
- 将每一个时间步最终隐藏层的状态作为注意力的key/value
- 上一时间步隐藏层的全状态将初始解码器的状态
- 解码器上一时间步的最终隐藏层将作为query

```
class Seq2SeqAttentionDecoder(AttentionDecoder):
	def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqAttentionDecoder, self).__init__(**kwargs)
        self.attention = d2l.AdditiveAttention(
            num_hiddens, num_hiddens, num_hiddens, dropout)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(
            embed_size + num_hiddens, num_hiddens, num_layers,
            dropout=dropout)
        self.dense = nn.Linear(num_hiddens, vocab_size)
        
    def init_state(self,enc_X,enc_valid_len,*args):
    	# outputs = (batch_size，num_steps，num_hiddens)
    	# hidden_state的形状为(num_layers，batch_size，num_hiddens)
    	outputs, hidden_state = enc_outputs
    	return (outputs.permute(1,0,2),hidden_state,enc_valid_lens)
    
    def forward(self, X, state):
    	# enc_outputs的形状为(batch_size,num_steps,num_hiddens).
        # hidden_state的形状为(num_layers,batch_size,
        # num_hiddens)
        enc_outputs, hidden_state, enc_valid_lens = state
    	# 输出X的形状为(num_steps,batch_size,embed_size)
    	X = self.embedding(X).permute(1,0,2)
    	outputs, self._attention_weights = [], []
    	
    	for x in X:
    		# query的形状为(batch_size,1,num_hiddens)
    		query = torch.unsqueeze(hidden_state[-1],dim=1)
    		# context = (batch_size,1,num_hiddens)
    		context = self.attention(query,enc_outputs,enc_outputs,enc_valid_lens)
    		# 在特征维度上连接
    		x = torch.cat(context,torch.unsqueeze(x,dim=1),dim=-1)
    		 # 将x变形为(1,batch_size,embed_size+num_hiddens)
            out, hidden_state = self.rnn(x.permute(1, 0, 2), hidden_state)
            outputs.append(out)
            self._attention_weights.append(self.attention.attention_weights)
    # 全连接层变换后，outputs的形状为
        # (num_steps,batch_size,vocab_size)
        outputs = self.dense(torch.cat(outputs, dim=0))
        return outputs.permute(1, 0, 2), [enc_outputs, hidden_state,
                                          enc_valid_lens]
    
    @property
    def attention_weights(self):
        return self._attention_weights
```

## 多头注意力

允许注意力机制组合使用查询、键和值的不同 *子空间表示*（representation subspaces）：模型基于相同的注意力去学习不同的行为和知识，最后将这些不同的知识组合在一起。

多头注意力：h组全连接层来分别变化k,q,v（独立学习不同的行为），最后将q,k,v并行送到注意力汇聚中。最终将h个注意力汇聚的输出拼接到一起，通过一个线性投影进行变换。

![image-20240728190956824](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20240728190956824.png)

### 数学表达

在实现多头注意力之前，让我们用数学语言将这个模型形式化地描述出来。给定查询$\mathbf{q} \in \mathbb{R}^{d_q}$、键$\mathbf{k} \in \mathbb{R}^{d_k}$和值$\mathbf{v} \in \mathbb{R}^{d_v}$，每个注意力头$\mathbf{h}_i$（$i = 1, \ldots, h$）的计算方法为：

$$\mathbf{h}_i = f(\mathbf W_i^{(q)}\mathbf q, \mathbf W_i^{(k)}\mathbf k,\mathbf W_i^{(v)}\mathbf v) \in \mathbb R^{p_v},$$

其中，可学习的参数包括
$\mathbf W_i^{(q)}\in\mathbb R^{p_q\times d_q}$、$\mathbf W_i^{(k)}\in\mathbb R^{p_k\times d_k}$和$\mathbf W_i^{(v)}\in\mathbb R^{p_v\times d_v}$，以及代表注意力汇聚的函数$f$。$f$可以是 :numref:`sec_attention-scoring-functions`中的**加性注意力和缩放点积注意力**。多头注意力的输出需要经过另一个线性转换，它对应着$h$个头连结后的结果，因此其可学习参数是
$\mathbf W_o\in\mathbb R^{p_o\times h p_v}$：

$$\mathbf W_o \begin{bmatrix}\mathbf h_1\\\vdots\\\mathbf h_h\end{bmatrix} \in \mathbb{R}^{p_o}.$$

基于这种设计，每个头都可能会关注输入的不同部分，可以表示比简单加权平均值更复杂的函数。

### 实现

```
class MultiHeadAttention:
	def __init__(self,key_size,query_size,vocab_size,num_hiddens,num_heads,dropout,bias=False,**kwargs):
	self.num_heads = num_heads
	self.attention = d2l.DotProductAttention(dropout)
	self.W_q = nn.Linear(query_size,num_hiddens,bias=bias)
	self.W_v = nn.Linear(vocab_size,num_hiddens,bias=bias)
	self.W_k = nn.Linear(key_size,num_hiddens,bias=bias)
	self.W_o = nn.Linear(num_hiddens,num_hiddens,bias=bias)
	
	def forward(self, queries, keys, values, valid_lens):
		# queries，keys，values的形状:
        # (batch_size，查询或者“键－值”对的个数，num_hiddens)
        # valid_lens　的形状:
        # (batch_size，)或(batch_size，查询的个数)
        # 经过变换后，输出的queries，keys，values　的形状:
        # (batch_size*num_heads，查询或者“键－值”对的个数，num_hiddens/num_heads)
        # 最大的不同就是把一组键值放在多个注意力内计算
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)
        
        if valid_lens is not None:
        	# 在轴0，将第一项（标量或者矢量）复制num_heads次，如此重复第二项
        	valid_lens = torch.repeat_interleave(valid_lens,repeat=num_heads,dim=0)
        	
        # output的形状:(batch_size*num_heads，查询的个数，num_hiddens/num_heads)	
        output = self.attention(queries, keys, values, valid_lens)
        
        # output_concat的形状:(batch_size，查询的个数，num_hiddens)
        output_concat = transpose_output(output,self.num_heads)
        
        return self.W_o(output_concat)
```

```
def transpose_qkv(X, num_heads):
    """为了多注意力头的并行计算而变换形状"""
    # 输入X的形状:(batch_size，查询或者“键－值”对的个数，num_hiddens)
    # 输出X的形状:(batch_size，查询或者“键－值”对的个数，num_heads，
    # num_hiddens/num_heads)
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)

    # 输出X的形状:(batch_size，num_heads，查询或者“键－值”对的个数,
    # num_hiddens/num_heads)
    X = X.permute(0, 2, 1, 3)

    # 最终输出的形状:(batch_size*num_heads,查询或者“键－值”对的个数,
    # num_hiddens/num_heads)
    return X.reshape(-1, X.shape[2], X.shape[3])


#@save
def transpose_output(X, num_heads):
    """逆转transpose_qkv函数的操作"""
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)
```

## 自注意力和位置编码

每个查询都会关注所有的键－值对并生成一个注意力输出。 由于查询、键和值来自同一组输入，因此被称为 *自注意力*（self-attention）

### 自注意力

![image-20240728190211948](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20240728190211948.png)

```
num_hiddens, num_heads = 100, 5
attention = d2l.MultiHeadAttention(num_hiddens, num_hiddens, num_hiddens, num_hiddens, num_heads, 0.5)
attention.eval()
```

### 比较卷积神经网络、循环神经网络和自注意力

![image-20240728192821220](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20240728192821220.png)

![image-20240728192844134](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20240728192844134.png)

### 位置编码

- 子注意力并没有记录位置信息

- 位置编码将位置信息注入到输入当中

- 假设有$\mathbf{n} \in \mathbb{R}^{n*d}$,那么使用位置编码信息$\mathbf{P} \in \mathbb{R}^{n*d}$，来输出X+P作为自编码输入

- ![image-20240728194354551](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20240728194354551.png)

  （基于正弦函数和余弦函数的固定位置编码）

```python
class PositionalEncodeing(nn.Module):
	def __init__(delf,num_hiddens,dropout,max_len=1000):
	super(PositionalEncodeing,self).__init__()
	self.dropout = nn.Dropout(dropout)
	self.P = torch.zeros((1,max_len,num_hiddens))
    X = torch.arange(max_len,dtype=torch.float32).reshape(-1,1)/(torch.pow(10000,torch.arange(0,num_hiddens, 2, dtype=torch.float32)/num_hiddens))
    self.P[:,:,0::2] = torch.sin(X)
    self.P[:,:,1::2] = torch.cos(X)
    
     def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)
    
    
    
    
X = pos_encoding(torch.zeros((1, num_steps, encoding_dim)))encoding_dim, num_steps = 32, 60
pos_encoding = PositionalEncoding(encoding_dim, 0)
pos_encoding.eval()
X = pos_encoding(torch.zeros((1, num_steps, encoding_dim)))
P = pos_encoding.P[:, :X.shape[1], :]
d2l.plot(torch.arange(num_steps), P[0, :, 6:10].T, xlabel='Row (position)',
         figsize=(6, 2.5), legend=["Col %d" % d for d in torch.arange(6, 10)])
```

#### 绝对位置信息

在二进制表示中，**较高比特位的交替频率低于较低比特位**， 与下面的热图所示相似，只是位置编码通过使用三角函数[**在编码维度上降低频率**]。 由于输出是浮点数，因此此类连续表示比二进制表示法更节省空间。

#### 相对位置信息

除了捕获绝对位置信息之外，上述的位置编码还允许模型学习得到输入序列中相对位置信息。 这是因为对于任何确定的位置偏移𝛿，位置𝑖+𝛿处 的位置编码可以线性投影位置𝑖处的位置编码来表示。(平移不变性)

这种投影的数学解释是，令𝜔𝑗=1/100002𝑗/𝑑， 对于任何确定的位置偏移𝛿， :eqref:`eq_positional-encoding-def`中的任何一对 (𝑝𝑖,2𝑗,𝑝𝑖,2𝑗+1)都可以线性投影到 (𝑝𝑖+𝛿,2𝑗,𝑝𝑖+𝛿,2𝑗+1)：

![image-20240728210621574](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20240728210621574.png)

2×2投影矩阵不依赖于任何位置的索引𝑖。

## transformer架构

- 基于encoder-decoder架构来处理序列对
- 跟使用注意力的seq2seq不同，transformer是纯基于注意力

[![image](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/68/68-01.png)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/68/68-01.png)

### 多头注意力

- 对同一key，value，query，希望抽取不同的信息

  - 例如短距离关系和长距离关系

- 多头注意力使用h个独立的注意力池化

  - 合并各个头（head）输出得到最终输出
  - [![image](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/68/68-02.png)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/68/68-02.png)

- 数学表达式

  [![image](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/68/68-02.png)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/68/68-02.png)

### 有掩码的多头注意力

- 解码器对序列中一个元素输出的时候，不应该考虑该元素之后的元素
- 可以用掩码来实现
  - 也就是计算$x_i$输出的时候，假装当前序列长度为i

### 基于位置的前馈网络

- 将输入形状变化（b,n,d）变换成（bn，d）；输出形状由（bn，d）变成（b，n，d）
- 作用两个全连接层
- 等价于两层核窗口为1的一维卷积层（全连接）



### 层归一化

- 批量归一化对每个特征/通道里元素进行归一化
  - 不适合序列长度会变的nlp应用
- 层归一化对每个样本里面的元素进行归一化（ layer norm ）

### 信息传递

- 将编码器输出作为解码中第i个transformer块中多头注意力的key和value
  - query来自目标序列
- 意味着编码器和解码器中块的个数，输出维度都是一样的

### 预测

- 预测第t+1个输出时
- 解码器中输入前t个预测值（顺序）
  - 在自注意力中，前t个预测值作为key和value，第t个预测值还作为query

### 总结

- transformer是一个纯使用注意力的encoder-decoder
- 编码器和解码器都有n个transformer块
- 每个块里面使用多头注意力，基于位置的前馈网络，层归一化

### QA

- 多头注意力，concat和相加取平均怎么选择？
  - 老师认为concat保留的信息更全面，更好
- 为什么在获取词向量之后，需要对词向量进行缩放（乘以embedding size的开方之后再加上PE）
  - embedding之后，向量长度变长，元素值变小，乘以之后可以保证在-1，1之间，和position大小差不多
- num of head是什么？
  - 类似卷积的多通道，多个attention关注的是不同的特征



# 自然语言处理

## bert

#### NLP里的迁移学习

- 使用预训练好的模型来抽取词，句子的特征
  - 例如word2vec或语言模型
- 不更新预训练好的模型
- 需要构建新的网络来抓取任务需要的信息
  - Word2vec忽略了时序信息
  - 语言模型只看了一个方向

#### BERT的动机

- 基于微调的NLP模型
- 预训练的模型抽取了足够多的信息
- 新的任务只需要增加一个简单地输出层

[![image](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/69/69-1.png)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/69/69-1.png)

#### BERT架构

- 只有编码器的Transformer
- 两个版本：
  - Base:#blocks=12,hidden size=768,#heads=12,#parameters=110M
  - Large:#blocks=24,hidden size=1024,#heads=16,#paramerter=340M
- 在大规模数据上训练>3B词

#### 对输入的修改

- 每个样本是一个句子对
- 加入额外的片段嵌入
- 位置编码可学习

[![image](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/69/69-2.png)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/69/69-2.png)

#### 预训练任务

#####  带掩码的语言模型

- Transformer的编码器是双向的，标准语言模型要求单向
- 带掩码的语言模型每次随机（15%概率）将一些词元换成
- [![image](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/57/57-5.png)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/57/57-5.png)

##### 下一个句子预测

- 预测一个句子对中两个句子是不是相邻
- 训练样本中：
  - 50%概率选择相邻句子对：this movie is great i like it
  - 50%概率选择随机句子对：this movie is great hello world
- 将对应的输出放到一个全连接层来预测

#### 总结

- BERT针对微调设计
- 基于Transformer的编码器做了如下修改
  - 模型更大，训练数据更多
  - 输入句子对，片段嵌入，可学习的位置编码
  - 训练时使用两个任务：
    - 带掩码的语言模型
    - 下一个句子预测

### 代码实现

#### 获取输入：

在自然语言处理中，有些任务（如情感分析）以单个文本作为输入，而有些任务（如自然语言推断）以一对文本序列作为输入。BERT输入序列明确地表示单个文本和文本对。当输入为单个文本时，BERT输入序列是特殊类别词元“”、文本序列的标记、以及特殊分隔词元“”的连结。当输入为文本对时，BERT输入序列是“”、第一个文本序列的标记、“”、第二个文本序列标记、以及“”的连结。我们将始终如一地将术语“BERT输入序列”与其他类型的“序列”区分开来。例如，一个*BERT输入序列*可以包括一个*文本序列*或两个*文本序列*。

为了区分文本对，根据输入序列学到的片段嵌入eA和eB分别被添加到第一序列和第二序列的词元嵌入中。对于单文本输入，仅使用eA。

下面的`get_tokens_and_segments`将一个句子或两个句子作为输入，然后返回BERT输入序列的标记及其相应的片段索引。

```
 def get_tokens_and_segments(tokens_a, tokens_b=None):
    """获取输入序列的词元及其片段索引"""
    tokens = ['<cls>'] + tokens_a + ['<sep>']
    # 0和1分别标记片段A和B
    segments = [0] * (len(tokens_a) + 2)
    if tokens_b is not None:
        tokens += tokens_b + ['<sep>']
        segments += [1] * (len(tokens_b) + 1)
    return tokens, segments
```

BERT选择Transformer编码器作为其双向架构。在Transformer编码器中常见是，位置嵌入被加入到输入序列的每个位置。然而，与原始的Transformer编码器不同，BERT使用*可学习的*位置嵌入。总之， 下图表明BERT输入序列的嵌入是词元嵌入、片段嵌入和位置嵌入的和。

[![image](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/raw/main/imgs/69/69-3.png)](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes/blob/main/imgs/69/69-3.png)

#### BERT实现

下面的`BERTEncoder`类类似于 [10.7节](https://zh-v2.d2l.ai/chapter_attention-mechanisms/transformer.html#sec-transformer)中实现的`TransformerEncoder`类。与`TransformerEncoder`不同，`BERTEncoder`使用片段嵌入和可学习的位置嵌入。

```
#@save
class BERTEncoder(nn.Module):
    """BERT编码器"""
    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, num_layers, dropout,
                 max_len=1000, key_size=768, query_size=768, value_size=768,
                 **kwargs):
        super(BERTEncoder, self).__init__(**kwargs)
        self.token_embedding = nn.Embedding(vocab_size, num_hiddens)
        self.segment_embedding = nn.Embedding(2, num_hiddens)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(f"{i}", d2l.EncoderBlock(
                key_size, query_size, value_size, num_hiddens, norm_shape,
                ffn_num_input, ffn_num_hiddens, num_heads, dropout, True))
        # 在BERT中，位置嵌入是可学习的，因此我们创建一个足够长的位置嵌入参数
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len,
                                                      num_hiddens))

    def forward(self, tokens, segments, valid_lens):
        # 在以下代码段中，X的形状保持不变：（批量大小，最大序列长度，num_hiddens）
        X = self.token_embedding(tokens) + self.segment_embedding(segments)
        X = X + self.pos_embedding.data[:, :X.shape[1], :]
        for blk in self.blks:
            X = blk(X, valid_lens)
        return X
```



假设词表大小为10000，为了演示`BERTEncoder`的前向推断，让我们创建一个实例并初始化它的参数。

```
vocab_size, num_hiddens, ffn_num_hiddens, num_heads = 10000, 768, 1024, 4
norm_shape, ffn_num_input, num_layers, dropout = [768], 768, 2, 0.2
encoder = BERTEncoder(vocab_size, num_hiddens, norm_shape, ffn_num_input,
                      ffn_num_hiddens, num_heads, num_layers, dropout)
```



我们将`tokens`定义为长度为8的2个输入序列，其中每个词元是词表的索引。使用输入`tokens`的`BERTEncoder`的前向推断返回编码结果，其中每个词元由向量表示，其长度由超参数`num_hiddens`定义。此超参数通常称为Transformer编码器的*隐藏大小*（隐藏单元数）

```
tokens = torch.randint(0, vocab_size, (2, 8))
segments = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1, 1, 1]])
encoded_X = encoder(tokens, segments, None)
encoded_X.shape
```

#### 预训练任务

##### 遮掩语言模型

我们实现了下面的`MaskLM`类来预测BERT预训练的掩蔽语言模型任务中的掩蔽标记。预测使用单隐藏层的多层感知机（`self.mlp`）。在前向推断中，它需要两个输入：`BERTEncoder`的编码结果和用于预测的词元位置。输出是这些位置的预测结果。

```
#@save
class MaskLM(nn.Module):
    """BERT的掩蔽语言模型任务"""
    def __init__(self, vocab_size, num_hiddens, num_inputs=768, **kwargs):
        super(MaskLM, self).__init__(**kwargs)
        self.mlp = nn.Sequential(nn.Linear(num_inputs, num_hiddens),
                                 nn.ReLU(),
                                 nn.LayerNorm(num_hiddens),
                                 nn.Linear(num_hiddens, vocab_size))

    def forward(self, X, pred_positions):
        num_pred_positions = pred_positions.shape[1]
        pred_positions = pred_positions.reshape(-1)
        batch_size = X.shape[0]
        batch_idx = torch.arange(0, batch_size)
        # 假设batch_size=2，num_pred_positions=3
        # 那么batch_idx是np.array（[0,0,0,1,1]）
        batch_idx = torch.repeat_interleave(batch_idx, num_pred_positions)
        masked_X = X[batch_idx, pred_positions]
        masked_X = masked_X.reshape((batch_size, num_pred_positions, -1))
        mlm_Y_hat = self.mlp(masked_X)
        return mlm_Y_hat
```



为了演示`MaskLM`的前向推断，我们创建了其实例`mlm`并对其进行了初始化。回想一下，来自`BERTEncoder`的正向推断`encoded_X`表示2个BERT输入序列。我们将`mlm_positions`定义为在`encoded_X`的任一输入序列中预测的3个指示。`mlm`的前向推断返回`encoded_X`的所有掩蔽位置`mlm_positions`处的预测结果`mlm_Y_hat`。对于每个预测，结果的大小等于词表的大小。

```
mlm = MaskLM(vocab_size, num_hiddens)
mlm_positions = torch.tensor([[1, 5, 2], [6, 1, 5]])
mlm_Y_hat = mlm(encoded_X, mlm_positions)
mlm_Y_hat.shape
```



通过掩码下的预测词元`mlm_Y`的真实标签`mlm_Y_hat`，我们可以计算在BERT预训练中的遮蔽语言模型任务的交叉熵损失。

```
mlm_Y = torch.tensor([[7, 8, 9], [10, 20, 30]])
loss = nn.CrossEntropyLoss(reduction='none')
mlm_l = loss(mlm_Y_hat.reshape((-1, vocab_size)), mlm_Y.reshape(-1))
mlm_l.shape
```

#####  下一句预测

下面的`NextSentencePred`类使用单隐藏层的多层感知机来预测第二个句子是否是BERT输入序列中第一个句子的下一个句子。由于Transformer编码器中的自注意力，特殊词元“”的BERT表示已经对输入的两个句子进行了编码。因此，多层感知机分类器的输出层（`self.output`）以`X`作为输入，其中`X`是多层感知机隐藏层的输出，而MLP隐藏层的输入是编码后的“”词元。

```
#@save
class NextSentencePred(nn.Module):
    """BERT的下一句预测任务"""
    def __init__(self, num_inputs, **kwargs):
        super(NextSentencePred, self).__init__(**kwargs)
        self.output = nn.Linear(num_inputs, 2)

    def forward(self, X):
        # X的形状：(batchsize,num_hiddens)
        return self.output(X)
```



我们可以看到，`NextSentencePred`实例的前向推断返回每个BERT输入序列的二分类预测。

```
encoded_X = torch.flatten(encoded_X, start_dim=1)
# NSP的输入形状:(batchsize，num_hiddens)
nsp = NextSentencePred(encoded_X.shape[-1])
nsp_Y_hat = nsp(encoded_X)
nsp_Y_hat.shape
```

#### 整合代码

在预训练BERT时，最终的损失函数是掩蔽语言模型损失函数和下一句预测损失函数的线性组合。现在我们可以通过实例化三个类`BERTEncoder`、`MaskLM`和`NextSentencePred`来定义`BERTModel`类。前向推断返回编码后的BERT表示`encoded_X`、掩蔽语言模型预测`mlm_Y_hat`和下一句预测`nsp_Y_hat`。

```
#@save
class BERTModel(nn.Module):
    """BERT模型"""
    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, num_layers, dropout,
                 max_len=1000, key_size=768, query_size=768, value_size=768,
                 hid_in_features=768, mlm_in_features=768,
                 nsp_in_features=768):
        super(BERTModel, self).__init__()
        self.encoder = BERTEncoder(vocab_size, num_hiddens, norm_shape,
                    ffn_num_input, ffn_num_hiddens, num_heads, num_layers,
                    dropout, max_len=max_len, key_size=key_size,
                    query_size=query_size, value_size=value_size)
        self.hidden = nn.Sequential(nn.Linear(hid_in_features, num_hiddens),
                                    nn.Tanh())
        self.mlm = MaskLM(vocab_size, num_hiddens, mlm_in_features)
        self.nsp = NextSentencePred(nsp_in_features)

    def forward(self, tokens, segments, valid_lens=None,
                pred_positions=None):
        encoded_X = self.encoder(tokens, segments, valid_lens)
        if pred_positions is not None:
            mlm_Y_hat = self.mlm(encoded_X, pred_positions)
        else:
            mlm_Y_hat = None
        # 用于下一句预测的多层感知机分类器的隐藏层，0是“<cls>”标记的索引
        nsp_Y_hat = self.nsp(self.hidden(encoded_X[:, 0, :]))
        return encoded_X, mlm_Y_hat, nsp_Y_hat
```

#### 小结

- word2vec和GloVe等词嵌入模型与上下文无关。它们将相同的预训练向量赋给同一个词，而不考虑词的上下文（如果有的话）。它们很难处理好自然语言中的一词多义或复杂语义。
- 对于上下文敏感的词表示，如ELMo和GPT，词的表示依赖于它们的上下文。
- ELMo对上下文进行双向编码，但使用特定于任务的架构（然而，为每个自然语言处理任务设计一个特定的体系架构实际上并不容易）；而GPT是任务无关的，但是从左到右编码上下文。
- BERT结合了这两个方面的优点：它对上下文进行双向编码，并且需要对大量自然语言处理任务进行最小的架构更改。
- BERT输入序列的嵌入是词元嵌入、片段嵌入和位置嵌入的和。
- 预训练包括两个任务：掩蔽语言模型和下一句预测。前者能够编码双向上下文来表示单词，而后者则显式地建模文本对之间的逻辑关系。