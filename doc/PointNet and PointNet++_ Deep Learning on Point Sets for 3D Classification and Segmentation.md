# 介绍
​	点云是一种重要的几何数据结构。由于其不规则的格式，大多数研究人员将这些数据转换为规则的3D体素网格或图像集合。然而，这使得数据不必要地庞大，并造成问题。研究人员设计了一种新型的神经网络PointNet，它直接以点云作为输入，考虑了输入点的排列不变性。PointNet提供了一个统一的架构，用于对象分类，物体分割，场景分割。

![f1](/doc/img/f1.png)

# 问题描述
​	研究者设计了一个深度学习框架，直接使用无序点集作为输入。点云表示为一组3D点{Pi| i = 1，.，n}，其中每个点Pi是其（x，y，z）坐标加上诸如颜色、法线等的额外特征通道的向量。
​	对于对象分类任务，Pointnet为所有k个候选类输出k个分数。对于语义分割，输入可以是用于部分区域分割的单个对象，或者是用于对象区域分割的3D场景。Pointnet将为n个点中的每个点生成m个语义类别分数，共输出n × m个分数。

# 相关工作
## 3D计算机视觉
​	Volumetric CNNs：将三维卷积神经网络应用于体素化3D视觉学习。然而，由于3D数据的稀疏性和三维卷积的计算成本，体素表示受到其分辨率的限制。FPNN和Vote3D提出了处理稀疏性问题的特殊方法，然而，处理非常大的点云仍然是一个挑战。
​	Multiview CNNs：将三维点云渲染成二维图像，然后应用二维卷积网络对其进行分类。通过设计良好的图像学习网络，这种方法在形状分类和检索任务上取得了主要的性能。然而，将它们扩展到场景分割或其他3D任务，如点云补全是非常困难的。
​	Feature-based DNNs：将三维数据转换为一个向量，然后利用全连接网络对形状进行分类。这样提取到特征的表示能力有很大的约束，因为将无序的点云转化成了有序的向量。

# 点云特性
​	无序性，与图像中的像素阵列或体素网格中的体素阵列不同，点云是一组没有特定顺序的点，输入为N个点的网络需要对N!保持输出不变性。
​	交互性，点云中的这些点来自于具有相同距离度量的欧几里得空间。这意味着点与点之间不是孤立的，相邻的点形成了一个相互关联的子集，这意味着模型需要从附近的点捕获局部结构的语义特征。
​	仿射性，点云是一个完整的仿射对象，即对整个点云进行仿射变换如旋转和平移后仍和原始点云语义特征保持一致。

# PointNet

## 针对无序输入的对称函数

​	应用点对称函数来近似点集上的一般函数，通过多层感知器网络近似h，通过单变量函数和最大池函数的组合近似g
$$
\begin{equation*} f(\{x_{1},\ \ldots,\ x_{n}\})\approx g(h(x_{1}),\ \ldots,\ h(x_{n}))\end{equation*}
$$

$$
\begin{equation*} \forall\epsilon >0,\exists h,\gamma,\quad st. \left\vert f(S)-\gamma\left(\underset{x_{i}\in S}{MAX} \{h(x_{i})\}\right)\right\vert < \epsilon \end{equation*}
$$

## 局部特征和全局特征的聚合

​	在计算全局点云特征向量之后，通过将全局特征与每个点特征拼接起来反馈给每个点。然后我们基于拼接的点特征提取新的逐点特征，其中包含了局部信息和全局信息。

## 仿射变换对齐网络

​	通过一个T-net预测仿射变换矩阵，并直接将此变换应用于输入点云，其中T-net只由简单的全连接层组成。这一思想也可以进一步扩展到特征空间的对齐。我们可以对逐点特征应用另一个对齐网络，使用同样的方法预测特征仿射变换矩阵，以对齐来自不同输入点云的特征。为了将特征变换矩阵约束为接近正交矩阵，将正则化项添加到训练损失中。
$$
\begin{equation*} L_{reg}=\Vert I-AA^{T}\Vert_{F}^{2},\tag{2} \end{equation*}
$$

## 局限性

​	PointNet只能捕捉点云的全局特征，不能捕获欧氏空间点集的局部结构特征，从而限制了其识别细粒度模式的能力和对复杂场景的泛化能力。

# PointNet++

## 分层点云特征学习

​	使用分层特征学习代替PointNet中使用单个最大池化聚合点云特征。具体来说，每一个抽象层由一个采样层，一个分组层和一个学习层组成。

​	采样层，给定输入特征点集，使用迭代最远点采样（FPS）来选择特征点集的子集，与随机采样相比，在相同的质心数下，该方法对整个点集的覆盖率更高。

​	分组层，使用球查询搜索到质心点的特征距离在半径内的所有点（在实现中设置上限K），与kNN相比，球查询的局部邻域保证了固定的区域尺度，从而使局部区域特征在空间上更具泛化性，且利于学习层中的PointNet提取特征。

​	学习层，使用PointNet学习逐点特征，局部区域中的点的坐标首先被转换成相对于质心点的局部坐标系，通过PointNet得到该局部区域的全局特征作为下一个采样层的输入。

![f2](/doc/img/f2.png)

## 采样优化策略：非均匀采样

​	多尺度分组（MSG），如图（a）所示，在每一个抽象层中使用具有不同尺度的分组层，然后根据PointNets提取每个尺度的特征。不同尺度的特征被连接以形成多尺度特征。

​	多分辨率分组（MRG），由于上面的MSG计算开销较大，因为它在每个质心点的大规模邻域中运行PointNet。通常由于在底层时质心点的数量相当大，因此时间成本是高昂的。MRG的层特征是两个向量的拼接。一个向量（图中左侧）是通过使用单尺度分组（SSG）汇总每个子区域的特征而获得的。另一个矢量（右）是通过使用单个PointNet直接处理局部区域中的所有原始点获得的特征。当局部区域的密度低时，左侧向量比右侧向量更不可靠，因为在计算左侧向量时子区域点云稀疏导致采样不足。在这种情况下，右侧向量的权重应该更高。相反，当局部区域的密度高时，左侧矢量提供了更精细的信息，此时应该提高左侧向量的权重。

![f3](/doc/img/f3.png)

## 基于点特征传播的点云分割方法

​	使用基于特征距离的插值传播方法逆向生成前一个抽象层所有点的近邻特征，再与前一个抽象层的分组学习特征拼接为逐点分割特征，使用PointNet的部分仿射架构进行特征学习，进行下一轮特征传播，直到传播至原始点云。在插值中，使用k个最近邻的反距离加权平均特征。


$$
f^{(j)}(x)=\frac{\sum_{i=1}^{k} w_{i}(x) f_{i}^{(j)}}{\sum_{i=1}^{k} w_{i}(x)} \quad \text { where } \quad w_{i}(x)=\frac{1}{d\left(x, x_{i}\right)^{p}}, j=1, \dots, C
$$










## 代码链接

我们代码的链接是：https://rec.ustc.edu.cn/share/bf6ef200-a2d1-11ee-a5ba-9b52b7de6191

## 工作类型

首先，我们成功运行了pointnet和pointnet++的开源代码。此外，因为论文作者没有给出MRG策略的实现代码，网上也搜索不到该策略的实现，所以我们自己写代码复现了分类任务下的MRG策略。

## 运行环境配置说明

我们的实验环境是NVIDIA GeForce RTX 3090，Pytorch 1.10.1，Python 3.7。创建好Python版本为3.7的conda虚拟环境之后，使用以下命令安装所需要的包和库：

```
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
```

```
conda install tqdm
```

并安装用于可视化的库：

```
conda install opencv
```

## 数据集介绍及下载链接

本工作分别在分类任务、部件分割任务和语义分割任务下进行了实验，以下是各个任务对应的数据集：

**ModelNet40数据集（用于分类任务）：**

ModelNet40数据集包含了40个类别（如飞机，汽车等）的CAD模型数据。训练集有9843个点云数据，测试集有2468个点云数据。

下载链接：https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip

**ShapeNet数据集（用于部件分割任务）：**ShapeNet数据集包含14个大类别（如飞机，椅子等）和55个小类别的CAD模型数据。训练集有14007个点云数据，测试集有2874个点云数据。

下载链接：https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip

**S3DIS数据集（用于语义分割任务）：**S3DIS是大规模室内三维空间数据集，包含了6个建筑物，共计271个房间。房间内包含13类物体： 天花板、地板、门、沙发等。

下载链接：https://pan.baidu.com/s/1BtDCIDYEbILDWms1__9tpw?pwd=0acs

## 指标介绍

本工作分别在分类、部件分割和语义分割下进行了实验，以下是各个任务对应的评价指标：

**ModelNet40数据集上的分类任务的评价指标：**

accuracy：准确率，即分类正确的样本占总样本个数的比例。

**ShapeNet数据集上的部件分割任务的评价指标：**

mIoU：平均交并比，计算公式为：

$mIoU = \dfrac{1}{k+1}\sum_{i=0}^{k}\dfrac{p_{ii}}{\sum_{j=0}^{k}p_{ij}+\sum_{j=0}^{k}p_{ji}-p_{ii}}$

其中，k为类别数，i表示真实值，j表示预测值，$p_{ij}$表示将i预测为j的点云个数，$p_{ji}$表示将j预测为i的点云个数。

**S3DIS数据集上的语义分割任务的评价指标：**

accuracy：准确率。

class avg IoU：类别平均交并比，计算语义分割中每个类的交并比，然后求平均值。

## 代码说明

代码结构及说明如下：

```python
└── Pointnet_Pointnet2_pytorch
    ├── data							# 存放三个任务对应的数据集
    ├── data_utils
    │   ├── collect_indoor3d_data.py	# S3DIS数据集的预处理
    │   ├── indoor3d_util.py
    │   ├── meta
    │   │   ├── anno_paths.txt
    │   │   └── class_names.txt
    │   ├── ModelNetDataLoader.py		# ModelNet40数据集的DataLoader
    │   ├── S3DISDataLoader.py			# S3DIS数据集的DataLoader
    │   └── ShapeNetDataLoader.py		# ShapeNet数据集的DataLoader
    ├── .gitattributes
    ├── .gitignore
    ├── LICENSE
    ├── log								# 存放训练日志
    │   ├── classification
    │   ├── part_seg
    │   └── sem_seg
    ├── models							# 存放可以使用的模型方法
    │   ├── pointnet2_cls_mrg.py		# 基于MRG策略执行分类任务的pointnet++模型
    │   ├── pointnet2_cls_msg.py		# 基于MSG策略执行分类任务的pointnet++模型
    │   ├── pointnet2_cls_ssg.py		# 基于SSG策略执行分类任务的pointnet++模型
    │   ├── pointnet2_part_seg_msg.py	# 基于MSG策略执行部件分割任务的pointnet++模型
    │   ├── pointnet2_part_seg_ssg.py	# 基于SSG策略执行部件分割任务的pointnet++模型
    │   ├── pointnet2_sem_seg_msg.py	# 基于MSG策略执行语义分割任务的pointnet++模型
    │   ├── pointnet2_sem_seg.py		# 基于SSG策略执行语义分割任务的pointnet++模型
    │   ├── pointnet2_utils.py
    │   ├── pointnet_cls.py				# 执行分类任务的pointnet模型
    │   ├── pointnet_part_seg.py		# 执行部件分割任务的pointnet模型
    │   ├── pointnet_sem_seg.py			# 执行语义分割任务的pointnet模型
    │   └── pointnet_utils.py
    ├── provider.py
    ├── README.md
    ├── test_classification.py			# 测试分类任务网络
    ├── test_partseg.py					# 测试部件分割任务网络
    ├── test_semseg.py					# 测试语义分割任务网络
    ├── train_classification.py			# 训练分类任务网络
    ├── train_partseg.py				# 训练部件分割任务网络
    ├── train_semseg.py					# 训练语义分割任务网络
    └── visualizer
        ├── build.sh					# 编译render_balls_so.cpp
        ├── eulerangles.py
        ├── pc_utils.py
        ├── plyfile.py
        ├── render_balls_so.cpp			# 用于ShapeNet数据集可视化的C++代码
        └── show3d_balls.py				# 用于ShapeNet数据集可视化的python代码
```

## 代码运行说明

### 1、对源数据进行可视化，方便分析源数据特点

**ModelNet40数据集：**

源数据格式为txt，每个点包含6个维度的信息，分别是[x, y, z, nx, ny, nz]。其中，(x, y, z)表示该点在空间中的坐标，(nx, ny, nz)表示该点在空间中的法向量。我们使用软件[MeshLab](https://www.meshlab.net/)进行可视化。

**ShapeNet数据集：**

源数据格式为txt，每个点包含7个维度的信息，分别是[x, y, z, nx, ny, nz, type]。其中，(x, y, z)表示该点在空间中的坐标，(nx, ny, nz)表示该点在空间中的法向量，type是部件编号（该点属于哪个部件）。我们运行以下代码，对随机一个源数据进行可视化：

```
## build C++ code for visualization
cd visualizer
bash build.sh 
## run one example 
python show3d_balls.py
```

**S3DIS数据集：**

源数据格式为txt，每个点包含6个维度的信息，分别是[x, y, z, r, g, b]。其中，(x, y, z)表示该点在空间中的坐标，(r, g, b)表示该点的rgb颜色信息。我们使用软件[CloudCompare](https://cloudcompare.org/)进行可视化。

### 2、在ModelNet40数据集上进行分类

#### 1) 数据准备

将下载的ModelNet40解压，保存为`./data/modelnet40_normal_resampled/`

#### 2) 运行参数说明

①`--model`：可以选择使用哪种模型进行分类任务。

在`./models`目录下，对于分类任务，有4个模型，分别是：

- pointnet_cls(pointnet模型)
- pointnet2_cls_ssg(采用SSG策略的pointnet++模型)
- pointnet2_cls_msg(采用MSG策略的pointnet++模型)
- pointnet2_cls_mrg(采用MRG策略的pointnet++模型)。**MRG策略的python代码是我们自己实现的**。

②`--use_normals`：是否使用法向量信息。

③`--log_dir`：生成日志文件。

#### 3) 代码运行(训练与测试)

**①pointnet without normal：**

```
python train_classification.py --model pointnet_cls --log_dir pointnet_cls
python test_classification.py --log_dir pointnet_cls
```

**②pointnet with normal：**

```
python train_classification.py --model pointnet_cls --use_normals --log_dir pointnet_cls_norm
python test_classification.py --use_normals --log_dir pointnet_cls_norm
```

**③pointnet2_ssg without normal：**

```
python train_classification.py --model pointnet2_cls_ssg --log_dir pointnet2_cls_ssg
python test_classification.py --log_dir pointnet2_cls_ssg
```

**④pointnet2_ssg with normal：**

```
python train_classification.py --model pointnet2_cls_ssg --use_normals --log_dir pointnet2_cls_ssg_normal
python test_classification.py --use_normals --log_dir pointnet2_cls_ssg_normal
```

**⑤pointnet2_msg without normal：**

```
python train_classification.py --model pointnet2_cls_msg --log_dir pointnet2_cls_msg
python test_classification.py --log_dir pointnet2_cls_msg
```

**⑥pointnet2_msg with normal：**

```
python train_classification.py --model pointnet2_cls_msg --use_normals --log_dir pointnet2_cls_msg_normal
python test_classification.py --use_normals --log_dir pointnet2_cls_msg_normal
```

**⑦pointnet2_mrg without normal：**

```
python train_classification.py --model pointnet2_cls_mrg --log_dir pointnet2_cls_mrg
python test_classification.py --log_dir pointnet2_cls_mrg
```

**⑧pointnet2_mrg with normal：**

```
python train_classification.py --model pointnet2_cls_mrg --use_normals --log_dir pointnet2_cls_mrg_normal
python test_classification.py --use_normals --log_dir pointnet2_cls_mrg_normal
```

### 3、在ShapeNet数据集上进行部件分割

#### 1) 数据准备

将下载的ShapeNet解压，保存为`./data/shapenetcore_partanno_segmentation_benchmark_v0_normal/`

#### 2) 运行参数说明

①`--model`：可以选择使用哪种模型进行部件分割任务。

在`./models`目录下，对于部件分割任务，有3个模型，分别是：

- pointnet_part_seg(pointnet模型)
- pointnet2_part_seg_ssg(采用SSG策略的pointnet++模型)
- pointnet2_part_seg_msg(采用MSG策略的pointnet++模型)

②`--normal`：是否使用法向量信息。

③`--log_dir`：生成日志文件。

#### 3) 代码运行(训练与测试)

**①pointnet without normal：**

```
python train_partseg.py --model pointnet_part_seg --log_dir pointnet_part_seg_wo_normal
python test_partseg.py --log_dir pointnet_part_seg_wo_normal
```

**②pointnet with normal：**

```
python train_partseg.py --model pointnet_part_seg --normal --log_dir pointnet_part_seg
python test_partseg.py --normal --log_dir pointnet_part_seg
```

**③pointnet2_ssg without normal：**

```
python train_partseg.py --model pointnet2_part_seg_ssg --log_dir pointnet2_part_seg_ssg_wo_normal
python test_partseg.py --log_dir pointnet2_part_seg_ssg_wo_normal
```

**④pointnet2_ssg with normal：**

```
python train_partseg.py --model pointnet2_part_seg_ssg --normal --log_dir pointnet2_part_seg_ssg
python test_partseg.py --normal --log_dir pointnet2_part_seg_ssg
```

**⑤pointnet2_msg without normal：**

```
python train_partseg.py --model pointnet2_part_seg_msg --log_dir pointnet2_part_seg_msg_wo_normal
python test_partseg.py --log_dir pointnet2_part_seg_msg_wo_normal
```

**⑥pointnet2_msg with normal：**

```
python train_partseg.py --model pointnet2_part_seg_msg --normal --log_dir pointnet2_part_seg_msg
python test_partseg.py --normal --log_dir pointnet2_part_seg_msg
```

### 4、在S3DIS数据集上进行语义分割，并对分割结果进行可视化

#### 1) 数据准备

将下载的S3DIS解压，保存为`./data/s3dis/Stanford3dDataset_v1.2_Aligned_Version/`

再进行数据预处理：

```
cd data_utils
python collect_indoor3d_data.py
```

处理后的数据将保存在`data/s3dis/stanford_indoor3d/`

#### 2) 运行参数说明

①`--model`：可以选择使用哪种模型进行语义分割任务。

在`./models`目录下，对于语义分割任务，有3个模型，分别是：

- pointnet_sem_seg(pointnet模型)
- pointnet2_sem_seg(采用SSG策略的pointnet++模型)
- pointnet2_sem_seg_msg(采用MSG策略的pointnet++模型)

②`--test_area`：采用6-fold交叉验证：训练时使用五个区域，测试时使用剩下的那个区域。通过该参数，可以指定测试使用的区域。

③`--log_dir`：生成日志文件。

④`--visual`：在测试时生成可视化结果(obj格式文件)并保存。

#### 3) 代码运行(训练与测试)

**①pointnet**：

```
python train_semseg.py --model pointnet_sem_seg --test_area 5 --log_dir pointnet_sem_seg
python test_semseg.py --log_dir pointnet_sem_seg --test_area 5 --visual
```

**②pointnet2_ssg**：

```
python train_semseg.py --model pointnet2_sem_seg --test_area 5 --log_dir pointnet2_sem_seg
python test_semseg.py --log_dir pointnet2_sem_seg --test_area 5 --visual
```

**③pointnet2_msg：**

```
python train_semseg.py --model pointnet2_sem_seg_msg --test_area 5 --log_dir pointnet2_sem_seg_msg
python test_semseg.py --log_dir pointnet2_sem_seg_msg --test_area 5 --visual
```

#### 4) 对分割结果进行可视化

以pointnet2_sem_seg_msg为例，可视化结果保存在`./log/sem_seg/pointnet2_sem_seg_msg/visual`。在该目录下，对于每个房间数据，分别有房间语义分割的ground truth(如Area_5_conferenceRoom_1_gt.obj)和predict结果(如Area_5_conferenceRoom_1_pred.obj)。

使用软件MeshLab，打开上述obj文件，即可得到ground truth和分割结果的可视化。

## 实验运行及复现结果

### 1、在ModelNet40数据集上进行分类，并查看显存占用

Subvolume和MVCNN是直接用的论文中的数据，pointnet和pointnet++的数据是自己跑出来的。

| 方法                           | Accuracy(%) | 显存占用(MB) |
| ------------------------------ | ----------- | ------------ |
| Subvolume                      | 89.2        | N/A          |
| MVCNN                          | 90.1        | N/A          |
| pointnet (without normal)      | 90.4        | 3053         |
| pointnet (with normal)         | 91.8        | 3025         |
| pointnet2_ssg (without normal) | 92.3        | 4671         |
| pointnet2_ssg (with normal)    | 92.5        | 4673         |
| pointnet2_msg (without normal) | 91.9        | 13671        |
| pointnet2_msg (with normal)    | **93.1**    | 13671        |
| pointnet2_mrg (without normal) | 92.2        | 10101        |
| pointnet2_mrg (with normal)    | 92.7        | 10101        |

可以看到：

- 使用法向量特征、采用MSG策略的pointnet++达到了最好的性能，而使用法向量特征、采用MRG策略的pointnet++的性能仅次于它。

- 虽然采用MSG策略的pointnet++性能最好，但是显存占用也是最高的。而采用MRG策略的pointnet++在性能没有明显下降的情况下，能够在一定程度上减少显存开销。

### 2、在ShapeNet数据集上进行部件分割：

Yi、SSCNN是直接用的论文中的数据，pointnet和pointnet++的数据是自己跑出来的。

| 方法                           | mIoU(%)  |
| ------------------------------ | -------- |
| Yi                             | 81.4     |
| SSCNN                          | 84.7     |
| pointnet (without normal)      | 83.3     |
| pointnet (with normal)         | 84.4     |
| pointnet2_ssg (without normal) | 85.1     |
| pointnet2_ssg (with normal)    | **85.5** |
| pointnet2_msg (without normal) | 85.2     |
| pointnet2_msg (with normal)    | **85.5** |

可以看到，使用法向量特征、采用SSG策略或者MSG策略的Pointnet++都达到了最好的性能。

### 3、在S3DIS数据集上进行语义分割，并对分割结果进行可视化

| 方法          | Accuracy(%) | Class avg IoU(%) |
| ------------- | ----------- | ---------------- |
| pointnet      | 78.6        | 41.9             |
| pointnet2_ssg | **83.4**    | 53.9             |
| pointnet2_msg | 83.0        | **63.2**         |

可以看到，相比于pointnet，无论采用哪种策略，pointnet++性能提升明显。

下图是数据集中一个房间的ground truth：

 <img src="D:\学习\研一上\计算机视觉\课程设计\report\img\exp_7_gt.png" alt="exp_7_gt" style="zoom: 33%;" />

以下是针对该房间的三种模型分割结果的可视化：

①pointnet：

 <img src="D:\学习\研一上\计算机视觉\课程设计\report\img\exp_7_pointnet_predict.png" alt="exp_7_pointnet_predict" style="zoom:33%;" />

②pointnet2_ssg：

 <img src="D:\学习\研一上\计算机视觉\课程设计\report\img\exp_7_ssg_predict.png" alt="exp_7_ssg_predict" style="zoom:33%;" />

③pointnet2_msg：

 <img src="D:\学习\研一上\计算机视觉\课程设计\report\img\exp_7_msg_predict.png" alt="exp_7_msg_predict" style="zoom:33%;" />





