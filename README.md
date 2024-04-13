# Transformers 机器翻译项目

这个项目实现了使用自定义的transformer模型进行英语到中文的机器翻译。通过该项目，你可以了解到如何构建transformer的各个组件，并使用这些组件训练一个机器翻译模型。

## 项目结构

- `transformers_define.py`: 包含构建transformer模型的代码。
- `train_loading.py`: 实现了训练所需的批处理、损失函数等代码。
- `config.py`: 包含基本的参数设置，包括字典路径、训练数据路径等。
- `train.py`: 用于构建模型、进行训练和保存模型的代码文件。
- `/data`: 包含训练和测试数据。

## 如何使用

### 1. 数据准备

在`/data`目录下准备你的训练和测试数据。
例如：
### 2. 配置参数

在`config.py`中配置你的参数：

- `dict_zh`: 中文词汇字典
- `dict_en`: 英文词汇字典
- `train_en`: 英文训练语料
- `train_zh`: 中文训练语料
- `epoch_num`: 迭代次数


### 3. 训练模型

运行`train.py`文件来开始训练模型：

```bash
python train.py