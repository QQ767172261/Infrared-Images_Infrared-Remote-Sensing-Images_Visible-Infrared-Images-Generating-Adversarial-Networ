红外图像/红外遥感图像/可见光红外图像对  近红外和可见光成对图像 生成对抗网络的风格迁移，或者图像融合/图像生成/图像转换 可见光遥感生成红外遥感图像，37500对图像数据
@[toc]
以下文字及代码仅供参考学习使用。

## **数据集描述：**


**红外图像/红外遥感图像/可见光红外图像对
包含近红外和可见光成对图像
针对于遥感地形数据，森林河流湖泊山脉等
主要用于生成对抗网络的风格迁移，或者图像融合/图像生成/图像转换
可见光遥感生成红外遥感图像，共37500对图像数据**

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/775bf85d32054f7798d0efba08221165.png)
使用 **生成对抗网络（GAN）** 来实现 **可见光遥感图像生成红外遥感图像** 或 **风格迁移 / 图像融合 / 图像转换**，使用 **Pix2Pix** 或 **CycleGAN** 经典图像到图像转换模型。代码示例，仅供参考。

- 数据准备（可见光 + 红外图像对）
- 模型选择（Pix2Pix）
- 环境搭建
- 模型训练、推理、评估代码
- 可视化结果

---

## 🧾 项目背景


- **37500对可见光和红外图像对**
- 可用于训练 **图像到图像的转换模型**
- 目标：**将可见光图像转换为红外图像**（图像生成 / 风格迁移 / 融合）

---
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/b2713a36ed724aa0b9a2fb368e342d13.png)

## 🧰 一、环境准备

### 1. 安装 Anaconda（如未安装）

下载地址：[https://www.anaconda.com/products/distribution](https://www.anaconda.com/products/distribution)

### 2. 创建虚拟环境

```bash
conda create -n ir_gan python=3.9
conda activate ir_gan
```

### 3. 安装依赖项

```bash
pip install torch torchvision torchaudio
pip install opencv-python matplotlib numpy
pip install tensorboard
```

---

## 📁 二、数据集结构与格式

你的数据应为成对图像，结构如下：

```
dataset/
├── train/
│   ├── A/
│   │   └── img1_A.jpg
│   │   └── img2_A.jpg
│   │   └── ...
│   ├── B/
│   │   └── img1_B.jpg
│   │   └── img2_B.jpg
│   │   └── ...
├── val/
│   ├── A/
│   ├── B/
└── test/
    ├── A/
    ├── B/
```

- A：可见光图像（输入）
- B：红外图像（目标输出）

---

## 🧠 三、选择模型：Pix2Pix GAN

我们选择 **Pix2Pix**，因为它是图像到图像转换的经典模型，适用于成对图像转换任务（如 A → B）。

GitHub 参考实现（PyTorch）：
- [https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)

你可以克隆项目：

```bash
git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
cd pytorch-CycleGAN-and-pix2pix
```

---

## 📦 四、数据预处理（构建成对图像）

你可以使用以下脚本将你的数据整理为 Pix2Pix 所需格式（拼接成双图）：

```python
import os
from PIL import Image

def create_pair_dataset(data_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    a_dir = os.path.join(data_dir, 'A')
    b_dir = os.path.join(data_dir, 'B')

    a_files = sorted(os.listdir(a_dir))
    b_files = sorted(os.listdir(b_dir))

    for a_file, b_file in zip(a_files, b_files):
        a_img = Image.open(os.path.join(a_dir, a_file)).convert('RGB')
        b_img = Image.open(os.path.join(b_dir, b_file)).convert('RGB')

        a_img = a_img.resize((256, 256))
        b_img = b_img.resize((256, 256))

        pair = Image.new('RGB', (256 * 2, 256))
        pair.paste(a_img, (0, 0))
        pair.paste(b_img, (256, 0))

        pair.save(os.path.join(output_dir, a_file))

# 示例
create_pair_dataset('dataset/train', 'pix2pix/datasets/ir_dataset/train')
create_pair_dataset('dataset/val', 'pix2pix/datasets/ir_dataset/val')
```

---

## 🚀 五、训练 Pix2Pix 模型

进入项目目录后，训练命令如下：

```bash
python train.py --dataroot ./datasets/ir_dataset --name ir2ir_model --model pix2pix --input_nc 3 --output_nc 3 --lambda_L1 100 --direction AtoB --display_id 0
```

### 参数说明：

- `--dataroot`：数据集路径
- `--name`：模型保存名称
- `--model`：模型类型（`pix2pix`）
- `--input_nc` / `--output_nc`：通道数（RGB）
- `--lambda_L1`：L1损失权重（增强细节）
- `--direction AtoB`：表示从 A（可见光）到 B（红外）转换
- `--display_id 0`：关闭可视化（可选）

---

## 🔍 六、模型推理（测试图像转换）

训练完成后，使用以下命令进行推理：

```bash
python test.py --dataroot ./datasets/ir_dataset --name ir2ir_model --model pix2pix --input_nc 3 --output_nc 3 --direction AtoB --num_test 100
```

推理结果将保存在：

```
results/ir2ir_model/test_latest/images/
```

包括：
- input_A：原始可见光图像
- fake_B：生成的红外图像
- real_B：真实红外图像（用于对比）

---

## 📈 七、可视化与评估

### 1. 可视化结果

你可以使用 OpenCV 或 Matplotlib 显示图像：

```python
import cv2
import os

result_dir = 'results/ir2ir_model/test_latest/images'
fake_B_files = [f for f in os.listdir(result_dir) if 'fake_B' in f]

for fake_B in fake_B_files[:5]:
    fake_B_img = cv2.imread(os.path.join(result_dir, fake_B))
    real_B_img = cv2.imread(os.path.join(result_dir, fake_B.replace('fake_B', 'real_B')))
    input_A_img = cv2.imread(os.path.join(result_dir, fake_B.replace('fake_B', 'input_A')))

    combined = cv2.hconcat([input_A_img, fake_B_img, real_B_img])
    cv2.imshow('Input A | Fake B | Real B', combined)
    cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 2. PSNR / SSIM 评估（可选）

```python
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def evaluate_pair(real_path, fake_path):
    real = cv2.imread(real_path)
    fake = cv2.imread(fake_path)

    p = psnr(real, fake)
    s = ssim(real, fake, multichannel=True, channel_axis=2)
    return p, s

# 示例
real_path = 'results/ir2ir_model/test_latest/images/real_B_1.png'
fake_path = 'results/ir2ir_model/test_latest/images/fake_B_1.png'
print("PSNR:", *evaluate_pair(real_path, fake_path))
```

---


