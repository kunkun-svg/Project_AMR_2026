# 论文拆解报告：Deep Learning Based Automatic Modulation Recognition: Models, Datasets, and Challenges

* **发表年份与期刊/会议**：2022，*Digital Signal Processing*，Vol.
129，Article 103650
* **开源/常用数据集**：RML2016.10a、RML2016.10b、RML2018.01a、HisarMod2019.1

## 1\. 问题转化（Problem Formulation）

* 作者将自动调制识别转化为一个**多类别分类问题（Multi-class
Classification）**：模型根据接收端获得的无线信号或其特征，判断发送信号所属的调制类别，例如
BPSK、QPSK、8PSK、16QAM
等。本文综述了多种深度学习模型，并对代表性模型进行了分类性能比较。

## 2\. 输入与输出（Inputs \& Outputs）

* **输入（Input）**：主要输入形式包括原始 I/Q 采样数据。RML2016.10a 和
RML2016.10b 的单个样本维度为 **2×128**；RML2018.01a 和
HisarMod2019.1 的单个样本维度为
**2×1024**。论文综述的其他模型还使用幅度/相位、统计特征、谱相关函数、星座图或频谱等输入。
* **输出（Output）**：调制类别的 Softmax 概率向量。若候选调制类别数为
$C$，则输出为：

$$
\mathbf{p}=[p_1,p_2,\ldots,p_C]
$$

其中，$p\_i$ 表示模型预测样本属于第 $i$
类调制方式的概率。预测类别通常取概率最大的类别：

$$
\hat{c}=\arg\max_{1\leq i\leq C}p_i
$$

例如，RML2016.10a 包含 11 种调制类别，因此对应分类模型输出 11
维类别概率。

## 3\. 数据集与实验设置（Datasets \& Experimental Setup）

* **数据集名称**：作者在 SISO 模型对比实验中使用了
RML2016.10a、RML2016.10b、RML2018.01a 和 HisarMod2019.1 四个数据集。
* **信噪比范围（SNR）**：

  * RML2016.10a：−20 dB 至 +18 dB，步长 2 dB。
  * RML2016.10b：−20 dB 至 +18 dB，步长 2 dB。
  * RML2018.01a：−20 dB 至 +30 dB，步长 2 dB。
  * HisarMod2019.1：−20 dB 至 +18 dB，步长 2 dB。

## 4\. 评价标准（Evaluation Metrics）

* **Recognition Accuracy**：不同 SNR 条件下的调制识别准确率，并绘制
Accuracy vs. SNR 曲线。
* **Confusion
Matrix**：混淆矩阵，用于分析不同调制类别之间的误分类情况，主对角线和次对角线都有实际的统计意义。
* **模型复杂度与训练过程指标**：可学习参数量（Learning
Parameters）、每个 epoch 的训练时间（Training
Time）、训练轮数（Training Epochs）和最小验证损失（Minimum
Validation Loss）。论文没有将单次推理延迟（Inference
Latency）作为统一对比指标报告。

## 5\. 对咱们课题的启发（Key Takeaways）

* **数据增强**：论文指出可以探索使用 GAN 作为生成器扩充 AMR
训练集，并将数据清洗与数据增强列为值得关注的问题。可借鉴这一思路扩充不同信道条件下的
I/Q 训练样本；但本文没有实际完成 GAN 数据增强实验。
* **迁移学习**：论文提出可以先使用 RML2016.10a
等离线数据集训练模型，再使用真实环境采集的在线数据进行微调，以提高模型对实际信道条件的适应能力。论文还介绍了已有的半监督迁移学习方法：先用无标签数据训练
CAE，再将编码器权重迁移到 AMR CNN 的特征层。

