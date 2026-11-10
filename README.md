# Project\_AMR\_2026

pjj 的 EI 论文科研学习与实践记录。

本仓库用于整理自动调制识别（Automatic Modulation Classification，AMC）方向的文献、学习笔记、实验代码与阶段成果，并通过 Git 记录修改过程，方便指导老师审核与交流。

## 研究主题

* 研究领域：深度学习与无线电自动调制识别。
* 当前关注：信号特征表示、数据增强与迁移学习。
* 论文暂定题目：**Data Augmentation and Transfer Learning for Modulation Classification**。
* 当前阶段：科研工具搭建、文献调研与学术写作环境准备。

题目为现阶段暂定题目，后续将结合文献阅读和实验情况调整。

## 任务 0：开发环境与版本控制

开发环境信息见 [Environment\_Ready\_Report.md](Environment_Ready_Report.md)。

本项目使用 Git 管理版本，并通过 GitHub 仓库提交代码、文档与阶段成果。

## 任务 1：学术调研与文献管理

|任务|内容|交付材料|
|-|-|-|
|1.2 Zotero 文献管理|整理 AMC 相关文献，包含 1 篇综述与 2 篇研究型论文；集合名称为 `Topic\\\\\\\_AMC\\\\\\\_Multimodal`|[综述条目截图](Docs/Stage1_Screenshots/1_2_Zotero_Review.png) · [研究型论文 1](Docs/Stage1_Screenshots/1_2_Zotero_Research1.png) · [研究型论文 2](Docs/Stage1_Screenshots/1_2_Zotero_Research2.png)|
|1.3 SciSpace 辅助阅读|提取论文中用于调制识别的特征输入与模态类型，并整理总结|[AI 对话截图](Docs/Stage1_Screenshots/1_3_SciSpace_Chat.png) · [特征总结截图](Docs/Stage1_Screenshots/1_3_SciSpace_Features.png)|
|1.4 NotebookLM 知识整理|围绕“深度学习在无线电调制识别中的研究现状与挑战”生成并整理笔记|[生成结果截图](Docs/Stage1_Screenshots/1_4_NotebookLM.png) · [文字笔记](Docs/Stage1_Notes/NotebookLM_AMC_Notes.md)|
|1.5 Overleaf 学术排版|使用 IEEEtran 会议论文模板，编译包含暂定题目及 Hello World 的 PDF|[编译成功截图](Docs/Stage1_Screenshots/1_5_Overleaf.png)|

## 阶段交付材料清单

以下勾选表示材料已在本地整理，供提交及审核时逐项核对。

* \[x] 开发环境报告：`Environment\\\\\\\_Ready\\\\\\\_Report.md`
* \[x] Zotero：1 篇综述与 2 篇研究型论文的条目截图
* \[x] SciSpace：AI 对话与特征总结截图
* \[x] NotebookLM：研究现状与挑战的笔记截图
* \[x] Overleaf：IEEE 模板编译成功截图

## 文档存放位置

|路径|用途|
|-|-|
|`Environment\\\\\\\_Ready\\\\\\\_Report.md`|开发环境记录|
|`Docs/Stage1\\\\\\\_Screenshots/`|第一阶段任务截图|
|`Docs/Stage1\\\\\\\_Notes/NotebookLM\\\\\\\_AMC\\\\\\\_Notes.md`|NotebookLM 文字笔记|
|`README.md`|项目说明、任务记录与成果索引|

后续将随任务进展补充文献阅读记录、实验代码、实验结果与论文写作材料。



\## 第二阶段进度：云端验证



\- \[x] 在 Colab 中挂载 Google Drive，并读取合成 IQ 样本。

\- \[x] 在云端运行复数 IQ 的 FFT，生成双边频谱图。

\- \[x] 在云端运行 STFT，生成时频图，输出形状为 (512, 125)。

\- \[ ] 本地 Python 3.10 与 PyTorch GPU 环境验证。

\- \[ ] IDE 与 AI 编程助手的安装、对话及代码运行验证。



交付材料：



\- \[已运行的 Notebook](Notebooks/Stage2\_Colab.ipynb)

\- \[Drive 读取截图](Docs/Stage2\_Screenshots/2\_2\_Colab\_Drive.png)

\- \[FFT 结果图](Docs/Stage2\_Outputs/iq\_fft.png)

\- \[STFT 结果图](Docs/Stage2\_Outputs/stft\_spectrogram.png)



上述信号处理演示在 Colab 云端运行，数据为环境验证用合成样本。

\## 2.3 IDE与AI编程助手



\- \[x] 使用 VS Code + Codex 完善IQ信号FFT代码，并使用Python 3.10.22运行验证。

\- \[x] 合成样本的频谱主峰为+300 Hz、−700 Hz，归一化幅度约为1、0.5。



交付材料：



\- \[FFT代码](Docs/Stage2\_Outputs/code/iq\_fft\_demo.py)

\- \[中文说明](Docs/Stage2\_Outputs/code/README\_iq\_fft.md)

\- \[样本数据](Docs/Stage2/Data/Stage2/sample\_iq.csv)

\- \[AI对话截图](Docs/Stage2\_Screenshots/2\_3\_AI\_FFT\_Chat.png)

\- \[运行结果截图](Docs/Stage2\_Screenshots/2\_3\_FFT\_Result.png)



