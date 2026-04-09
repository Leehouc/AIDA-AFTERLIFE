# AIDA-AFTERLIFE

AIDA是一个面向六子棋的训练与推理项目

## 项目特点

1. **多维显式表征与轻量化输入** ：构建轻量多通道棋盘矩阵，将双方落子状态、上一手位置、合法掩码及对局阶段进行显式编码，为网络提供无歧义的高效底层数据支撑。
2. **残差驱动的双头决策架构** ：以深度残差网络（ResNet）作为策略-价值网络的主干，同步输出落点概率分布与局面胜负评估。
3. **深度适配“落双子”的规则约束** ：针对六子棋核心规则，在“第二子”阶段引入基于优先级距离的合法性约束，从数据层直接阻断海量无效分支，消除规则壁垒。
4. **双重注意力机制的特征增强** ：融合空间注意力与坐标注意力机制，既聚焦局部交战区域的特征激活，又精准捕获长距离连子的坐标依赖，实现关键信息的精细化建模。
5. **全链路评估体系** ：指标体系同时覆盖“策略预测准确率”与“价值回归误差”，全面、客观地衡量模型在博弈决策中的真实水平。

## 模型评估结果

### 指标含义：

- `value_mse`：价值头映射到 `{-1,0,1}` 后的均方误差
- `policy_top1`：真实落点被排在第 1 名的比例
- `policy_top5`：真实落点出现在前 5 个候选中的比例

### value_mse

| 模型/论文                                 | 棋种         |               数值 | 来源                                                                                                                                                                            |
| ----------------------------------------- | ------------ | -----------------: | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| AIDA（我的模型）                          | Connect6     |  **0.22131** | —                                                                                                                                                                              |
| AlphaGo value network                     | Go           |    **0.234** | ([Google Cloud Storage](https://storage.googleapis.com/deepmind-media/alphago/AlphaGoNaturePaper.pdf "https://storage.googleapis.com/deepmind-media/alphago/AlphaGoNaturePaper.pdf")) |
| Multi-Labelled Value Networks（VN）       | Go           |  **0.35388** | ([arXiv](https://arxiv.org/pdf/1705.10701 "https://arxiv.org/pdf/1705.10701"))                                                                                                        |
| Multi-Labelled Value Networks（ML-VN）    | Go           | **0.346082** | ([arXiv](https://arxiv.org/pdf/1705.10701 "https://arxiv.org/pdf/1705.10701"))                                                                                                        |
| Multi-Labelled Value Networks（BV-VN）    | Go           |  **0.35366** | ([arXiv](https://arxiv.org/pdf/1705.10701 "https://arxiv.org/pdf/1705.10701"))                                                                                                        |
| Multi-Labelled Value Networks（BV-ML-VN） | Go           | **0.348138** | ([arXiv](https://arxiv.org/pdf/1705.10701 "https://arxiv.org/pdf/1705.10701"))                                                                                                        |
| GomokuNet / GomokuNet                     | Renju/Gomoku |    **0.664** | ([IEEE COG](https://ieee-cog.org/2021/assets/papers/paper_286.pdf "https://ieee-cog.org/2021/assets/papers/paper_286.pdf"))                                                           |

### policy_top1

| 模型/论文                                                     | 棋种     |               数值 | 来源                                                                                                                                                                            |
| ------------------------------------------------------------- | -------- | -----------------: | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| AIDA（我的模型）                                              | Connect6 | **73.4214%** | —                                                                                                                                                                              |
| AlphaGo supervised policy network                             | Go       |    **57.0%** | ([Google Cloud Storage](https://storage.googleapis.com/deepmind-media/alphago/AlphaGoNaturePaper.pdf "https://storage.googleapis.com/deepmind-media/alphago/AlphaGoNaturePaper.pdf")) |
| AlphaGo supervised policy network                             | Go       |    **55.7%** | ([Google Cloud Storage](https://storage.googleapis.com/deepmind-media/alphago/AlphaGoNaturePaper.pdf "https://storage.googleapis.com/deepmind-media/alphago/AlphaGoNaturePaper.pdf")) |
| Learning to Play Othello with DNN（Conv8+BN）                 | Othello  |    **62.7%** | ([arXiv](https://arxiv.org/pdf/1711.06583 "https://arxiv.org/pdf/1711.06583"))                                                                                                        |
| Learning to Play Othello with DNN（Conv8+BN+bagging）         | Othello  |    **64.0%** | ([arXiv](https://arxiv.org/pdf/1711.06583 "https://arxiv.org/pdf/1711.06583"))                                                                                                        |
| Game Phase Specific Models in AlphaZero（opening expert）     | Chess    |   **68.24%** | ([ml-research.github.io](https://ml-research.github.io/papers/helfenstein2024game.pdf "https://ml-research.github.io/papers/helfenstein2024game.pdf"))                                |
| Game Phase Specific Models in AlphaZero（no-phases baseline） | Chess    |   **56.31%** | ([ml-research.github.io](https://ml-research.github.io/papers/helfenstein2024game.pdf "https://ml-research.github.io/papers/helfenstein2024game.pdf"))                                |

### policy_top5

| AIDA（我的模型）      | 棋种         |               数值 | 来源                                                                                                                  |
| --------------------- | ------------ | -----------------: | --------------------------------------------------------------------------------------------------------------------- |
| 你的新结果            | Connect6     | **91.4914%** | —                                                                                                                    |
| GomokuNet / ConvNet   | Renju/Gomoku |   **77.00%** | ([IEEE COG](https://ieee-cog.org/2021/assets/papers/paper_286.pdf "https://ieee-cog.org/2021/assets/papers/paper_286.pdf")) |
| GomokuNet / AlphaNet  | Renju/Gomoku |   **80.22%** | ([IEEE COG](https://ieee-cog.org/2021/assets/papers/paper_286.pdf "https://ieee-cog.org/2021/assets/papers/paper_286.pdf")) |
| GomokuNet / PolyNet   | Renju/Gomoku |   **82.88%** | ([IEEE COG](https://ieee-cog.org/2021/assets/papers/paper_286.pdf "https://ieee-cog.org/2021/assets/papers/paper_286.pdf")) |
| GomokuNet / MobileNet | Renju/Gomoku |   **83.51%** | ([IEEE COG](https://ieee-cog.org/2021/assets/papers/paper_286.pdf "https://ieee-cog.org/2021/assets/papers/paper_286.pdf")) |
| GomokuNet / KataNet   | Renju/Gomoku |   **84.08%** | ([IEEE COG](https://ieee-cog.org/2021/assets/papers/paper_286.pdf "https://ieee-cog.org/2021/assets/papers/paper_286.pdf")) |
| GomokuNet / GomokuNet | Renju/Gomoku |   **85.02%** | ([IEEE COG](https://ieee-cog.org/2021/assets/papers/paper_286.pdf "https://ieee-cog.org/2021/assets/papers/paper_286.pdf")) |

## 数据集

- 最终实验的数据集包含 `891227` 个样本
- 数据来自爬取近年优质六子棋对局日志
- /data下提供一条数据样例以供参考

## 快速启动

### 1. 安装依赖

```powershell
pip install -r requirements.txt
```

### 2. 构建数据集

```powershell
python prepare_dataset.py --input data --output-dir outputs/dataset
```

## 3. 训练

```powershell
python train.py --data-dir outputs/dataset --output-dir outputs/run01 --epochs 30 --batch-size 128 --blocks 34 --channels 128 --lr 2.5e-4 --weight-decay 5e-4 --value-loss-weight 0.25 --label-smoothing 0.05 --augment-symmetry --device cuda --amp --num-workers 8
```

输出目录中包含：

- `best.pt`
- `last.pt`
- `metrics.jsonl`
- `train_args.json`

### 4. 评估

```powershell
python evaluate.py --checkpoint outputs/run01/best.pt --data-dir outputs/dataset --split val --batch-size 128 --device cuda --amp
```

### 5. 导出比赛模型

```powershell
python export_competition_model.py --checkpoint outputs/run01/best.pt --output outputs/run01/con6_resnet_big.pth
```
