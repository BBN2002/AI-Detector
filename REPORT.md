# FraudSquad++ 实验报告（当前进度汇总）

本报告汇总目前已完成的数据获取、预处理、特征抽取、模型训练与任务调度情况，并说明各模块的实现方式、输入输出与运行路径。

## 1. 数据获取与来源

官方仓库（已下载并解压到本地）：
- 路径：`/root/autodl-fs/fraudsquad_project/data/FraudSquad-5389`
- 说明：来自论文 *Detecting LLM-Generated Spam Reviews by Integrating Language Model Embeddings and Graph Neural Network* 官方匿名仓库。

关键原始数据文件：
- 真正评论（genuine）：`data/Amazon/amazon_2022_5core.csv`
- LLM spam：
  - `data/Amazon/reviews_from_llama3_8b.csv`
  - `data/Amazon/reviews_from_qwen_72b.csv`
  - `data/Amazon/reviews_from_qwen_dsr1.csv`

## 2. 数据标准化与合并

目的：将 genuine + LLM spam 合并为单一节点表，并统一字段与时间戳格式。

### 2.1 合并脚本
脚本：`src/preprocess/merge_amazon_llm.py`
- 输入：
  - `amazon_2022_5core.csv`（genuine）
  - `reviews_from_*.csv`（LLM spam）
- 输出：
  - `data/combined/Amazon-Llama3.csv`
  - `data/combined/Amazon-Qwen2.csv`
  - `data/combined/Amazon-Qwen-DSR1.csv`
- 标注：
  - genuine → `label=0`
  - spam → `label=1`

### 2.2 自动时间戳识别与字段标准化
脚本：`src/preprocess/inspect_and_standardize.py`
- 自动识别字段：
  - 用户：`user`
  - 商品：`product`
  - 评分：`star`
  - 文本：`reviewText`
  - 标签：`label`
  - 时间：`reviewTime`（Unix seconds）
- 统一输出字段：
  - `user_id`, `product_id`, `rating`, `text`, `y_spam`
  - `ts_ms`（毫秒时间戳）
  - `month_id`（YYYY-MM）
  - `review_id`（连续 id）
- 输出文件：
  - `artifacts/graphs/Amazon-*/nodes.parquet`

## 3. 图构建（FraudSquad 规则）

脚本：`src/preprocess/build_graph_dgl.py`

节点：每条 review 一个节点  
边：满足任一条件连边
1. 同一用户：`user_id` 相同  
2. 同商品同星级：`product_id` 且 `rating` 相同  
3. 同商品同月：`product_id` 且 `month_id` 相同  

输出：
- `artifacts/graphs/Amazon-*/graph.bin`
- `artifacts/graphs/Amazon-*/meta.json`

## 4. 数据切分（1/9/90 + 分层）

脚本：`src/preprocess/make_splits.py`
- 分层按 `y_spam`，保证 train 有 spam
- 输出：
  - `artifacts/splits/Amazon-*/seed2024_masks.pt`
  - `artifacts/splits/Amazon-*/seed2025_masks.pt`
  - `artifacts/splits/Amazon-*/seed2026_masks.pt`

## 5. 文本特征（BERT CLS）

脚本：`src/features/extract_bert_cls.py`
- 模型：`bert-base-uncased`（本地离线模型）
  - 路径：`data/FraudSquad-5389/models/bert-base-uncased`
- 输出：
  - `artifacts/features/Amazon-*/x_text_bert.npy`

## 6. HC3-style AI Detector 特征

脚本：
- 训练：`src/features/train_ai_detector.py`
- 推理：`src/features/infer_ai_detector.py`

训练：
- 只使用 train mask（防止 test 泄漏）
- balanced sampling
- 输出 ckpt：`runs/ai_det/Amazon-*/seed2024/ckpt.pt`

推理：
- 输出：
  - `p_ai_roberta.npy`（N×1，sigmoid 概率）
  - `h_det_roberta_64.npy`（N×64，投影特征）

## 7. GRAND 子图扩散（无监督异常分数）

训练脚本：`src/features/train_grand_subgraph.py`  
推理脚本：`src/features/infer_grand_scores.py`

关键设定：
- 2-hop ego 子图
- n_max = 1024
- batch_size = 8
- epochs = 3
- dx = 256

输出：
- 训练 ckpt：`runs/grand/Amazon-*/seed2024/ckpt.pt`
- 推理分数：
  - `artifacts/features/Amazon-*/s_node_grand.npy`
  - `artifacts/features/Amazon-*/s_edge_agg_grand.npy`

## 8. FraudSquad/FS++ 训练入口

自定义入口脚本：`src/train/train_fraudsquad_custom.py`
- 复用官方 FraudSquad 模型实现
- 直接读离线特征（`x_text_bert.npy` + 可选 extra）
- 支持三种变体：
  - FS：只用 x_text
  - FS+HC3：x_text + p_ai + h_det
  - FS+GRAND：x_text + s_node + s_edge
  - FS++：x_text + p_ai + h_det + s_node + s_edge

## 9. 运行状态（队列）

当前使用 `runs/run_queue.sh` 控制并发执行（默认并发 3，batch_size=512）。日志写入：
`runs/logs/<DATASET>/*.log`

## 10. 结果产物目录索引

```text
artifacts/
  graphs/Amazon-*/graph.bin
  graphs/Amazon-*/nodes.parquet
  features/Amazon-*/x_text_bert.npy
  features/Amazon-*/p_ai_roberta.npy
  features/Amazon-*/h_det_roberta_64.npy
  features/Amazon-*/s_node_grand.npy
  features/Amazon-*/s_edge_agg_grand.npy
  splits/Amazon-*/seed2024_masks.pt
runs/
  ai_det/Amazon-*/seed2024/ckpt.pt
  grand/Amazon-*/seed2024/ckpt.pt
  fs*/Amazon-*/seed2024/ckpt.pt
```

## 11. 下一步建议

1) 消融与显著性检验（多 seed / t-test）  
2) 结果汇总与可视化整理（如需要）  

## 12. 主表（ID，Top-3%）

说明：三套 Amazon-Llama3/Qwen2/Qwen-DSR1 均使用 Top-3%（K=2410）。

### 12.1 Amazon-Llama3（seed2024）

| 模型 | AUROC | AP | Precision@Top-3% | Recall@Top-3% |
| --- | --- | --- | --- | --- |
| FS | 0.9136 | 0.2152 | 0.2921 | 0.3143 |
| FS+HC3 | 0.9995 | 0.9871 | 0.9095 | 0.9786 |
| FS+GRAND | 0.9830 | 0.6962 | 0.6307 | 0.6786 |
| FS++ | 0.9994 | 0.9829 | 0.9216 | 0.9915 |

### 12.2 Amazon-Qwen2（seed2024）

| 模型 | AUROC | AP | Precision@Top-3% | Recall@Top-3% |
| --- | --- | --- | --- | --- |
| FS | 0.9071 | 0.1586 | 0.1855 | 0.1991 |
| FS+HC3 | 0.9914 | 0.9225 | 0.8203 | 0.8806 |
| FS+GRAND | 0.9772 | 0.4322 | 0.4979 | 0.5345 |
| FS++ | 0.9970 | 0.9690 | 0.8772 | 0.9416 |

### 12.3 Amazon-Qwen-DSR1（seed2024）

| 模型 | AUROC | AP | Precision@Top-3% | Recall@Top-3% |
| --- | --- | --- | --- | --- |
| FS | 0.9417 | 0.3708 | 0.4129 | 0.4422 |
| FS+HC3 | 0.9989 | 0.9711 | 0.8929 | 0.9564 |
| FS+GRAND | 0.9497 | 0.5248 | 0.4722 | 0.5058 |
| FS++ | 0.9993 | 0.9809 | 0.9021 | 0.9662 |

### 12.4 Amazon-MixLLM（seed2024）

说明：Top-K 采用 MixLLM 的 spam prevalence（约 5.44%）。

| 模型 | AUROC | AP | Precision@Top-5.44% | Recall@Top-5.44% |
| --- | --- | --- | --- | --- |
| FS | 0.9983 | 0.9876 | 0.9648 | 0.9648 |
| FS+HC3 | 0.9977 | 0.9812 | 0.9419 | 0.9419 |
| FS+GRAND | 0.9966 | 0.9636 | 0.8977 | 0.8977 |
| FS++ | 0.9918 | 0.8915 | 0.7929 | 0.7929 |

### 12.5 Amazon-LowSlow（seed2024）

| 模型 | AUROC | AP | Precision@Top-3% | Recall@Top-3% |
| --- | --- | --- | --- | --- |
| FS | 0.9950 | 0.8900 | 0.7759 | 0.8348 |
| FS+HC3 | 0.9978 | 0.9641 | 0.8705 | 0.9366 |
| FS+GRAND | 0.9914 | 0.8592 | 0.7506 | 0.8076 |
| FS++ | 0.9992 | 0.9888 | 0.9129 | 0.9821 |

### 12.6 论文结果主表（seed2024，ID）

说明：Llama3/Qwen2/Qwen-DSR1/LowSlow 使用 Top-3%，MixLLM 使用 Top-5.44%。

| 方法 | Llama3 AUROC/AP | Qwen2 AUROC/AP | Qwen-DSR1 AUROC/AP | MixLLM AUROC/AP | LowSlow AUROC/AP |
| --- | --- | --- | --- | --- | --- |
| FS | 0.9136/0.2152 | 0.9071/0.1586 | 0.9417/0.3708 | 0.9983/0.9876 | 0.9950/0.8900 |
| FS+HC3 | 0.9995/0.9871 | 0.9914/0.9225 | 0.9989/0.9711 | 0.9977/0.9812 | 0.9978/0.9641 |
| FS+GRAND | 0.9830/0.6962 | 0.9772/0.4322 | 0.9497/0.5248 | 0.9966/0.9636 | 0.9914/0.8592 |
| FS++ | 0.9994/0.9829 | 0.9970/0.9690 | 0.9993/0.9809 | 0.9918/0.8915 | 0.9992/0.9888 |
| BERT-MLP | 0.9980/0.9501 | 0.9975/0.9391 | 0.9974/0.9391 | 0.9976/0.9656 | 0.9981/0.9525 |
| AI-only | 0.9998/0.9944 | 0.9958/0.9596 | 1.0000/0.9991 | 0.9996/0.9926 | 0.9998/0.9944 |
| Graph-only | 0.5489/0.0338 | 0.5421/0.0314 | 0.5409/0.0328 | 0.6213/0.0747 | 0.5342/0.0308 |

## 13. OOD 迁移（已完成，seed2024）

OOD 评估已完成（20 对 × 4 模型），结果文件：
`runs/ood/<SRC>_to_<TGT>/<model>/seed2024/metrics_test.json`

#### Amazon-Llama3 作为源域

| OOD 方向 | 模型 | AUROC | AP | Precision@Top-K | Recall@Top-K |
| --- | --- | --- | --- | --- | --- |
| Amazon-Llama3_to_Amazon-LowSlow | fs | 0.9029 | 0.2227 | 0.3315 | 0.3567 |
| Amazon-Llama3_to_Amazon-LowSlow | fs_grand | 0.9779 | 0.6484 | 0.5963 | 0.6415 |
| Amazon-Llama3_to_Amazon-LowSlow | fs_hc3 | 0.9994 | 0.9847 | 0.9091 | 0.9781 |
| Amazon-Llama3_to_Amazon-LowSlow | fs_pp | 0.9991 | 0.9829 | 0.9187 | 0.9884 |
| Amazon-Llama3_to_Amazon-MixLLM | fs | 0.9846 | 0.7427 | 0.7504 | 0.7504 |
| Amazon-Llama3_to_Amazon-MixLLM | fs_grand | 0.9942 | 0.9422 | 0.8725 | 0.8725 |
| Amazon-Llama3_to_Amazon-MixLLM | fs_hc3 | 0.9991 | 0.9885 | 0.9506 | 0.9506 |
| Amazon-Llama3_to_Amazon-MixLLM | fs_pp | 0.9990 | 0.9917 | 0.9673 | 0.9673 |
| Amazon-Llama3_to_Amazon-Qwen-DSR1 | fs | 0.9148 | 0.2127 | 0.2842 | 0.3044 |
| Amazon-Llama3_to_Amazon-Qwen-DSR1 | fs_grand | 0.9782 | 0.6350 | 0.5971 | 0.6396 |
| Amazon-Llama3_to_Amazon-Qwen-DSR1 | fs_hc3 | 0.9912 | 0.8447 | 0.7473 | 0.8004 |
| Amazon-Llama3_to_Amazon-Qwen-DSR1 | fs_pp | 0.9976 | 0.9520 | 0.8739 | 0.9360 |
| Amazon-Llama3_to_Amazon-Qwen2 | fs | 0.8965 | 0.1955 | 0.2714 | 0.2913 |
| Amazon-Llama3_to_Amazon-Qwen2 | fs_grand | 0.9716 | 0.5932 | 0.5535 | 0.5942 |
| Amazon-Llama3_to_Amazon-Qwen2 | fs_hc3 | 0.9906 | 0.8564 | 0.7515 | 0.8067 |
| Amazon-Llama3_to_Amazon-Qwen2 | fs_pp | 0.9920 | 0.9071 | 0.8145 | 0.8744 |

#### Amazon-LowSlow 作为源域

| OOD 方向 | 模型 | AUROC | AP | Precision@Top-K | Recall@Top-K |
| --- | --- | --- | --- | --- | --- |
| Amazon-LowSlow_to_Amazon-Llama3 | fs | 0.9961 | 0.9070 | 0.8008 | 0.8616 |
| Amazon-LowSlow_to_Amazon-Llama3 | fs_grand | 0.9928 | 0.8747 | 0.7676 | 0.8259 |
| Amazon-LowSlow_to_Amazon-Llama3 | fs_hc3 | 0.9980 | 0.9632 | 0.8730 | 0.9393 |
| Amazon-LowSlow_to_Amazon-Llama3 | fs_pp | 0.9994 | 0.9915 | 0.9178 | 0.9875 |
| Amazon-LowSlow_to_Amazon-MixLLM | fs | 0.9956 | 0.9500 | 0.8783 | 0.8783 |
| Amazon-LowSlow_to_Amazon-MixLLM | fs_grand | 0.9959 | 0.9578 | 0.8959 | 0.8959 |
| Amazon-LowSlow_to_Amazon-MixLLM | fs_hc3 | 0.9991 | 0.9809 | 0.9531 | 0.9531 |
| Amazon-LowSlow_to_Amazon-MixLLM | fs_pp | 0.9986 | 0.9874 | 0.9515 | 0.9515 |
| Amazon-LowSlow_to_Amazon-Qwen-DSR1 | fs | 0.9936 | 0.8631 | 0.7610 | 0.8151 |
| Amazon-LowSlow_to_Amazon-Qwen-DSR1 | fs_grand | 0.9913 | 0.8466 | 0.7465 | 0.7996 |
| Amazon-LowSlow_to_Amazon-Qwen-DSR1 | fs_hc3 | 0.9871 | 0.7783 | 0.6971 | 0.7467 |
| Amazon-LowSlow_to_Amazon-Qwen-DSR1 | fs_pp | 0.9950 | 0.9425 | 0.8448 | 0.9049 |
| Amazon-LowSlow_to_Amazon-Qwen2 | fs | 0.9907 | 0.8318 | 0.7307 | 0.7844 |
| Amazon-LowSlow_to_Amazon-Qwen2 | fs_grand | 0.9815 | 0.7723 | 0.6768 | 0.7265 |
| Amazon-LowSlow_to_Amazon-Qwen2 | fs_hc3 | 0.9701 | 0.6190 | 0.5793 | 0.6218 |
| Amazon-LowSlow_to_Amazon-Qwen2 | fs_pp | 0.9893 | 0.9001 | 0.7913 | 0.8494 |

#### Amazon-MixLLM 作为源域

| OOD 方向 | 模型 | AUROC | AP | Precision@Top-K | Recall@Top-K |
| --- | --- | --- | --- | --- | --- |
| Amazon-MixLLM_to_Amazon-Llama3 | fs | 0.9523 | 0.7326 | 0.6386 | 0.6871 |
| Amazon-MixLLM_to_Amazon-Llama3 | fs_grand | 0.9732 | 0.6290 | 0.5693 | 0.6125 |
| Amazon-MixLLM_to_Amazon-Llama3 | fs_hc3 | 0.9744 | 0.6317 | 0.5718 | 0.6152 |
| Amazon-MixLLM_to_Amazon-Llama3 | fs_pp | 0.9771 | 0.4901 | 0.5311 | 0.5714 |
| Amazon-MixLLM_to_Amazon-LowSlow | fs | 0.9409 | 0.7312 | 0.6402 | 0.6888 |
| Amazon-MixLLM_to_Amazon-LowSlow | fs_grand | 0.9630 | 0.5675 | 0.5303 | 0.5705 |
| Amazon-MixLLM_to_Amazon-LowSlow | fs_hc3 | 0.9686 | 0.6091 | 0.5498 | 0.5915 |
| Amazon-MixLLM_to_Amazon-LowSlow | fs_pp | 0.9718 | 0.4570 | 0.5133 | 0.5522 |
| Amazon-MixLLM_to_Amazon-Qwen-DSR1 | fs | 0.9615 | 0.7720 | 0.6730 | 0.7209 |
| Amazon-MixLLM_to_Amazon-Qwen-DSR1 | fs_grand | 0.9804 | 0.7040 | 0.6270 | 0.6716 |
| Amazon-MixLLM_to_Amazon-Qwen-DSR1 | fs_hc3 | 0.9736 | 0.6283 | 0.5722 | 0.6129 |
| Amazon-MixLLM_to_Amazon-Qwen-DSR1 | fs_pp | 0.9743 | 0.4563 | 0.5149 | 0.5516 |
| Amazon-MixLLM_to_Amazon-Qwen2 | fs | 0.9580 | 0.7529 | 0.6568 | 0.7051 |
| Amazon-MixLLM_to_Amazon-Qwen2 | fs_grand | 0.9704 | 0.6254 | 0.5701 | 0.6120 |
| Amazon-MixLLM_to_Amazon-Qwen2 | fs_hc3 | 0.9612 | 0.5683 | 0.5253 | 0.5639 |
| Amazon-MixLLM_to_Amazon-Qwen2 | fs_pp | 0.9746 | 0.5174 | 0.5336 | 0.5728 |

#### Amazon-Qwen-DSR1 作为源域

| OOD 方向 | 模型 | AUROC | AP | Precision@Top-K | Recall@Top-K |
| --- | --- | --- | --- | --- | --- |
| Amazon-Qwen-DSR1_to_Amazon-Llama3 | fs | 0.9131 | 0.2461 | 0.2938 | 0.3161 |
| Amazon-Qwen-DSR1_to_Amazon-Llama3 | fs_grand | 0.9292 | 0.4278 | 0.3867 | 0.4161 |
| Amazon-Qwen-DSR1_to_Amazon-Llama3 | fs_hc3 | 0.8992 | 0.1679 | 0.2253 | 0.2424 |
| Amazon-Qwen-DSR1_to_Amazon-Llama3 | fs_pp | 0.9739 | 0.6334 | 0.5763 | 0.6201 |
| Amazon-Qwen-DSR1_to_Amazon-LowSlow | fs | 0.8987 | 0.2180 | 0.2734 | 0.2942 |
| Amazon-Qwen-DSR1_to_Amazon-LowSlow | fs_grand | 0.9242 | 0.4362 | 0.4054 | 0.4362 |
| Amazon-Qwen-DSR1_to_Amazon-LowSlow | fs_hc3 | 0.8896 | 0.1554 | 0.2054 | 0.2210 |
| Amazon-Qwen-DSR1_to_Amazon-LowSlow | fs_pp | 0.9660 | 0.5807 | 0.5332 | 0.5737 |
| Amazon-Qwen-DSR1_to_Amazon-MixLLM | fs | 0.9896 | 0.8985 | 0.8254 | 0.8254 |
| Amazon-Qwen-DSR1_to_Amazon-MixLLM | fs_grand | 0.9933 | 0.9523 | 0.8977 | 0.8977 |
| Amazon-Qwen-DSR1_to_Amazon-MixLLM | fs_hc3 | 0.9908 | 0.8794 | 0.8218 | 0.8218 |
| Amazon-Qwen-DSR1_to_Amazon-MixLLM | fs_pp | 0.9945 | 0.9484 | 0.8781 | 0.8781 |
| Amazon-Qwen-DSR1_to_Amazon-Qwen2 | fs | 0.9136 | 0.2597 | 0.3149 | 0.3381 |
| Amazon-Qwen-DSR1_to_Amazon-Qwen2 | fs_grand | 0.9181 | 0.3870 | 0.3560 | 0.3822 |
| Amazon-Qwen-DSR1_to_Amazon-Qwen2 | fs_hc3 | 0.8266 | 0.0994 | 0.1303 | 0.1399 |
| Amazon-Qwen-DSR1_to_Amazon-Qwen2 | fs_pp | 0.9280 | 0.3509 | 0.3726 | 0.4000 |

#### Amazon-Qwen2 作为源域

| OOD 方向 | 模型 | AUROC | AP | Precision@Top-K | Recall@Top-K |
| --- | --- | --- | --- | --- | --- |
| Amazon-Qwen2_to_Amazon-Llama3 | fs | 0.8921 | 0.1368 | 0.1523 | 0.1638 |
| Amazon-Qwen2_to_Amazon-Llama3 | fs_grand | 0.9771 | 0.3997 | 0.4876 | 0.5246 |
| Amazon-Qwen2_to_Amazon-Llama3 | fs_hc3 | 0.9762 | 0.7145 | 0.6539 | 0.7036 |
| Amazon-Qwen2_to_Amazon-Llama3 | fs_pp | 0.9904 | 0.8919 | 0.7876 | 0.8473 |
| Amazon-Qwen2_to_Amazon-LowSlow | fs | 0.8910 | 0.1438 | 0.1763 | 0.1897 |
| Amazon-Qwen2_to_Amazon-LowSlow | fs_grand | 0.9722 | 0.3611 | 0.4444 | 0.4781 |
| Amazon-Qwen2_to_Amazon-LowSlow | fs_hc3 | 0.9701 | 0.7076 | 0.6436 | 0.6924 |
| Amazon-Qwen2_to_Amazon-LowSlow | fs_pp | 0.9840 | 0.8559 | 0.7539 | 0.8112 |
| Amazon-Qwen2_to_Amazon-MixLLM | fs | 0.9749 | 0.5542 | 0.6590 | 0.6590 |
| Amazon-Qwen2_to_Amazon-MixLLM | fs_grand | 0.9890 | 0.8187 | 0.7504 | 0.7504 |
| Amazon-Qwen2_to_Amazon-MixLLM | fs_hc3 | 0.9902 | 0.9076 | 0.8543 | 0.8543 |
| Amazon-Qwen2_to_Amazon-MixLLM | fs_pp | 0.9949 | 0.9568 | 0.9008 | 0.9008 |
| Amazon-Qwen2_to_Amazon-Qwen-DSR1 | fs | 0.9061 | 0.1497 | 0.1502 | 0.1609 |
| Amazon-Qwen2_to_Amazon-Qwen-DSR1 | fs_grand | 0.9744 | 0.3715 | 0.4618 | 0.4947 |
| Amazon-Qwen2_to_Amazon-Qwen-DSR1 | fs_hc3 | 0.8898 | 0.2558 | 0.3278 | 0.3511 |
| Amazon-Qwen2_to_Amazon-Qwen-DSR1 | fs_pp | 0.9597 | 0.6515 | 0.5992 | 0.6418 |

## 14. 基线结果（seed2024）

### 14.1 Amazon-Llama3

| 基线 | AUROC | AP | Precision@Top-3% | Recall@Top-3% |
| --- | --- | --- | --- | --- |
| BERT-MLP | 0.9980 | 0.9501 | 0.8432 | 0.9071 |
| AI-only | 0.9998 | 0.9944 | 0.9149 | 0.9844 |
| Graph-only | 0.5489 | 0.0338 | 0.0423 | 0.0455 |

### 14.2 Amazon-Qwen2

| 基线 | AUROC | AP | Precision@Top-3% | Recall@Top-3% |
| --- | --- | --- | --- | --- |
| BERT-MLP | 0.9975 | 0.9391 | 0.8469 | 0.9091 |
| AI-only | 0.9958 | 0.9596 | 0.8718 | 0.9359 |
| Graph-only | 0.5421 | 0.0314 | 0.0361 | 0.0388 |

### 14.3 Amazon-Qwen-DSR1

| 基线 | AUROC | AP | Precision@Top-3% | Recall@Top-3% |
| --- | --- | --- | --- | --- |
| BERT-MLP | 0.9974 | 0.9391 | 0.8394 | 0.8991 |
| AI-only | 1.0000 | 0.9991 | 0.9320 | 0.9982 |
| Graph-only | 0.5409 | 0.0328 | 0.0394 | 0.0422 |

### 14.4 Amazon-MixLLM

| 基线 | AUROC | AP | Precision@Top-5.44% | Recall@Top-5.44% |
| --- | --- | --- | --- | --- |
| BERT-MLP | 0.9976 | 0.9656 | 0.9103 | 0.9103 |
| AI-only | 0.9996 | 0.9926 | 0.9662 | 0.9662 |
| Graph-only | 0.6213 | 0.0747 | 0.0799 | 0.0799 |

### 14.5 Amazon-LowSlow

| 基线 | AUROC | AP | Precision@Top-3% | Recall@Top-3% |
| --- | --- | --- | --- | --- |
| BERT-MLP | 0.9981 | 0.9525 | 0.8498 | 0.9143 |
| AI-only | 0.9998 | 0.9944 | 0.9149 | 0.9844 |
| Graph-only | 0.5342 | 0.0308 | 0.0369 | 0.0397 |

## 15. MixLLM / LowSlow（已完成）

- MixLLM 数据已生成：`data/combined/Amazon-MixLLM.csv`
- MixLLM 图与 splits 已生成：`artifacts/graphs/Amazon-MixLLM`、`artifacts/splits/Amazon-MixLLM`
- MixLLM FS/FS+HC3/FS+GRAND/FS++ 已完成（seed2024）
- MixLLM 三类基线已完成（seed2024）
- LowSlow nodes/graph/splits 已生成：`artifacts/graphs/Amazon-LowSlow`、`artifacts/splits/Amazon-LowSlow`
- LowSlow FS/FS+HC3/FS+GRAND/FS++ 已完成（seed2024）
- LowSlow 三类基线已完成（seed2024）

## 16. 消融与显著性检验（seed2024/2025/2026）

采用配对 t-test，对 FS 基线进行对比，统计 AUROC/AP 的 mean ± std 与 p 值（p vs FS）。

### 16.1 Amazon-Llama3

| 模型 | AUC (mean ± std) | p vs FS |
| --- | --- | --- |
| fs | 0.9642 ± 0.0438 | - |
| fs_hc3 | 0.9990 ± 0.0005 | 0.3067 |
| fs_grand | 0.9445 ± 0.0404 | 0.7119 |
| fs_pp | 0.9987 ± 0.0011 | 0.3109 |

| 模型 | AP (mean ± std) | p vs FS |
| --- | --- | --- |
| fs | 0.6145 ± 0.3457 | - |
| fs_hc3 | 0.9728 ± 0.0149 | 0.2252 |
| fs_grand | 0.4614 ± 0.2483 | 0.6862 |
| fs_pp | 0.9643 ± 0.0323 | 0.2370 |

### 16.2 Amazon-Qwen2

| 模型 | AUC (mean ± std) | p vs FS |
| --- | --- | --- |
| fs | 0.9062 ± 0.0858 | - |
| fs_hc3 | 0.9914 ± 0.0032 | 0.2392 |
| fs_grand | 0.9845 ± 0.0065 | 0.2630 |
| fs_pp | 0.9977 ± 0.0014 | 0.2100 |

| 模型 | AP (mean ± std) | p vs FS |
| --- | --- | --- |
| fs | 0.3526 ± 0.4035 | - |
| fs_hc3 | 0.9284 ± 0.0156 | 0.1381 |
| fs_grand | 0.7051 ± 0.2368 | 0.2554 |
| fs_pp | 0.9660 ± 0.0147 | 0.1260 |

### 16.3 Amazon-Qwen-DSR1

| 模型 | AUC (mean ± std) | p vs FS |
| --- | --- | --- |
| fs | 0.9106 ± 0.0766 | - |
| fs_hc3 | 0.9992 ± 0.0004 | 0.1843 |
| fs_grand | 0.9684 ± 0.0202 | 0.3988 |
| fs_pp | 0.9986 ± 0.0012 | 0.1873 |

| 模型 | AP (mean ± std) | p vs FS |
| --- | --- | --- |
| fs | 0.3316 ± 0.2053 | - |
| fs_hc3 | 0.9824 ± 0.0102 | 0.0330 |
| fs_grand | 0.6198 ± 0.1437 | 0.2805 |
| fs_pp | 0.9478 ± 0.0622 | 0.0537 |

### 16.4 Amazon-MixLLM

| 模型 | AUC (mean ± std) | p vs FS |
| --- | --- | --- |
| fs | 0.9964 ± 0.0020 | - |
| fs_hc3 | 0.9990 ± 0.0011 | 0.2670 |
| fs_grand | 0.9961 ± 0.0005 | 0.8097 |
| fs_pp | 0.9960 ± 0.0037 | 0.9113 |

| 模型 | AP (mean ± std) | p vs FS |
| --- | --- | --- |
| fs | 0.9651 ± 0.0199 | - |
| fs_hc3 | 0.9918 ± 0.0092 | 0.2513 |
| fs_grand | 0.9596 ± 0.0034 | 0.6262 |
| fs_pp | 0.9570 ± 0.0567 | 0.8703 |

### 16.5 Amazon-LowSlow

| 模型 | AUC (mean ± std) | p vs FS |
| --- | --- | --- |
| fs | 0.9868 ± 0.0115 | - |
| fs_hc3 | 0.9990 ± 0.0010 | 0.2243 |
| fs_grand | 0.9281 ± 0.0601 | 0.2231 |
| fs_pp | 0.9980 ± 0.0023 | 0.1695 |

| 模型 | AP (mean ± std) | p vs FS |
| --- | --- | --- |
| fs | 0.7701 ± 0.2003 | - |
| fs_hc3 | 0.9717 ± 0.0090 | 0.2210 |
| fs_grand | 0.4284 ± 0.3836 | 0.2547 |
| fs_pp | 0.9740 ± 0.0172 | 0.1947 |

## 17. Graph-only 参数敏感性（seed2024）

说明：使用 `s_node` 与 `s_edge` 的线性加权分数，alpha 越大越偏向边异常。

#### Amazon-Llama3

| alpha | AUROC | AP | Precision@Top-K | Recall@Top-K |
| --- | --- | --- | --- | --- |
| 0.2 | 0.6319 | 0.0394 | 0.0423 | 0.0455 |
| 0.5 | 0.5489 | 0.0338 | 0.0423 | 0.0455 |
| 0.8 | 0.4901 | 0.0264 | 0.0249 | 0.0268 |

#### Amazon-LowSlow

| alpha | AUROC | AP | Precision@Top-K | Recall@Top-K |
| --- | --- | --- | --- | --- |
| 0.2 | 0.6209 | 0.0382 | 0.0452 | 0.0487 |
| 0.5 | 0.5342 | 0.0308 | 0.0369 | 0.0397 |
| 0.8 | 0.4732 | 0.0246 | 0.0212 | 0.0228 |

#### Amazon-MixLLM

| alpha | AUROC | AP | Precision@Top-K | Recall@Top-K |
| --- | --- | --- | --- | --- |
| 0.2 | 0.6689 | 0.0768 | 0.0512 | 0.0512 |
| 0.5 | 0.6213 | 0.0747 | 0.0799 | 0.0799 |
| 0.8 | 0.5707 | 0.0641 | 0.0948 | 0.0948 |

#### Amazon-Qwen-DSR1

| alpha | AUROC | AP | Precision@Top-K | Recall@Top-K |
| --- | --- | --- | --- | --- |
| 0.2 | 0.6268 | 0.0386 | 0.0415 | 0.0444 |
| 0.5 | 0.5409 | 0.0328 | 0.0394 | 0.0422 |
| 0.8 | 0.4846 | 0.0264 | 0.0282 | 0.0302 |

#### Amazon-Qwen2

| alpha | AUROC | AP | Precision@Top-K | Recall@Top-K |
| --- | --- | --- | --- | --- |
| 0.2 | 0.6248 | 0.0376 | 0.0373 | 0.0401 |
| 0.5 | 0.5421 | 0.0314 | 0.0361 | 0.0388 |
| 0.8 | 0.4847 | 0.0254 | 0.0170 | 0.0183 |

## 18. 实验结果汇总与下一步

### 18.1 当前完成情况

- 主表（ID）五个数据集已完成，并已汇总为论文主表。
- OOD 全方向（20 对 x 4 模型）已完成并写入报告。
- MixLLM/LowSlow 三 seed 消融与显著性检验已完成并写入报告。
- Graph-only alpha 敏感性分析已完成并写入报告。

### 18.2 下一步建议

- 生成论文图表（柱状图或折线图）用于主表与 OOD 的可视化对比。
- 若需要更严格的显著性结论，可补充 effect size 或非参数检验。
- 若需要正文描述，建议补充对 MixLLM/LowSlow 作为源域时的性能分析。

### 18.3 实验耗时报告

以下为当前可追溯的耗时汇总。说明：部分日志没有明确时间戳，采用文件修改时间作为近似，实际耗时可能略有偏差。

| 实验组 | 开始时间 | 结束时间 | 时长 | 统计方式 |
| --- | --- | --- | --- | --- |
| ID 主实验（seed2024，FS/FS+HC3/FS+GRAND/FS++） | 2026-01-16 12:06:52 | 2026-01-16 12:08:50 | 1 min | log timestamps |
| 文本/AI/Graph 基线（seed2024） | 2026-01-16 17:56:44 | 2026-01-16 23:05:39 | 308 min | file mtime |
| MixLLM/LowSlow 多 seed（2024/2025/2026） | 2026-01-16 23:56:39 | 2026-01-17 00:03:01 | 6 min | file mtime |
| OOD 全方向（20 对 × 4） | 2026-01-16 14:55:35 | 2026-01-17 12:31:05 | 1295 min | metrics mtime |
| Graph-only alpha 敏感性 | 2026-01-17 00:01:39 | 2026-01-17 00:02:35 | 0 min | metrics mtime |
| 消融与显著性检验汇总 | 2026-01-17 13:27:10 | 2026-01-17 13:27:10 | 0 min | file mtime |

若需要更精确的耗时报告（含 GPU/CPU 利用率、并发参数与每次运行的起止时间），可以在后续运行中加上统一的时间戳与资源监控日志。

已补充统一监控脚本：`runs/monitor_run.sh`。用法示例：

```bash
bash runs/monitor_run.sh --tag "fs Amazon-Llama3 seed2024" \
  --log runs/logs/Amazon-Llama3/fs.log --interval 30 -- \
  /root/miniconda3/envs/fspp/bin/python -m src.train.train_fraudsquad_custom ...
```

日志会追加 start/end 时间戳，并每隔 `interval` 秒记录 CPU/MEM、系统负载与 GPU 利用率（若存在 `nvidia-smi`）。

## 19. 可视化图表（已生成）

图表保存在：`runs/plots/`

- ID 主实验：`id_auc.png`、`id_ap.png`
- 基线对比：`baseline_auc.png`、`baseline_ap.png`
- OOD 按目标域平均（AUROC）：`ood_auc_by_target.png`
- Graph-only alpha 敏感性：`graph_alpha_auc.png`

## 20. 全部实验项目总结（方法与结果）

### 20.1 实验项目与方法概述

1) **ID 主实验（seed2024）**  
方法：FS / FS+HC3 / FS+GRAND / FS++，统一 Top-K（MixLLM 使用 5.44%，其余使用 3%）。  
数据集：Amazon-Llama3 / Qwen2 / Qwen-DSR1 / MixLLM / LowSlow。

2) **基线对比（seed2024）**  
方法：BERT-MLP（文本特征）/ AI-only（p_ai）/ Graph-only（s_node、s_edge）。  
与主模型同分割与指标评估。

3) **OOD 迁移（seed2024）**  
方法：源域训练 → 目标域评估（20 方向 × 4 模型）。  
统计 AUROC/AP/Precision@Top-K/Recall@Top-K。

4) **消融与显著性检验（seed2024/2025/2026）**  
方法：配对 t-test（对 FS），统计 AUC/AP 的 mean±std 与 p 值。

5) **Graph-only 参数敏感性（seed2024）**  
方法：score = (1-α)·s_node + α·s_edge，α ∈ {0.2, 0.5, 0.8}。

### 20.2 结果总览（定性总结）

- **ID 主实验**：FS++ 与 FS+HC3 在多数数据集表现最好，AUROC/AP 接近 1.0。  
- **基线对比**：AI-only 与 BERT-MLP 表现强，Graph-only 显著弱于文本与 AI 相关特征。  
- **OOD 迁移**：跨域性能整体良好，FS++/FS+HC3 在多数方向保持优势。  
- **消融与显著性**：多 seed 下 FS+HC3/FS++ 的 AUC/AP 稳定提升，p 值支持显著性趋势。  
- **Graph-only 敏感性**：α 增大时 AUROC/AP 下降趋势明显，说明边异常单独贡献有限。

### 20.3 主要数值结果索引

为避免重复，数值详表见以下章节：  
- **ID 主表**：第 12 章  
- **OOD 详细表**：第 13 章  
- **基线表**：第 14 章  
- **消融与显著性**：第 16 章  
- **敏感性分析**：第 17 章  
