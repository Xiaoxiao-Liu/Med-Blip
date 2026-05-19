# 多语言翻译任务

## 目录结构

```
data/translations/
├── README.md          ← 你在看的文件
├── {lang}/            ← 语言代码: zh, fr, de, es, ja, ko, pt, ar
│   ├── vqa/
│   │   ├── todo/      ← 翻译任务文件（含prompt和原始数据）
│   │   │   ├── train_translate.json
│   │   │   └── val_translate.json
│   │   └── done/      ← 翻译完成后，把结果放这里
│   │       ├── train.jsonl    ← 翻译后的训练数据
│   │       └── val.jsonl      ← 翻译后的验证数据
│   └── medcon/
│       ├── todo/      ← 翻译任务文件
│       │   ├── reference_translate.json
│       │   └── prediction_translate.json
│       └── done/      ← 翻译完成后，把结果放这里
│           ├── reference.json
│           └── prediction.json
```

## 语言列表

| 代码 | 语言 |
|------|------|
| zh | 中文（简体中文） |
| fr | 法语（Français） |
| de | 德语（Deutsch） |
| es | 西班牙语（Español） |
| ja | 日语（日本語） |
| ko | 韩语（한국어） |
| pt | 葡萄牙语（Português） |
| ar | 阿拉伯语（العربية） |

## 工作流程

### 第1步：翻译

打开每个 `todo/` 目录下的 `*_translate.json` 文件，里面包含：
- `prompt`: 翻译要求（中文说明）
- `data`: 需要翻译的原始数据
- `target_file`: 翻译结果应保存的路径

把 `prompt` 和 `data` 一起喂给你的AI，让它完成翻译。

### 第2步：保存结果

#### VQA数据
翻译结果保存为 **JSONL格式**（每行一个JSON），放入 `done/` 目录。

格式示例（以中文为例）：
```jsonl
{"image": "data/demo/images/sample_049.png", "question": "显示的是什么形状？", "answer": "圆形"}
{"image": "data/demo/images/sample_018.png", "question": "这个形状是什么颜色的？", "answer": "蓝色"}
```

注意：`image` 字段保持原始英文路径不变，只翻译 `question` 和 `answer`。

#### MedCon数据
翻译结果保存为 **JSON格式**，放入 `done/` 目录。

格式示例（以中文为例）：
```json
[
    {
        "encounter_id": "ex_001",
        "responses": [
            {
                "content_en": "Based on the symptoms described...",
                "content_zh": "根据所描述的症状...",
                "author_id": "doc_01",
                "completeness": 1.0,
                "contains_freq_ans": 1.0
            }
        ]
    }
]
```

注意：保留原始 `content_en` 字段，新增 `content_{lang}` 字段。

### 第3步：翻译完成后

所有 `done/` 目录填好后，回来告诉我，我会：
1. 将所有语言的VQA数据合并，生成多语言训练/验证集
2. 为每种语言创建MedCon评估配置
3. 修改训练和评估流程支持多语言
4. 创建一键运行脚本

## 共需翻译的文件数量

- VQA: 8种语言 × 2个文件（train + val）= **16个文件**
- MedCon: 8种语言 × 2个文件（reference + prediction）= **16个文件**
- **总计: 32个翻译任务文件**
