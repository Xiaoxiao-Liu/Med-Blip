# MedCon 多语言数据

英文源数据与各语言翻译、评测文件均在本目录下按语言分文件夹。

## 目录结构

```
data/medcon/
├── README.md
├── en/                          ← 英文（源语言）
│   ├── reference.json
│   ├── prediction.json
│   ├── final_UMLS_sets.json
│   └── UMLS_stop_words.json
└── {lang}/                      ← zh, fr, de, es, ja, ko, pt, ar
    ├── todo/
    │   ├── reference_translate.json
    │   └── prediction_translate.json
    ├── reference.json           ← 翻译完成后的 reference（与 en 对齐）
    └── prediction.json          ← 翻译完成后的 prediction
```

Demo 任务数据在 `data/demo/`，不在此目录。

## 翻译流程

1. 打开 `data/medcon/{lang}/todo/*_translate.json`，按 `prompt` 翻译 `data` 中的 `content_en`。
2. 将结果保存为 `data/medcon/{lang}/reference.json` 或 `prediction.json`（路径见各文件中的 `target_file`）。
3. 保留 `content_en`，新增 `content_{lang}`；其余字段不变。

## 评测

```bash
bash scripts/run_medcon.sh
```

各语言配置见 `configs/medcon_en.yaml`、`configs/medcon_zh.yaml` 等。
