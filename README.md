# 中文情感分析：從豆瓣書評到《台北人》情節弧線

以豆瓣書評（1 星 / 5 星）訓練中文情感分類模型，再套用到白先勇短篇小說集《台北人》全文，逐段預測情感分數，畫出全書 14 篇故事的「情節弧線」，並以互動式網頁呈現所有結果。

**互動式分析頁面**：`docs/index.html`（資料已內嵌，可直接用瀏覽器開啟，或部署到 GitHub Pages）

## 專案結構

```
Assignment2/
├── README.md
├── data/
│   ├── train.csv          # 豆瓣書評訓練集（480 筆，正負各 240）
│   └── test.csv           # 測試集（120 筆，正負各 60）
├── corpora/
│   ├── 7_台北人.txt        # 白先勇《台北人》全文（語料來源：中文小说100强）
│   └── douban.json        # 豆瓣書評原始語料（32 MB，未納入 git，僅存本機）
├── scripts/
│   ├── train_sentiment.py     # 訓練 TF-IDF + LogisticRegression 模型
│   ├── sentiment_arc.py       # 對《台北人》逐段打分，輸出情節弧線
│   └── plot_prob_distribution.py  # 畫預測機率分布圖
├── output/
│   ├── sentiment_model.joblib # 訓練好的模型（vectorizer + classifier）
│   ├── metrics.json           # 準確率、混淆矩陣、分類報告
│   ├── test_predictions.csv   # 測試集預測明細（text, label, pred, prob_positive）
│   ├── sentiment_arc.csv      # 《台北人》223 段的情感分數
│   ├── sentiment_arc.png      # 情節弧線圖
│   ├── confusion_matrix.png   # 混淆矩陣圖
│   ├── prob_distribution.png  # 預測機率分布圖
│   ├── error_analysis.csv     # 25 筆錯誤預測（含完整書評與信心）
│   └── error_analysis_full.md # 錯誤分析完整報告
└── docs/
    └── index.html         # 互動式分析頁面（自包含，資料已內嵌）
```

## 方法

### 1. 訓練資料
從豆瓣書評語料（`corpora/douban.json`）抽取 1 星與 5 星書評各 240 筆作為訓練集（負面 = 0、正面 = 1），另抽 120 筆（各 60）作為測試集。書評長度 ≥ 10 字且去重。

### 2. 模型
- **前處理**：jieba 分詞
- **特徵**：TF-IDF（詞級 n-gram）
- **分類器**：LogisticRegression
- 輸出 `prob_positive` = P(正面)，決策閾值 0.5

### 3. 情節弧線
將《台北人》全文依標點切成句子、每約 500 字合併為一段（共 223 段），逐段計算情感分數：

$$\text{score} = P(\text{positive}) - P(\text{negative})$$

分數範圍 −1（極負）到 +1（極正），並對應到 14 篇故事的段落區間（如〈永远的尹雪艳〉第 1–18 段、〈游园惊梦〉第 164–193 段）。

## 結果

### 模型效能（測試集 120 筆）

| 指標 | 數值 |
|------|------|
| 準確率 | **79.2%** |
| Macro F1 | 0.792 |
| Macro Precision | 0.792 |
| Macro Recall | 0.792 |

| 類別 | Precision | Recall | F1 | 支援數 |
|------|-----------|--------|-----|--------|
| 負面（0） | 0.807 | 0.767 | 0.786 | 60 |
| 正面（1） | 0.778 | 0.817 | 0.797 | 60 |

混淆矩陣：負面 46/60 正確、正面 49/60 正確。

### 錯誤分析（25 筆錯誤，20.8%）

- **所有錯誤都緊貼決策邊界**：錯誤預測的 `prob_positive` 全部落在 0.42–0.57，無高信心誤判（見 `output/prob_distribution.png`）
- **主要錯誤模式**：
  1. **引用陷阱**：負評整段引用原書名句（如《目送》「我慢慢地、慢慢地了解到…」），正面詞彙密度高誤導模型
  2. **混合情感**：正評含大量「喜欢」但結尾「好可惜好可惜」，先貶後褒的評論
  3. **訓練集外語言**：英文書評（"full of b.s."）完全失效
  4. **無情感詞的純敘事**：模型只能靠微弱線索猜測

### 情節弧線觀察
《台北人》整體情感偏負面（多數段落 score < 0），符合該書「今昔對比、繁華落盡」的基調；個別故事內部有明顯起伏，如〈金大班的最後一夜〉中段出現正向高峰。

## 如何重現

```bash
# 1. 安裝依賴
pip install pandas scikit-learn jieba matplotlib joblib

# 2. 訓練模型（需要 corpora/douban.json，未隨 repo 提供）
python scripts/train_sentiment.py

# 3. 產生情節弧線
python scripts/sentiment_arc.py

# 4. 畫預測機率分布圖
python scripts/plot_prob_distribution.py
```

訓練完成後 `output/` 會重新生成所有結果檔案；`docs/index.html` 為自包含頁面，直接用瀏覽器開啟即可。

## 互動頁面功能

| 分頁 | 內容 |
|------|------|
| 📊 模型總覽 | KPI 卡、互動式混淆矩陣、各類別 Precision/Recall/F1 長條圖、分類報告表 |
| 🔍 預測探索 | 120 筆測試預測，可依「全部/正確/錯誤」過濾 + 信心門檻滑桿 |
| 〽️ 情節弧線 | 223 段情感折線 + 移動平均，可選單篇故事放大，點擊資料點看原文 |
| 🔤 關鍵詞彙 | 模型係數前 30 個正向/負向詞彙，字級對應權重 |

## 資料來源

- 訓練語料：豆瓣書評（1 星 / 5 星）
- 分析對象：白先勇《台北人》（收錄於「中文小说100强」語料庫）
