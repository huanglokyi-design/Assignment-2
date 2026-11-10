import re
from pathlib import Path
import joblib
import pandas as pd
import jieba
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["PingFang SC", "Arial Unicode MS"]  # macOS 中文字型

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "output"
OUT.mkdir(exist_ok=True)

# 1. 讀入小說，切成約 500 字、只在句尾（。！？）斷開的段落
text = (BASE / "corpora/7_台北人.txt").read_text(encoding="utf-8")
text = re.sub(r"\s+", "", text)  # 去掉空白與換行，方便計字

sentences = re.split(r"(?<=[。！？])", text)

passages, buf = [], ""
for s in sentences:
    if not s:
        continue
    if buf and len(buf) + len(s) > 500:
        passages.append(buf)
        buf = s
    else:
        buf += s
if buf:
    passages.append(buf)

# 2. jieba 分詞 + 模型打分：score = P(positive) - P(negative)
bundle = joblib.load(OUT / "sentiment_model.joblib")
vec, clf = bundle["vectorizer"], bundle["model"]

def seg(s):
    return " ".join(jieba.cut(s))

X = vec.transform([seg(p) for p in passages])
proba = clf.predict_proba(X)
scores = proba[:, list(clf.classes_).index(1)] - proba[:, list(clf.classes_).index(0)]

# 3. 存 CSV
df = pd.DataFrame({
    "passage_number": range(1, len(passages) + 1),
    "score": scores.round(4),
    "text": passages,
})
df.to_csv(OUT / "sentiment_arc.csv", index=False, encoding="utf-8")

# 4. 畫情節弧線：原始分數 + 10 段移動平均 + 0 水平線
ma = df["score"].rolling(10, min_periods=1).mean()

plt.figure(figsize=(12, 5))
plt.plot(df["passage_number"], df["score"], alpha=0.35, label="score")
plt.plot(df["passage_number"], ma, linewidth=2.5, color="crimson", label="moving avg (10)")
plt.axhline(0, color="gray", linestyle="--", linewidth=1)
plt.xlabel("Passage number")
plt.ylabel("Sentiment score (P(pos) − P(neg))")
plt.title("Sentiment arc — 台北人 (白先勇)")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "sentiment_arc.png", dpi=200)
plt.close()

print("passages:", len(passages))
print("avg chars per passage:", round(df["text"].str.len().mean(), 1))

top3 = df.nlargest(3, "score")
bot3 = df.nsmallest(3, "score")
print("\n=== 最高分 3 段 ===")
for _, r in top3.iterrows():
    print(f"#{r.passage_number}  score={r.score:+.4f}  {r.text[:60]}...")
print("\n=== 最低分 3 段 ===")
for _, r in bot3.iterrows():
    print(f"#{r.passage_number}  score={r.score:+.4f}  {r.text[:60]}...")
