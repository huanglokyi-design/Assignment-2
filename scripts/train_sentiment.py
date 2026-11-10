import json, random, joblib
from pathlib import Path
import pandas as pd
import jieba
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "output"
OUT.mkdir(exist_ok=True)
random.seed(42)

# 1. 讀豆瓣資料，只留 5 星(1) 和 1 星(0)
with open(BASE / "corpora/douban.json", encoding="utf-8") as f:
    data = json.load(f)

def clean(xs, label):
    seen, out = set(), []
    for d in xs:
        t = d["text"].strip()
        if len(t) >= 10 and t not in seen:
            seen.add(t)
            out.append({"text": t, "label": label})
    return out

pos = clean([d for d in data if d["label"] == 5], 1)
neg = clean([d for d in data if d["label"] == 1], 0)
random.shuffle(pos); random.shuffle(neg)

# 2. 每類抽 300，合併
rows = pos[:300] + neg[:300]
random.shuffle(rows)
df = pd.DataFrame(rows)

# 3. 80/20 分層切分
train, test = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["label"]
)
train.to_csv(BASE / "data/train.csv", index=False, encoding="utf-8")
test.to_csv(BASE / "data/test.csv", index=False, encoding="utf-8")
print("train:", train.shape, "test:", test.shape)
print("train labels:", train["label"].value_counts().to_dict())
print("test labels:", test["label"].value_counts().to_dict())

# 4. jieba 分詞
def seg(s):
    return " ".join(jieba.cut(str(s)))

train["seg"] = train["text"].map(seg)
test["seg"] = test["text"].map(seg)

# 5. TF-IDF，只 fit 訓練集
vec = TfidfVectorizer(max_features=5000)
Xtr = vec.fit_transform(train["seg"])
Xte = vec.transform(test["seg"])

# 6. 訓練邏輯迴歸
clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(Xtr, train["label"])

# 7. 測試集預測與評估
pred = clf.predict(Xte)
proba = clf.predict_proba(Xte)[:, 1]

pd.DataFrame({
    "text": test["text"], "label": test["label"],
    "pred": pred, "prob_positive": proba
}).to_csv(OUT / "test_predictions.csv", index=False, encoding="utf-8")

acc = accuracy_score(test["label"], pred)
cm = confusion_matrix(test["label"], pred, labels=[0, 1])
rep = classification_report(test["label"], pred, labels=[0, 1],
      target_names=["negative(0)", "positive(1)"], output_dict=True)

json.dump({"accuracy": acc, "confusion_matrix": cm.tolist(), "report": rep},
          open(OUT / "metrics.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Pred 0", "Pred 1"],
            yticklabels=["True 0", "True 1"])
plt.tight_layout(); plt.savefig(OUT / "confusion_matrix.png", dpi=200); plt.close()

print("Accuracy:", acc)
print(cm)
print(classification_report(test["label"], pred,
      target_names=["negative(0)", "positive(1)"], zero_division=0))

# 8. 存 vectorizer + model
joblib.dump({"vectorizer": vec, "model": clf},
            OUT / "sentiment_model.joblib")

# 9. 印出最正、最負的 20 個詞（老師提示有要求，報告也可用）
feature_names = vec.get_feature_names_out()
coef = clf.coef_[0]
top_pos = coef.argsort()[-20:][::-1]
top_neg = coef.argsort()[:20]
print("\n最正面 20 詞:")
for i in top_pos:
    print(feature_names[i], round(coef[i], 4))
print("\n最負面 20 詞:")
for i in top_neg:
    print(feature_names[i], round(coef[i], 4))
