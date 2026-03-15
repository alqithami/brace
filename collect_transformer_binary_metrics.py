import json, glob, os
import pandas as pd
import subprocess

rows=[]
for path in glob.glob("models/rumor_transformer_binary/**/metrics.json", recursive=True):
    with open(path,"r",encoding="utf-8") as f:
        m=json.load(f)
    rows.append({
        "run_dir": os.path.dirname(path),
        "model": m.get("model"),
        "task": m.get("task"),
        "run_id": m.get("run_id"),
        "dev_macro_f1": m["dev"].get("macro_f1"),
        "dev_auc": m["dev"].get("auc"),
        "dev_acc": m["dev"].get("acc"),
        "test_macro_f1": m["test"].get("macro_f1"),
        "test_auc": m["test"].get("auc"),
        "test_acc": m["test"].get("acc"),
    })

df=pd.DataFrame(rows).sort_values("run_dir")
df.to_csv("transformer_binary_metrics_summary.csv", index=False)
print("Wrote transformer_binary_metrics_summary.csv with", len(df), "rows")

with open("metrics_files.txt","w",encoding="utf-8") as f:
    for p in glob.glob("models/rumor_transformer_binary/**/metrics.json", recursive=True):
        f.write(p + "\n")
subprocess.run(["tar","-czf","transformer_binary_metrics_only.tgz","-T","metrics_files.txt"], check=True)
print("Wrote transformer_binary_metrics_only.tgz")
