import os
import glob
import re
import pandas as pd
from collections import Counter

# adjust this path if needed
BASE_PATH = r"Evaluation/cf_methods/Foil_Trees/results"
PATTERN = os.path.join(BASE_PATH, "*_counterfactual_report.csv")

# regex to extract feature names (assumes feature names are word characters or underscores)
feature_regex = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*(?:<=|<|>=|>|==|!=)")

# container: dataset_name → Counter of feature usage
dataset_counters = {}

for filepath in glob.glob(PATTERN):
    fname = os.path.basename(filepath)
    # extract dataset name (prefix before first underscore)
    dataset = fname.split("_")[0]
    df = pd.read_csv(filepath, usecols=["Counterfactual_Rules"])
    
    # ensure a counter exists
    ctr = dataset_counters.setdefault(dataset, Counter())
    
    # iterate all rules and count features
    for rule in df["Counterfactual_Rules"].dropna():
        matches = feature_regex.findall(rule)
        ctr.update(matches)

# now produce one report per dataset
for dataset, counter in dataset_counters.items():
    # convert to DataFrame, sort descending
    report_df = (
        pd.DataFrame.from_records(
            list(counter.items()), columns=["Feature", "Count"]
        )
        .sort_values("Count", ascending=False)
        .reset_index(drop=True)
    )
    
    out_path = os.path.join(BASE_PATH, f"{dataset}_feature_usage.csv")
    report_df.to_csv(out_path, index=False)
    print(f"Saved feature usage report for '{dataset}' → {out_path}")
