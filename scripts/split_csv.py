import pandas as pd
import os

src = '/Users/mjshetty/Downloads/NTSB-AviationData.csv'
out_dir = '/Users/mjshetty/devops-workspace/aws/AI/RAG/data'
os.makedirs(out_dir, exist_ok=True)

df = pd.read_csv(src)
df['EventDate'] = pd.to_datetime(df['EventDate'], errors='coerce')
df['Year'] = df['EventDate'].dt.year

bands = [
    ('2010_2014', 2010, 2014),
    ('2015_2019', 2015, 2019),
    ('2020_2024', 2020, 2024),
    ('2025_2026', 2025, 2026),
]

for label, start, end in bands:
    subset = df[(df['Year'] >= start) & (df['Year'] <= end)].drop(columns=['Year'])
    out_path = os.path.join(out_dir, f'NTSB_{label}.csv')
    subset.to_csv(out_path, index=False)
    print(f'NTSB_{label}.csv  -> {len(subset)} records')
