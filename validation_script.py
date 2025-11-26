import pandas as pd
import sys

df = pd.read_csv('bns_sections.csv')
print(df.columns.tolist)

required_columns = ["Chapter", "Chapter_name","Chapter_subtype", "Section", "Section _name", "Description"]
missing = [col for col in required_columns if col not in df.columns]

if missing:
    print(f"ERROR: Missing columns: {missing}")
    sys.exit(1)

print(f"SUCCESS: {len(df)} sections found")
print(f"Chapters: {df['Chapter'].nunique()}")
print(f"Sample: {df['Section'].iloc[0]} - {df['Section _name'].iloc[0]}")