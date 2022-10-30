import pandas as pd

df = pd.read_sas('LLCP2017.XPT')
df.to_csv('data/LLCP2017.csv', index=False)
