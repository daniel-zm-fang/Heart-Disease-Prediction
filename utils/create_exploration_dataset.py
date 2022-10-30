import pandas as pd

# This script is used to generate the data for the exploration page

def race_reverse_one_hot(row):
    if row['Race_1.0']:
        return 'White'
    elif row['Race_2.0']:
        return 'Black'
    elif row['Race_3.0']:
        return 'Asian'
    elif row['Race_4.0']:
        return 'American Indian/Alaskan Native'
    elif row['Race_5.0']:
        return 'Hispanic'
    else:
        return 'Other Race'

if __name__ == '__main__':
    data = pd.read_csv('data/LLCP_agg_cleaned.csv', low_memory=False)
    # sample 10% of the data
    data = data.sample(frac=0.1, random_state=42)
    data['Race'] = data.apply(lambda row: race_reverse_one_hot(row), axis=1)

    data.to_csv('data/LLCP_agg_cleaned_10_percent.csv', index=False)