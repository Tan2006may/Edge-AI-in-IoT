import pandas as pd

# Load your data
df = pd.read_csv('cybersecurity_intrusion_data.csv')

# Correlation of each numeric column with target
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.drop('attack_detected')
correlations = df[numeric_cols].corrwith(df['attack_detected'])
print(correlations)
# Example for a categorical feature
print(df.groupby('protocol_type')['attack_detected'].mean())

# For all object (categorical) columns except session_id
cat_cols = [col for col in df.select_dtypes('object').columns if col != 'session_id']
for col in cat_cols:
    print(df.groupby(col)['attack_detected'].mean())
