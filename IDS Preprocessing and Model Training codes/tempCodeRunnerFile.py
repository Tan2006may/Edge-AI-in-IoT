import pandas as pd

# Load dataset
df = pd.read_csv('cybersecurity_intrusion_data.csv')

# 1. Feature types
print(df.dtypes)

# 2. Summarize categorical columns
cat_cols = ['session_id', 'protocol_type', 'encryption_used', 'browser_type']
for col in cat_cols:
    print(f"{col} unique values: {df[col].unique()}")

# 3. Target/Class balance
print('Attack detected balance:', df['attack_detected'].value_counts())

# 4. Numeric summary
print(df.describe())

# 5. Frequency of failed logins
print('Failed logins distribution:', df['failed_logins'].value_counts())

# 6. Unusual time of access analysis
print('Unusual time access:', df['unusual_time_access'].value_counts())

# 7. Check for repeated session IDs
print('Duplicate session IDs:', df['session_id'].duplicated().sum())

# 8. Example encoding (for model input)
encoded_protocol = pd.get_dummies(df['protocol_type'])
print(encoded_protocol.head())
