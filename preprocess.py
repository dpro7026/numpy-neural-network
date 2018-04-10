import pandas as pd

# import data from csv
df = pd.read_csv('loan.csv', nrows=1000)

# Choose a subset of predictors - target is 'grade' (of loan)
df = df[['loan_amnt', 'int_rate', 'annual_inc', 'grade']]
# get basic stats
print(df.describe(include='all'))
print(df.dtypes)
print (df.head(100))

# drop any rows contain null values
df_no_missing = df.dropna()
print(df_no_missing.describe(include='all'))

# training data
training_data = df[['loan_amnt', 'int_rate', 'annual_inc']]
print(training_data.head())
# training target
training_target = df[['grade']]
print(training_target.head())

# one-hot encoding from categorical to numeric
encoded_training_target = pd.get_dummies(training_target['grade'])
print(encoded_training_target.head())

# write to csv
training_data.to_csv('raw_training_data.csv')
training_target.to_csv('training_target.csv')
encoded_training_target.to_csv('encoded_training_target.csv')

# normalise data to within range (-3,3)
# z-Score: new=old-mean/standard deviation
training_data['loan_amnt']=(training_data['loan_amnt']-training_data['loan_amnt'].mean())/training_data['loan_amnt'].std()
training_data['annual_inc']=(training_data['annual_inc']-training_data['annual_inc'].mean())/training_data['annual_inc'].std()
training_data['int_rate']=(training_data['int_rate']-training_data['int_rate'].mean())/training_data['int_rate'].std()
print(training_data.head())

# write to csv
training_data.to_csv('training_data.csv')
