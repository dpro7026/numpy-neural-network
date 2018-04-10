import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# import data from csv
df = pd.read_csv('loan.csv', nrows=100)

# print out list of all headers
headers = df.columns.values.tolist()
print(headers)

# print count for values for each 'grade'
print(df['grade'].value_counts())

# CLEAN DATA
# Choose a subset of predictors - target is 'grade' (of loan)
df = df[['loan_amnt', 'term', 'int_rate', 'annual_inc', 'emp_length', 'grade']]
print(df.describe(include='all'))

print(df.dtypes)

# Check what the categroies for 'term' are
print(df['term'].unique())
# Grouping
df_group_term=df[['term','loan_amnt','annual_inc']]
# grouping results
df_group_term=df_group_term.groupby(['term'],as_index= False).mean()
print(df_group_term)

# make a pivot table
grouped_pivot=df_group_term.pivot(index='term',columns='loan_amnt')
print(grouped_pivot)

# create a heatmap
#use the grouped results
plt.pcolor(grouped_pivot, cmap='RdBu')
plt.colorbar()
plt.show()

# Categorical to numerical values
# get the variable names for the dummy variable
# dummy_1 = pd.get_dummies(df["term"])
# print(dummy_1.head())
# dummy_1.rename(columns={'term':'36 months', 'term':'60 months'}, inplace=True)
# print(dummy_1.head())
# # merge data frame "df" and "dummy_1"
# df = pd.concat([df, dummy_1], axis=1)
# # drop original column "fuel-type" from "df"
# df.drop("term", axis = 1, inplace=True)
# print(df.head())
# print(df.dtypes)



# grouping results
# df_gptest=df[['term','loan_amnt','annual_inc']]
# grouped_test1=df_gptest.groupby(['term','body-style'],as_index= False).mean()
# print(grouped_test1)

# make a pivot table
# grouped_pivot=grouped_test1.pivot(index='drive-wheels',columns='body-style')
# print(grouped_pivot)
# # fill in missing values with 0
# grouped_pivot=grouped_pivot.fillna(0) #fill missing values with 0
# print(grouped_pivot)

# create a heatmap
#use the grouped results
# plt.pcolor(grouped_pivot, cmap='RdBu')
# plt.colorbar()
# plt.show()








#
# # Annual income as potential predictor variable of loan grade
# sns.regplot(x="annual_inc", y="loan_amnt", data=df)
# plt.ylim(0,)
# plt.show()
# print(df[["annual_inc", "loan_amnt"]].corr())
#
# # catergorical data - box plots
# sns.boxplot(x="annual_inc", y="grade", data=df)
# plt.show()
#
# # convert grade categories to numerical
# grouped_test2=df[['grade','annual_inc']].groupby(['grade'])
# print(grouped_test2.head(2))
# grouped_test2.get_group('B')['annual_inc']
#
# # possible grade categories
# print(df['grade'].unique())
#
# df_group_one=df[['grade','annual_inc','loan_amnt']]
# # grouping results
# df_group_one=df_group_one.groupby(['grade'],as_index= False).mean()
# print(df_group_one)
#
# # grouping results
# # df_gptest=df[['drive-wheels','body-style','price']]
# # grouped_test1=df_gptest.groupby(['drive-wheels','body-style'],as_index= False).mean()
# # print(grouped_test1)
#
# # make a pivot table
# # grouped_pivot=grouped_test1.pivot(index='drive-wheels',columns='body-style')
# # print(grouped_pivot)
#
#
#
# # visualise the bins with a histogram
#
#
#
# # remove null rows
# # drop rows with missing values
# # df.dropna(axis=1, inplace=True)
# # print(df.describe(include='all'))
#
# # simply drop whole row with NaN in "price" column
# # df.dropna(subset=["price"], axis=0, inplace = True)
#
# # normalise the data
#
# # Z-Score or Standard Score: new=old-mean/standard deviation
# # df['length']=(df['length']-df['length'].mean())/df['length'].std()
#
#
# # def PreprocessDataset():
# #     from sklearn import preprocessing
# #     data = pd.read_csv('Processed/Cleaned_loans_2007.csv',index_col=False,low_memory=False)
# #     data = data.reindex(np.random.permutation(data.index))
# #     cols = data.columns
# #     x_columns = cols.drop("loan_status")
# #
# #     x = data[x_columns]
# #     y = data["loan_status"]
# #
# #     train_max_row = int(data.shape[0]*0.9)
# #
# #     x_train = x.iloc[:train_max_row]
# #     x_test = x.iloc[train_max_row:]
# #
# #     y_train = y.iloc[:train_max_row]
# #     y_test = y.iloc[train_max_row:]
# #
# #     y_train = np_utils.to_categorical(y_train)
# #     y_test = np_utils.to_categorical(y_test)
# #
# #     ################Pre-processing###########
# #     x_train = preprocessing.scale(x_train)
# #     x_test = preprocessing.scale(x_test)
# #
# #     return x_train, x_test, y_train, y_test
# # x_train, x_test, y_train, y_test = PreprocessDataset()
