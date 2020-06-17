import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('AB_NYC_2019.csv')
desc = df.describe()
print(desc)
sns.lmplot('price','number_of_reviews',data = df,fit_reg=False)
plt.show()
sns.lmplot('minimum_nights','availability_365',data = df,fit_reg=False)
plt.show()