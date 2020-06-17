import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('AB_NYC_2019.csv')
desc = df.describe()
print(desc)
##show all of the host owner's price and reviews
sns.lmplot('price','number_of_reviews',data = df,fit_reg=False)
plt.show()
# sns.lmplot('minimum_nights','availability_365',data = df,fit_reg=False)
# plt.show()
df_alina = pd.read_csv('AB_NYC_2019.csv')
hostname = df_alina.groupby('host_name')
alina = hostname.get_group('Alina')
##show the host owner Alina's lists and how to impact the price with reviews
sns.lmplot('price','number_of_reviews',data = alina,fit_reg=False)
plt.show()
