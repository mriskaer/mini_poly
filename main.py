from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import operator
from sklearn.metrics import mean_squared_error, r2_score
from DataClean import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from datetime import datetime

# Read csv files no paths pwease! Just keep data in seperate data folder
vacdata = pd.read_csv(r'C:\Users\lucas\Desktop\Data science & visualization\mini_poly\data\country_vaccinations.csv')
popdata = pd.read_csv(r'C:\Users\lucas\Desktop\Data science & visualization\mini_poly\data\population_by_country_2020.csv')

#cleanup

# Rename columns in population dataset
popdata_new = popdata.rename(columns={'Country (or dependency)': 'country', 'Population (2020)': 'population'}, inplace=False)

clean_data_vac = DataClean(vacdata)
clean_data_pop = DataClean(popdata_new)

# Drops Items in dropList
dropListVac = ['iso_code', 'total_vaccinations', 'people_vaccinated', 'daily_vaccinations_raw',
               'daily_vaccinations', 'total_vaccinations_per_hundred', 'people_vaccinated_per_hundred',
               'people_fully_vaccinated_per_hundred', 'daily_vaccinations_per_million', 'vaccines',
               'source_name', 'source_website']
dropListPop = ['Yearly Change', 'Net Change', 'Density (P/Km²)', 'Land Area (Km²)', 'Migrants (net)',
               'Fert. Rate', 'Med. Age', 'Urban Pop %', 'World Share']

clean_data_vac.removeCols(dropListVac)
clean_data_pop.removeCols(dropListPop)

# Group data
people_fully_vaccinated = vacdata.groupby(by=['country'], sort=False, as_index=False)['people_fully_vaccinated'].max()

def interpolate_country(df, country):

    firs = df.loc[df['country'] == country, 'people_fully_vaccinated'].index[0]
    col = df.columns.get_loc('people_fully_vaccinated')
    df.iloc[firs, col] = 0
    specific_col = 'people_fully_vaccinated'
    return df.loc[vacdata['country'] == country, specific_col].interpolate(limit_direction='both', limit=df.shape[0])

for country in vacdata['country'].unique():
    vacdata.loc[vacdata['country'] == country, 'people_fully_vaccinated'] = interpolate_country(vacdata, country)


# merge datasets
mergedata = pd.merge(vacdata, popdata_new)

#newDate = [datetime.date(mergedata['date'].min()), datetime.date(mergedata['data'].max())]
#print(newDate)
spec_country = mergedata[mergedata.country == 'Denmark']
spec_country['x'] = np.arange(len(spec_country))

x = spec_country['x']
y = spec_country['people_fully_vaccinated']
print(mergedata.head(5))

# transforming the data to include another axis
x = x[:, np.newaxis]
y = y[:, np.newaxis]

#print(spec_country['x'].max)

polynomial_features= PolynomialFeatures(degree=2)
x_poly = polynomial_features.fit_transform(x)

model = LinearRegression()
model.fit(x_poly, y)
y_poly_pred = model.predict(x_poly)

rmse = np.sqrt(mean_squared_error(y,y_poly_pred))
r2 = r2_score(y,y_poly_pred)
print(rmse)
print(r2)

plt.scatter(x, y, s=10)
# sort the values of x before line plot
sort_axis = operator.itemgetter(0)
sorted_zip = sorted(zip(x,y_poly_pred), key=sort_axis)
x, y_poly_pred = zip(*sorted_zip)
plt.plot(x, y_poly_pred, color='m')
plt.show()
