from DataClean import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# Read csv files no paths pwease! Just keep data in seperate data folder
vacdata = pd.read_csv('data/country_vaccinations.csv')
popdata = pd.read_csv('data/population_by_country_2020.csv')

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

spec_country = mergedata[mergedata.country == 'Denmark']
spec_country['x'] = np.arange(len(spec_country))

x = spec_country['x']
y = spec_country['people_fully_vaccinated']

model = np.poly1d(np.polyfit(x, y, 2))
line = np.linspace(0, 104, 100) #last value = precision

day = model(30)
prediction = model(spec_country['population'])

#if people_fully_vaccinated == population:
#print(date)

print("Prediction:")
print("At day 30, this many people will be fully vaccinated: ", prediction)
#print("The country has this many citizens: ", spec_country['population'])
print("We know this, with certainty from 0-1: ", r2_score(y, model(x)))

plt.scatter(x, y)
plt.plot(line, model(line))
plt.show()
