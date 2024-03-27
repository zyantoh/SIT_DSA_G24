import pandas as pd
import random

# Read CSV file
data = pd.read_csv("airports.csv")

# Convert column data to list
iata = data['iata'].tolist()
icao = data['icao'].tolist()

# Read new CSV into DataFrame
df = pd.read_csv("planes_co2_price.csv")

for i in range(1, 15):
    try:
        new_list1 = []
        new_list2 = []

        # Generate 1000 Random samples, find specific array index
        for j in random.sample(range(1, len(iata)),1200):
            new_list1.append(iata[j])
            new_list2.append(icao[j])
        new_df = pd.DataFrame({'iata': [new_list1], 'icao': [new_list2]})
        new_df.to_csv("data.csv",mode="a", index=False,header=False)
    except ValueError:
        print("error")