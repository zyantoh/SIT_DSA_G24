import pandas as pd
import random

# Read CSV file
data = pd.read_csv("airports.csv")

# Convert column data to list
# iata = data['iata'].tolist()
# icao = data['icao'].tolist()
airport = data['airportid'].tolist()

# Read new CSV into DataFrame
# df = pd.read_csv("planes_co2_price.csv")

for i in range(1, 15):
    try:
        new_list1 = []
        new_list2 = []

        # Generate Random samples, find specific array index
        for j in random.sample(range(1, len(airport)),450):
            new_list1.append(airport[j])
            # new_list2.append(icao[j])
            print(new_list1)
        new_list1 = sorted(new_list1)
        
        # new_list1 = [eval(i) for i in new_list1]
        # new_list1 = new_list1.sort()
        new_df = pd.DataFrame({'airport': [new_list1]})
        new_df.to_csv("data.csv",mode="a", index=False,header=False)
    except ValueError:
        print("error")