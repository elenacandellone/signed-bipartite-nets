import pandas as pd
import requests
from tqdm import tqdm

# Create a session object
s = requests.Session()

dataframes = []

for year in tqdm(range(1990, 2024), desc='Year'):
    exceptions = 0
    for bill in tqdm(range(1, 1000), desc='Bill', leave=False):
        url = f"https://clerk.house.gov/Votes/{year}{bill}"

        try:
            table = pd.read_html(url, attrs={"class": "allvotes-table"})[0]
            table.columns = ["Person", "Person", "Party", "State", "State_short", "Vote"]
            table["year"] = year
            table["bill"] = bill
            dataframes.append(table)
        except Exception as e:
            print(f"Error at year {year}, bill {bill}. Url {url}")
            print(e)
            exceptions += 1
            if exceptions > 10:
                break

# Write all data to file
with open("./data/us-house/real-data/clerk_data.tsv", "w+") as f:
    for df in dataframes:
        df.to_csv(f, sep="\t", index=None, header=None)