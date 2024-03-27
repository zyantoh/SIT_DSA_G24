import csv

# Assuming 'airports.csv' is your CSV file and it's in the current directory
filename = 'airports.csv'

with open(filename, mode='r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    for row in csv_reader:
        try:
            # Replace '\\N' with '0' for the altitude field
            altitude = row['altitude'].replace('\\N', '0')
            # Now convert altitude to float
            altitude = float(altitude)
            
            # Proceed with processing the row...
            # For example, print the airport name and its altitude
            print(f"{row['city']} - Altitude: {altitude}")
        except ValueError as e:
            print(f"Skipping airport due to parsing error: {row['id']},\"{row['name']}\", error: {e}")
