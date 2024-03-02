# imports here #
import csv
import networkx as nx
import matplotlib.pyplot as plt

# data structures here #
class FlightMap:
    def __init__(self):
        self.graph = nx.Graph()

    def add_flight(self, origin, destination, distance):
        self.graph.add_edge(origin, destination, distance=distance)

    def find_shortest_path(self, origin, destination):
        try:
            path = nx.shortest_path(self.graph, origin, destination, weight='distance')
            distance = nx.shortest_path_length(self.graph, origin, destination, weight='distance')
            return path, distance
        except nx.NetworkXNoPath:
            return None, float('inf')

    def display_route(self, path, distance):
        if path:
            print("Shortest Route:", path)
            print("Total Distance:", distance)
        else:
            print("No route found.")

    def visualize_flight_map(self):
        pos = nx.spring_layout(self.graph)
        nx.draw(self.graph, pos, with_labels=True, node_size=700)
        edge_labels = nx.get_edge_attributes(self.graph, 'distance')
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels)
        plt.show()

# Function to load data from .dat.text files
def load_data(filename):
    data = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            data.append(row)
    return data

# Load data files
countries_data = load_data('countries.dat.txt')
planes_data = load_data('planes.dat.txt')
routes_data = load_data('routes.dat.txt')
airport_data = load_data('airports.dat.txt')

# Create a map of airport codes to names from countries_data
airport_codes = {code: name for name, code, _ in countries_data}

# Now let's add the routes to the flight map
flight_map = FlightMap()

for route in routes_data:
    # Here, directly use airport codes as origin and destination
    _, _, origin_code, destination_code, _, _, _, _, equipment = route
    # Using a placeholder for distance, replace with actual logic if available
    distance = 100  
    flight_map.add_flight(origin_code, destination_code, distance)

# Use actual airport codes for origin and destination from airport_data
# Example: Set these based on your needs and data
origin = 'AAL'  # Replace with an actual airport code from airport_data
destination = 'AAR'  # Replace with another actual airport code from airport_data

shortest_path, distance = flight_map.find_shortest_path(origin, destination)
flight_map.display_route(shortest_path, distance)

# Visualize the flight map
flight_map.visualize_flight_map()
