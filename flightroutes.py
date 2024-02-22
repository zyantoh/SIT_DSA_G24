# imports here #
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

# main program starts here (user interface needed) #
flight_map = FlightMap()

    # Example data
flight_map.add_flight('A', 'B', 100)
flight_map.add_flight('B', 'C', 200)
flight_map.add_flight('A', 'C', 300)
flight_map.add_flight('C', 'D', 400)
flight_map.add_flight('D', 'E', 500)

origin = 'A'
destination = 'D'

# Find and display shortest path
shortest_path, distance = flight_map.find_shortest_path(origin, destination)
flight_map.display_route(shortest_path, distance)

# Visualize flight map
flight_map.visualize_flight_map()

# take note of the dataset text files as well #