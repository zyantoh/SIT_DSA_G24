import heapq
import csv
import plotly.graph_objects as go
import numpy as np

# Define the Airport and Route classes
class Airport:
    def __init__(self, airport_id, name, city, country, iata, icao, latitude, longitude, altitude, timezone, type, source):
        self.airport_id = airport_id
        self.name = name
        self.city = city
        self.country = country
        self.iata = iata
        self.icao = icao
        self.latitude = float(latitude)
        self.longitude = float(longitude)
        self.altitude = int(altitude)
        self.timezone = timezone
        self.type = type
        self.source = source

class Route:
    def __init__(self, airline, airline_id, source_airport_id, dest_airport_id, stops, equipment):
        self.airline = airline
        self.airline_id = airline_id
        self.source_airport_id = source_airport_id
        self.dest_airport_id = dest_airport_id
        self.stops = int(stops)
        self.equipment = equipment

# Define the Graph class
class Graph:
    def __init__(self):
        self.airports = {}  # Key: Airport ID, Value: Airport object
        self.adjacency_list = {}  # Key: Airport ID, Value: List of destination Airport IDs
    
    def add_airport(self, airport):
        self.airports[airport.airport_id] = airport
        if airport.airport_id not in self.adjacency_list:
            self.adjacency_list[airport.airport_id] = []
    
    def add_route(self, route):
        if route.source_airport_id in self.airports and route.dest_airport_id in self.airports:
            self.adjacency_list[route.source_airport_id].append(route.dest_airport_id)

# Function to load airports and routes from the dataset
def load_data(graph, airports_filename, routes_filename):
    with open(airports_filename, 'r', encoding='utf-8') as airports_file:
        csv_reader = csv.DictReader(airports_file)
        for row in csv_reader:
            graph.add_airport(Airport(
                airport_id=row['airportid'],
                name=row['name'],
                city=row['city'],
                country=row['country'],
                iata=row['iata'],
                icao=row['icao'],
                latitude=row['latitude'],
                longitude=row['longitude'],
                altitude=row['altitude'],
                timezone=row['timezone'],
                type=row['type'],
                source=row['source']
            ))
            
    with open(routes_filename, 'r', encoding='utf-8') as routes_file:
        csv_reader = csv.DictReader(routes_file)
        for row in csv_reader:
            graph.add_route(Route(
                airline=row['airline'],
                airline_id=row['airline_ID'],
                source_airport_id=row['source_Airport_ID'],
                dest_airport_id=row['destination_Airport_ID'],
                stops=row['stops'],
                equipment=row['equipment']
            ))

# Dijkstra's algorithm to find multiple paths
def find_multiple_routes(graph, start_id, end_id, num_routes=3):
    def dijkstra_with_exclusions(excluded_paths):
        distances = {airport_id: float('infinity') for airport_id in graph.airports}
        previous = {airport_id: None for airport_id in graph.airports}
        distances[start_id] = 0
        pq = [(0, start_id)]

        while pq:
            current_distance, current_vertex = heapq.heappop(pq)
            if current_vertex == end_id:
                break

            for neighbor in graph.adjacency_list[current_vertex]:
                if [current_vertex, neighbor] in excluded_paths:
                    continue

                distance = current_distance + 1
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    previous[neighbor] = current_vertex
                    heapq.heappush(pq, (distance, neighbor))

        path, current_vertex = [], end_id
        while current_vertex is not None:
            path.insert(0, current_vertex)
            current_vertex = previous[current_vertex]

        return path if path[0] == start_id else []

    routes = []
    excluded_paths = []

    for _ in range(num_routes):
        path = dijkstra_with_exclusions(excluded_paths)
        if not path or path in routes:
            break
        routes.append(path)
        for i in range(len(path) - 1):
            excluded_paths.append([path[i], path[i+1]])

    return routes

def plot_routes(graph, routes):
    fig = go.Figure()

    for route_index, route in enumerate(routes, start=1):
        # Add route segments
        for i in range(len(route) - 1):
            start_airport = graph.airports[route[i]]
            end_airport = graph.airports[route[i+1]]
            fig.add_trace(go.Scattergeo(
                lon = [start_airport.longitude, end_airport.longitude],
                lat = [start_airport.latitude, end_airport.latitude],
                mode = 'lines+markers',
                name = f'Route {route_index}',
                line = dict(width = 2, color = f'rgb({255//route_index}, {55*route_index}, {50*route_index})', dash='dash'),
                marker = dict(size = 4, color = 'blue'),
                text = [start_airport.iata, end_airport.iata],
                hoverinfo = 'text'
            ))

    # Customize the layout of the map
    fig.update_geos(
        projection_type = 'orthographic',
        showland = True,
        landcolor = 'rgb(243, 243, 243)',
        countrycolor = 'rgb(204, 204, 204)'
    )
    fig.update_layout(
        title = 'Flight Routes',
        showlegend = True,
        geo = dict(
            scope = 'world',
            showland = True,
            landcolor = 'rgb(243, 243, 243)',
            countrycolor = 'rgb(204, 204, 204)',
            showcountries = True,
        )
    )

    fig.show()


def find_route_cli(graph):
    start_iata = input("Enter start airport IATA code: ").upper()
    end_iata = input("Enter destination airport IATA code: ").upper()

    start_id = next((id for id, airport in graph.airports.items() if airport.iata == start_iata), None)
    end_id = next((id for id, airport in graph.airports.items() if airport.iata == end_iata), None)

    if not start_id or not end_id:
        print("\nInvalid airport IATA code.")
        return

    num_routes = int(input("Enter number of routes to find: "))
    routes = find_multiple_routes(graph, start_id, end_id, num_routes)
    
    if not routes:
        print("\nNo routes found.")
    else:
        for i, route in enumerate(routes, start=1):
            print(f"\nRoute {i} found:")
            for airport_id in route:
                airport = graph.airports[airport_id]
                print(f"{airport.name} ({airport.iata})")

        plot_routes(graph, routes)
        
graph = Graph()
load_data(graph, 'airports.csv', 'routes.csv')

if __name__ == "__main__":
    find_route_cli(graph)
