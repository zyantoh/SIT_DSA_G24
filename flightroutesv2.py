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
        if route.source_airport_id in self.airports and route.dest_airport_id in self.airports:  # Ensure both airports exist
            self.adjacency_list[route.source_airport_id].append(route.dest_airport_id)

# Function to load airports from the dataset
def load_airports(graph, filename):
    with open(filename, 'r', encoding='utf-8') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            try:
                # Convert latitude, longitude, and altitude to appropriate types
                latitude = float(row['latitude'])
                longitude = float(row['longitude'])
                altitude = int(row['altitude'])
            except ValueError as e:
                print(f"Skipping airport due to parsing error: {row['airportid']},\"{row['name']}\", error: {e}")
                continue
            
            airport = Airport(
                airport_id=row['airportid'],
                name=row['name'],
                city=row['city'],
                country=row['country'],
                iata=row['iata'],
                icao=row['icao'],
                latitude=latitude,
                longitude=longitude,
                altitude=altitude,
                timezone=row['timezone'],
                type=row['type'],
                source=row['source']
            )
            graph.add_airport(airport)

# Function to load routes from the dataset
def load_routes(graph, filename):
    with open(filename, 'r', encoding='utf-8') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            route = Route(
                airline=row['airline'],
                airline_id=row['airline_ID'],
                source_airport_id=row['source_Airport_ID'],
                dest_airport_id=row['destination_Airport_ID'],
                stops=int(row['stops']),
                equipment=row['equipment']
            )
            graph.add_route(route)


# Simplified Dijkstra's algorithm
def dijkstra(graph, start_id, end_id):
    distances = {airport_id: float('infinity') for airport_id in graph.airports}
    previous = {airport_id: None for airport_id in graph.airports}
    distances[start_id] = 0
    pq = [(0, start_id)]
    
    while pq:
        current_distance, current_vertex = heapq.heappop(pq)
        
        if current_vertex == end_id:
            break
        
        for neighbor in graph.adjacency_list[current_vertex]:
            distance = current_distance + 1
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous[neighbor] = current_vertex
                heapq.heappush(pq, (distance, neighbor))
    
    path, current_vertex = [], end_id
    while current_vertex is not None:
        path.append(current_vertex)
        current_vertex = previous[current_vertex]
    path.reverse()
    
    return path if path[0] == start_id else []

def validate_coordinates(lon, lat):
    return np.isfinite(lon) and np.isfinite(lat) and -180 <= lon <= 180 and -90 <= lat <= 90

def plot_great_circle(start_lon, start_lat, end_lon, end_lat):
    fig = go.Figure()

    # Add great circle route using the Scattergeo plot type
    fig.add_trace(go.Scattergeo(
        lon = [start_lat, end_lat],
        lat = [start_lon, end_lon],
        mode = 'lines+markers',
        line = dict(width = 2, color = 'red', dash='dash'),
        geo = 'geo'
    ))
    
    # Customize the layout of the map
    fig.update_layout(
        showlegend = False,
        geo = dict(
            projection_type = 'orthographic',
            showland = True,
            landcolor = 'rgb(243, 243, 243)',
            countrycolor = 'rgb(204, 204, 204)',
        ),
    )

    print("START LON    : ", start_lon)
    print("END LON      : ", end_lon)
    print("START LAT    : ", start_lat)
    print("END LAT      : ", end_lat)
    fig.show()

def find_route_cli():
    start_iata = input("Enter start airport IATA code: ").upper()
    end_iata = input("Enter destination airport IATA code: ").upper()

    start_id = next((id for id, airport in graph.airports.items() if airport.iata == start_iata), None)
    end_id = next((id for id, airport in graph.airports.items() if airport.iata == end_iata), None)

    if not start_id or not end_id:
        print("\nInvalid airport IATA code.")
        return

    path = dijkstra(graph, start_id, end_id)
    if not path:
        print("\nNo route found.")
    else:
        print("\nRoute found:")
        for airport_id in path:
            airport = graph.airports[airport_id]
            print(f"{airport.name} ({airport.iata})")
            
        # Plot the route on the map
        start_airport = graph.airports[path[0]]
        end_airport = graph.airports[path[-1]]
        plot_great_circle(start_airport.latitude, start_airport.longitude, end_airport.latitude, end_airport.longitude)

graph = Graph()
load_airports(graph, 'airports.csv')
load_routes(graph, 'routes.csv')

if __name__ == "__main__":
    find_route_cli()
