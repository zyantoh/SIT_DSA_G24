import heapq
import csv
import numpy as np
from dash import Dash, dcc, html, Input, Output, State
import plotly.graph_objs as go
from math import radians, cos, sin, asin, sqrt

# Define the Airport, Route, and Graph classes
class Airport:
    def __init__(self, airportid, name, city, country, iata, icao, latitude, longitude, altitude, timezone, dst, tz_database_timezone, type, source):
        self.airportid = airportid  # Ensure this matches the CSV header for airport ID
        self.name = name
        self.city = city
        self.country = country
        self.iata = iata
        self.icao = icao
        self.latitude = float(latitude)
        self.longitude = float(longitude)
        self.altitude = int(altitude)
        self.timezone = timezone
        self.dst = dst
        self.tz_database_timezone = tz_database_timezone
        self.type = type
        self.source = source

class Route:
    def __init__(self, airline, airlineID, sourceAirport, sourceAirportID, destinationAirport, destinationAirportID, codeshare, stops, equipment):
        self.airline = airline
        self.airlineID = airlineID
        self.sourceAirport = sourceAirport
        self.sourceAirportID = sourceAirportID  # Ensure this matches the CSV header
        self.destinationAirport = destinationAirport
        self.destinationAirportID = destinationAirportID  # Ensure this matches the CSV header
        self.codeshare = codeshare
        self.stops = int(stops)
        self.equipment = equipment

class Graph:
    def __init__(self):
        self.airports = {}
        self.adjacency_list = {}

    def add_airport(self, airport):
        self.airports[airport.airportid] = airport
        if airport.airportid not in self.adjacency_list:
            self.adjacency_list[airport.airportid] = []

    def add_route(self, route):
        if route.sourceAirportID in self.airports and route.destinationAirportID in self.airports:
            self.adjacency_list[route.sourceAirportID].append(route.destinationAirportID)

# Load data function
def load_data(graph, airports_filename, routes_filename):
    with open(airports_filename, 'r', encoding='utf-8') as airports_file:
        csv_reader = csv.DictReader(airports_file)
        for row in csv_reader:
            airport = Airport(**row)
            graph.add_airport(airport)

    with open(routes_filename, 'r', encoding='utf-8') as routes_file:
        csv_reader = csv.DictReader(routes_file)
        for row in csv_reader:
            route = Route(**row)
            graph.add_route(route)

def haversine(lat1, lon1, lat2, lon2):
    # Calculate the great circle distance in kilometers between two points on the earth specified in decimal degrees.
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of Earth in kilometers
    return c * r

def estimate_cost(distance, cost_per_km=0.1):
    # Estimate the cost of a flight given the distance.
    return distance * cost_per_km


# Dijkstra's algorithm to find multiple paths
def find_multiple_routes(graph, start_id, end_id, num_routes=3, cost_per_km=0.1):
    def dijkstra_with_exclusions(start_id, end_id, excluded_paths):
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

    routes_info = []
    excluded_paths = []

    for _ in range(num_routes):
        path = dijkstra_with_exclusions(start_id, end_id, excluded_paths)
        if not path or any(path == route_info['route'] for route_info in routes_info):
            break  # Stop if no path found or if the path is already included

        # Calculate the total distance for the route
        total_distance = sum(haversine(
            graph.airports[path[i]].latitude, graph.airports[path[i]].longitude,
            graph.airports[path[i+1]].latitude, graph.airports[path[i+1]].longitude
        ) for i in range(len(path) - 1))
        
        # Estimate the total cost for the route
        total_cost = estimate_cost(total_distance, cost_per_km)
        
        # Add the path, distance, and cost to the routes_info
        routes_info.append({
            'route': path,
            'distance': total_distance,
            'cost': total_cost
        })
        
        # Add the edges of the path to the excluded_paths to prevent reuse
        for i in range(len(path) - 1):
            excluded_paths.append([path[i], path[i+1]])
            
    return routes_info

# Create the Dash app
app = Dash(__name__)
graph = Graph()
load_data(graph, 'airports.csv', 'routes.csv')

# Function to generate the figure with all routes
def plot_routes(graph, route_infos):
    fig = go.Figure()

    # Define a list of colors for the routes
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'yellow', 'pink', 'cyan']

    for route_index, route_info in enumerate(route_infos, start=1):
        route = route_info['route']
        distance = route_info['distance']
        cost = route_info['cost']
        color = colors[route_index % len(colors)]  # Cycle through colors list

        latitudes = []
        longitudes = []
        hover_texts = []

        # Add route segments
        for i in range(len(route) - 1):
            start_airport = graph.airports[route[i]]
            end_airport = graph.airports[route[i + 1]]
            latitudes.extend([start_airport.latitude, end_airport.latitude, None])  # None to create separate lines
            longitudes.extend([start_airport.longitude, end_airport.longitude, None])
            hover_texts.extend([f"{start_airport.name} ({start_airport.iata})", 
                                f"{end_airport.name} ({end_airport.iata})", None])

        # Get the IATA codes for the original start and final destination
        start_iata = graph.airports[route[0]].iata
        end_iata = graph.airports[route[-1]].iata

        # Plot the route using a single color
        fig.add_trace(go.Scattergeo(
            lon=longitudes,
            lat=latitudes,
            mode='lines+markers',
            name=f'Route{route_index}, Distance: {distance:.2f} km, Cost: ${cost:.2f}',
            line=dict(width=2, color=color),
            marker=dict(size=4, color=color),
            text=hover_texts[:-1],  # Exclude the last None value
            hoverinfo='text',
            customdata=[[route_index]],  # Include the route index and segment index
        ))

    # The center for the orthographic projection is set to the start of the first route
    center_lat = graph.airports[route_infos[0]['route'][0]].latitude if route_infos else 0
    center_lon = graph.airports[route_infos[0]['route'][0]].longitude if route_infos else 0

    # Customize the layout of the map
    fig.update_geos(
        projection_type='natural earth',
        showland=True,
        landcolor='rgb(243, 243, 243)',
        countrycolor='rgb(204, 204, 204)',
    )
    fig.update_layout(
        title='Flight Routes',
        showlegend=True,
        geo=dict(
            scope="world",
            showland=True,
            landcolor='rgb(243, 243, 243)',
            countrycolor='rgb(204, 204, 204)',
            showcountries=True,
            showsubunits=True,
            showframe=False
        ),
        clickmode='event+select',
        autosize=True
    )
    return fig


# Define the layout of the app
app.layout = html.Div([
    dcc.Input(id='start-iata', type='text', placeholder='Enter start IATA code'),
    dcc.Input(id='end-iata', type='text', placeholder='Enter destination IATA code'),
    html.Button('Find Routes', id='find-routes', n_clicks=0),
    dcc.Store(id='stored-routes'),  # Store component for the routes
    dcc.Graph(id='flight-map', style={'height': '80vh'}),  # Adjust 'height' as desired
    html.Div(id='flight-info', style={'margin-left': '20px'})
])


# Callback to store the routes data
@app.callback(
    Output('stored-routes', 'data'),
    [Input('find-routes', 'n_clicks')],
    [State('start-iata', 'value'), State('end-iata', 'value')]
)

def update_stored_routes(n_clicks, start_iata, end_iata):
    if n_clicks > 0:
        start_id = next((airport.airportid for airport in graph.airports.values() if airport.iata == start_iata), None)
        end_id = next((airport.airportid for airport in graph.airports.values() if airport.iata == end_iata), None)
        if start_id and end_id:
            routes = find_multiple_routes(graph, start_id, end_id)
            return routes
    return []

# Callback to update the map based on the routes data stored
@app.callback(
    Output('flight-map', 'figure'),
    [Input('stored-routes', 'data')]
)
def update_map(routes_data):
    if routes_data:
        figure = plot_routes(graph, routes_data)
        return figure
    return go.Figure()

@app.callback(
    Output('flight-info', 'children'),
    [Input('flight-map', 'clickData')],
    [State('stored-routes', 'data')]
)
def display_click_data(clickData, routes_data):
    if clickData:
        points = clickData.get('points', [])
        if points:
            route_index = points[0]['curveNumber']  # assuming curveNumber is used to index the routes
            route_info = routes_data[route_index]  # retrieve the selected route information

            # Generate information about the route
            info = [
                html.Div([
                    html.Strong("Route Information:"),
                    html.Br(),
                    f"Start: {graph.airports[route_info['route'][0]].name} ({graph.airports[route_info['route'][0]].iata})",
                    html.Br()
                ])
            ]
            
            # List all layovers except the first and last points
            layovers = [graph.airports[airport_id].name + f" ({graph.airports[airport_id].iata})"
                        for airport_id in route_info['route'][1:-1]]
            if layovers:
                info.append("Layover(s): " + ", ".join(layovers))
                info.append(html.Br())
            
            info.extend([
                f"End: {graph.airports[route_info['route'][-1]].name} ({graph.airports[route_info['route'][-1]].iata})",
                html.Br(),
                f"Total Distance: {route_info['distance']:.2f} km",
                html.Br(),
                f"Estimated Cost: ${route_info['cost']:.2f}"
            ])

            return info
    return "Click on a route to see detailed information."

if __name__ == '__main__':
    app.run_server(debug=True)
