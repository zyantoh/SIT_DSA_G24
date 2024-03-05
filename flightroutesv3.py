import heapq
import csv
import numpy as np
from dash import Dash, dcc, html, Input, Output, State
import plotly.graph_objs as go

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

# Create the Dash app
app = Dash(__name__)
graph = Graph()
load_data(graph, 'airports.csv', 'routes.csv')

# Function to generate the figure with all routes
def create_figure(graph, routes):
    fig = go.Figure()

    for route_index, route in enumerate(routes, start=1):
        for i in range(len(route) - 1):
            start_airport = graph.airports[route[i]]
            end_airport = graph.airports[route[i+1]]
            # Use the airport name for hover text instead of the IATA code
            hover_text = [f"{start_airport.name} ({start_airport.iata})", 
                            f"{end_airport.name} ({end_airport.iata})"]
            fig.add_trace(go.Scattergeo(
                lon = [start_airport.longitude, end_airport.longitude],
                lat = [start_airport.latitude, end_airport.latitude],
                mode = 'lines+markers',
                name = f'Route {route_index}',
                line = dict(width = 2),
                marker = dict(size = 4),
                text = hover_text,
                hoverinfo = 'text',
                customdata = [route_index]
            ))

    fig.update_geos(
        projection_type = 'orthographic',
        showland = True,
        landcolor = 'rgb(243, 243, 243)',
        countrycolor = 'rgb(204, 204, 204)',
    ),
    fig.update_layout(
        title = 'Flight Routes',
        showlegend = True,
        geo = dict(
            scope = "world",
            showland = True,
            landcolor='rgb(243, 243, 243)',
            countrycolor='rgb(204, 204, 204)',
            showcountries=True,
            showsubunits=True,
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
    dcc.Graph(id='flight-map', style={'height': '90vh'}),  # Adjust 'height' as desired
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
        figure = create_figure(graph, routes_data)
        return figure
    return go.Figure()

# Callback to display the clicked route information
@app.callback(
    Output('flight-info', 'children'),
    [Input('flight-map', 'clickData')],
    [State('stored-routes', 'data')]  # Use the stored routes data
)
def display_click_data(clickData, routes_data):
    if clickData is not None:
        route_index = clickData['points'][0]['customdata'][0]
        route = routes_data[route_index - 1]  # Adjust index since route_index starts at 1
        info = f"Route {route_index} Information:"
        
        # Generate information about the route
        for i, airport_id in enumerate(route):
            airport = graph.airports[airport_id]
            if i == 0:
                info += f" Start: {airport.name} ({airport.iata})"
            elif i == len(route) - 1:
                info += f" End: {airport.name} ({airport.iata})"
            else:
                info += f" Layover {i}: {airport.name} ({airport.iata})"
        
        return info
    # return "Click on a route to see the flight information."

if __name__ == '__main__':
    app.run_server(debug=True)