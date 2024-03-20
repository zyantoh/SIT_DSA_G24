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
        # Ensure this matches the CSV header
        self.destinationAirportID = destinationAirportID
        self.codeshare = codeshare
        self.stops = int(stops)
        self.equipment = equipment


class Graph:
    def __init__(self):
        self.airports = {}
        self.adjacency_list = {}

    def add_airport(self, airport):
        # Add airportid (key): airport data (val) in dictionary of airports
        self.airports[airport.airportid] = airport
        # give airport a adjacency_list. mapped in airportid:[]
        if airport.airportid not in self.adjacency_list:
            self.adjacency_list[airport.airportid] = []

    def add_route(self, route):
        if route.sourceAirportID in self.airports and route.destinationAirportID in self.airports:
            self.adjacency_list[route.sourceAirportID].append(
                route.destinationAirportID)

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
    # Haversine formula to calculate distance between two points on the surface of the sphere
    # Calculate the great circle distance in kilometers between two points on the earth specified in decimal degrees.
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of Earth in kilometers
    return c * r


def estimate_cost(distance, cost_per_km=0.1):
    # Estimate the cost of a flight given the distance.
    return distance * cost_per_km

# dist_start_to_end for a* algo.
# potential = heuristic estimate of distance (air distance)
def potential(graph, start_id, end_id):
    return haversine(graph.airports[start_id].latitude,
                     graph.airports[start_id].longitude,
                     graph.airports[end_id].latitude,
                     graph.airports[end_id].longitude)


# Dijkstra's algorithm with weighted edge (A* algorithm) to find multiple paths
# A* formula : f(n) = g(n) + h(n)
def find_multiple_routes(graph, start_id, end_id, num_routes=5, cost_per_km=0.1):
    def a_star_with_exclusions(start_id, end_id, excluded_paths):

        # map of vertices starting with infinity value
        distances = {airport_id: float('infinity')
                     for airport_id in graph.airports}

        # map of vertices that have been visited before
        previous = {airport_id: None for airport_id in graph.airports}

        # set starting distance to 0
        distances[start_id] = 0

        # (current distance, current vertex)
        pq = [(0, start_id)]

        # potential distance between start to end
        original_potential = potential(graph, start_id, end_id)

        while pq:
            # first loop: current_distance = 0, current_vertex = start_id
            current_distance, current_vertex = heapq.heappop(pq)
            # potential of current vertex (distance from vertex to end)
            current_vertex_potential = potential(graph, current_vertex, end_id)

            # destination reached
            if current_vertex == end_id:
                break
            # reach out to all nearby nodes
            for neighbor in graph.adjacency_list[current_vertex]:
                # number_of_adj = len(graph.adjacency_list[current_vertex])

                # potential weight from neighbour to end
                neighbour_vertex_potential = potential(graph, neighbor, end_id)
                # weight of edge between current vertex and neighbouring vertex
                # h(n)
                weight_of_edge = original_potential + \
                    neighbour_vertex_potential - current_vertex_potential

                if [current_vertex, neighbor] in excluded_paths:
                    continue

                # distance is the distance of the neighbouring vertex
                # (OLD CODE) distance = current_distance + 1
                # g(n) = distance from start to current vertex + distance from current vertex to neighbouring vertex
                # distance from current vertex to neighbour vertex
                distance = current_distance + \
                    potential(graph, current_vertex, neighbor)
                weight_of_vertex = distance + weight_of_edge
                # updates neighbouring vertex distance if it took a shorter distance
                if weight_of_vertex < distances[neighbor]:
                    # closest distance from start
                    distances[neighbor] = weight_of_vertex
                    # previous vertex path taken
                    previous[neighbor] = current_vertex
                    heapq.heappush(pq, (weight_of_vertex, neighbor))
                    # f(n) : weight of each vertex

                    # boundary_vertices map stores weight of edge

        path, current_vertex = [], end_id
        while current_vertex is not None:
            path.insert(0, current_vertex)
            current_vertex = previous[current_vertex]  # !!!ERROR!!!

        return path if path[0] == start_id else []
        # ---end of a_star_with_exclusions function---

    # ---------------------------------------------------------
    routes_info = []
    excluded_paths = []

    # search route based on the number of times
    for _ in range(num_routes):
        path = a_star_with_exclusions(start_id, end_id, excluded_paths)
        if not path or any(path == route_info['route'] for route_info in routes_info):
            break  # Stop if no path found or if the path is already included

        # Calculate the total distance for the route
        total_distance = sum(haversine(
            graph.airports[path[i]
                           ].latitude, graph.airports[path[i]].longitude,
            graph.airports[path[i+1]
                           ].latitude, graph.airports[path[i+1]].longitude
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
    colors = ['red', 'blue', 'green', 'purple',
              'orange', 'yellow', 'pink', 'cyan']

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
            # None to create separate lines
            latitudes.extend(
                [start_airport.latitude, end_airport.latitude, None])
            longitudes.extend(
                [start_airport.longitude, end_airport.longitude, None])
            hover_texts.extend([f"{start_airport.name} ({start_airport.iata})",
                                f"{end_airport.name} ({end_airport.iata})", None])

        # Get the IATA codes for the original start and final destination
        start_iata = graph.airports[route[0]].iata
        end_iata = graph.airports[route[-1]].iata

        start_lon = graph.airports[route[0]].longitude
        start_lat = graph.airports[route[0]].latitude
        # Plot the route using a single color
        fig.add_trace(go.Scattergeo(
            lon=longitudes,
            lat=latitudes,
            mode='lines+markers',
            name=f'Route {route_index}',
            line=dict(width=2, color=color),
            marker=dict(size=4, color=color),
            text=hover_texts[:-1],  # Exclude the last None value
            hoverinfo='text',
            # Include the route index and segment index
            customdata=[[route_index]],
        ))

    # Customize the layout of the map
    fig.update_geos(
        projection_type='orthographic',
        showland=True,
        landcolor='rgb(243, 243, 243)',
        countrycolor='rgb(204, 204, 204)',
        projection_rotation=dict(lon=start_lon, lat=start_lat)
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
            showframe=False,
        ),
        clickmode='event+select',
        autosize=True
    )
    return fig


# Define the layout of the app
app.layout = html.Div([
    dcc.Input(id='start-iata', type='text',
              placeholder='Enter start IATA code'),
    dcc.Input(id='end-iata', type='text',
              placeholder='Enter destination IATA code'),
    html.Button('Find Routes', id='find-routes', n_clicks=0),
    dcc.Store(id='stored-routes'),  # Store component for the routes
    # Adjust 'height' as desired
    dcc.Graph(id='flight-map', style={
        'height': '90vh',
    }),
    html.Div(id='flight-info', style={
        'position': 'absolute',
        'right': '0px',  # Adjusts the position to the right
        'top': '30vh',  # Adjusts the position from the top
        'margin-right': '20px',  # Adjusts the margin from the right
    })
])


# Callback to store the routes data
@app.callback(
    Output('stored-routes', 'data'),
    [Input('find-routes', 'n_clicks')],
    [State('start-iata', 'value'), State('end-iata', 'value')]
)
def update_stored_routes(n_clicks, start_iata, end_iata):
    if n_clicks > 0:
        start_id = next((airport.airportid for airport in graph.airports.values(
        ) if airport.iata == start_iata), None)
        end_id = next((airport.airportid for airport in graph.airports.values(
        ) if airport.iata == end_iata), None)
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
    else:
        # Return a blank figure or figure with a default globe setup
        return go.Figure(
            data=[go.Scattergeo()],
            layout=go.Layout(
                title='No routes to display',
                geo=dict(
                    showland=True,
                    landcolor='rgb(243, 243, 243)',
                    countrycolor='rgb(204, 204, 204)',
                    showcountries=True,
                    showsubunits=True,
                    showframe=False,
                    projection=dict(
                        type='orthographic',
                    )
                )
            )
        )


@app.callback(
    Output('flight-info', 'children'),
    [Input('flight-map', 'clickData')],
    [State('stored-routes', 'data')]
)
def display_click_data(clickData, routes_data):
    if clickData:
        points = clickData.get('points', [])
        if points:
            # assuming curveNumber is used to index the routes
            route_index = points[0]['curveNumber']
            # retrieve the selected route information
            route_info = routes_data[route_index]

            # Initialize the list for the route information
            info = [
                html.Strong("Route Information:"),
                html.Br(),
                html.Strong("Start: "),
                f"{graph.airports[route_info['route'][0]].name} ({graph.airports[route_info['route'][0]].iata}), {graph.airports[route_info['route'][0]].country}",
                html.Br(), html.Br()
            ]

            # Add layovers if there are any
            layovers_info = [
                f"{i+1}. {graph.airports[airport_id].name} ({graph.airports[airport_id].iata}), {graph.airports[airport_id].country}"
                for i, airport_id in enumerate(route_info['route'][1:-1])
            ]
            if layovers_info:
                info.append(html.Strong("Layover(s):"))
                info.append(html.Br())
                info.extend([html.Div(layover) for layover in layovers_info])
                info.append(html.Br())

            # Add the end airport information
            info.append(html.Strong("End: "))
            info.append(
                f"{graph.airports[route_info['route'][-1]].name} ({graph.airports[route_info['route'][-1]].iata}), {graph.airports[route_info['route'][-1]].country}"
            )
            info.append(html.Br())
            info.append(html.Br())

            # Add total distance and estimated cost
            info.append(
                html.Div(f"Total Distance: {route_info['distance']:.2f} km"))
            info.append(html.Div(f"Estimated Cost: ${route_info['cost']:.2f}"))

            return html.Div(info, style={'white-space': 'pre-line'})
    return "Click on a route to see detailed information."


if __name__ == '__main__':
    app.run_server(debug=True)
