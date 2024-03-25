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
        self.timezone = timezone
        self.dst = dst
        self.tz_database_timezone = tz_database_timezone
        self.type = type
        self.source = source
        try:
            self.latitude = float(latitude)
            self.longitude = float(longitude)
            self.altitude = int(altitude)
        except ValueError:
            print(f"Invalid data format in airport: {name}, ID: {airportid}")


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

class Co2:
    def __init__(self,name, equipment, icao, co2_emission_per_km, price_per_km):
        self.name = name
        self.equipment = equipment
        self.icao = icao
        self.co2_emission_per_km = float(co2_emission_per_km)
        self.price_per_km = float(price_per_km)
        

class Graph:
    def __init__(self):
        self.airports = {}
        self.co2_data = {}  # New dictionary to store CO2 data
        self.routes = {}  # New dictionary to store Route objects
        self.adjacency_list = {}

    def add_airport(self, airport):
        # Add airportid (key): airport data (val) in dictionary of airports
        self.airports[airport.airportid] = airport
        # give airport a adjacency_list. mapped in airportid:[]
        if airport.airportid not in self.adjacency_list:
            self.adjacency_list[airport.airportid] = []

    def add_route(self, route):
        # Check if airports in the route exist in the graph
        if route.sourceAirportID not in self.airports or route.destinationAirportID not in self.airports:
            return

        # Add the route to the adjacency list
        if route.sourceAirportID not in self.adjacency_list:
            self.adjacency_list[route.sourceAirportID] = []
        self.adjacency_list[route.sourceAirportID].append(route.destinationAirportID)

        # Store detailed route information
        route_key = (route.sourceAirportID, route.destinationAirportID)
        self.routes[route_key] = route


# Load data function


def load_data(graph, airports_filename, routes_filename, co2_filename):
    try:
        with open(airports_filename, 'r', encoding='utf-8') as airports_file:
            csv_reader = csv.DictReader(airports_file)
            for row in csv_reader:
                try:
                    airport = Airport(**row)
                    graph.add_airport(airport)
                except ValueError as error:
                    print(f"Data conversion error in airports data: {error}")
                except Exception as error:
                    print(f"Unexpected error in airports data: {error}")
    except FileNotFoundError:
        print(f"Airport file not found: {airports_filename}")
    except csv.Error as error:
        print(f"Error reading airports CSV file: {error}")
    except Exception as error:
        print(f"Unexpected error when loading airports: {error}")

    # Exception handling for reading route data
    try:
        with open(routes_filename, 'r', encoding='utf-8') as routes_file:
            csv_reader = csv.DictReader(routes_file)
            for row in csv_reader:
                try:
                    route = Route(**row)
                    graph.add_route(route)
                except ValueError as error:
                    print(f"Data conversion error in routes data: {error}")
                except Exception as error:
                    print(f"Unexpected error in routes data: {error}")
    except FileNotFoundError:
        print(f"Route file not found: {routes_filename}")
    except csv.Error as error:
        print(f"Error reading routes CSV file: {error}")
    except Exception as error:
        print(f"Unexpected error when loading routes: {error}")
        
    # Load CO2 data
    try:
        with open(co2_filename, 'r', encoding='utf-8') as co2_file:
            csv_reader = csv.DictReader(co2_file)
            for row in csv_reader:
                try:
                    co2_instance = Co2(**row)
                    graph.co2_data[co2_instance.equipment] = co2_instance
                except ValueError as error:
                    print(f"Data conversion error in CO2 data: {error}")
                except Exception as error:
                    print(f"Unexpected error in CO2 data: {error}")
    except FileNotFoundError:
        print(f"CO2 data file not found: {co2_filename}")
    except csv.Error as error:
        print(f"Error reading CO2 data CSV file: {error}")
    except Exception as error:
        print(f"Unexpected error when loading CO2 data: {error}")

def haversine(lat1, lon1, lat2, lon2):
    # Haversine formula to calculate distance between two points on the surface of the sphere
    # Calculate the great circle distance in kilometers between two points on the earth specified in decimal degrees.
    # Convert decimal degrees to radians
    try:
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        r = 6371  # Radius of Earth in kilometers
        return c * r
    except ValueError as error:
        print(f"Invalid input values for haversine calculation: {error}")
        return 0  # or an appropriate default value
    except Exception as error:
        print(f"Unexpected error in haversine calculation: {error}")
        return 0


def estimate_cost(distance, cost_per_km=0.1):
    # Estimate the cost of a flight given the distance.
    return distance * cost_per_km

def estimate_co2(graph, route, default_co2_emission_per_km=150):
    total_co2 = 0.0
    for i in range(len(route) - 1):
        # Calculate the distance between each pair of airports
        distance = haversine(
            graph.airports[route[i]].latitude, graph.airports[route[i]].longitude,
            graph.airports[route[i+1]].latitude, graph.airports[route[i+1]].longitude
        )

        # Create a unique key for the route segment
        route_key = (route[i], route[i+1])

        # Retrieve the Route object using the route_key
        route_segment = graph.routes.get(route_key)

        # Continue only if route_segment is found
        if route_segment:
            aircraft_type = route_segment.equipment

            # Get the CO2 emissions per km for the given aircraft type
            co2_data = graph.co2_data.get(aircraft_type)
            if co2_data:
                co2_emission_per_km = co2_data.co2_emission_per_km
            else:
                co2_emission_per_km = default_co2_emission_per_km

            # Calculate CO2 emissions for this segment
            total_co2 += distance * co2_emission_per_km

    return total_co2


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
            graph.airports[path[i]].latitude, graph.airports[path[i]].longitude,
            graph.airports[path[i+1]].latitude, graph.airports[path[i+1]].longitude
        ) for i in range(len(path) - 1))

        # Estimate the total cost for the route
        total_cost = estimate_cost(total_distance, cost_per_km)
        #Fix estimate total co2 emission
        total_co2 = estimate_co2(graph, path)  # Use the entire route path and the graph
        # Add the path, distance, and cost, co2 to the routes_info
        routes_info.append({
            'route': path,
            'distance': total_distance,
            'cost': total_cost,
            'environmental impact' : total_co2
        })

        # Add the edges of the path to the excluded_paths to prevent reuse
        for i in range(len(path) - 1):
            excluded_paths.append([path[i], path[i+1]])

    return routes_info


# Create the Dash app
app = Dash(__name__)
graph = Graph()
load_data(graph, 'airports.csv', 'routes.csv', 'planes_co2_price.csv')

# Function to generate the figure with all routes


def plot_routes(graph, route_infos):
    fig = go.Figure()

    # Define a list of colors for the routes
    colors = ['#B22222',  # Firebrick
              '#00008B',  # DarkBlue
              '#006400',  # DarkGreen
              '#4B0082',  # Indigo
              '#FF8C00',  # DarkOrange
              '#8B0000',  # DarkRed
              '#800080',  # Purple
              '#2F4F4F']  # DarkSlateGray

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
    title_text = f"{len(route_infos)} Flight Route{'s' if len(route_infos) != 1 else ''} Found"

    fig.update_layout(
        title=title_text,
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
    html.Div([
        dcc.Input(
            id='start-iata',
            type='text',
            placeholder='Enter start IATA code',
            style={'marginRight': '10px', 'width': '20%', 'height': '36px',
                   'padding': '0 12px'}  # Adjust marginRight as needed
        ),

        dcc.Input(
            id='end-iata',
            type='text',
            placeholder='Enter destination IATA code',
            style={'marginRight': '10px', 'width': '20%', 'height': '36px',
                   'padding': '0 12px'}  # Adjust marginRight as needed
        ),
        html.Button(
            'Find Routes',
            id='find-routes',
            n_clicks=0,
            # Adjust marginRight as needed
            style={'marginRight': '10px', 'height': '36px',
                   'background-color': "rgb(51,117,229)",
                   "color": "white"}
        ),
        html.Label('Sort Routes By: '),
        dcc.Dropdown(
            id='sort-by-dropdown',
            options=[
                {'label': 'Distance', 'value': 'Distance'},
                {'label': 'Cost', 'value': 'Cost'},
                {'label': 'Environmental Impact', 'value': 'Environmental Impact'}
            ],
            value='',  # Set default value to 'Distance'
            placeholder='Sort routes by',
            # Ensure this is the same width as the input boxes
            style={'width': '40%', 'padding-left': '20px'}
        ),
    ], style={'display': 'flex', 'alignItems': 'center'}),
    dcc.Store(id='stored-routes'),  # Store component for the routes

    # flight map
    # Adjust 'height' as desired
    dcc.Graph(id='flight-map', style={'height': '90vh'}),

    # listed routes and instructions beside flight map
    html.Div(id='route-instructions', style={
        'position': 'absolute',
        'right': '0px',  # Adjusts the position to the right
        'top': '30vh',  # Adjusts the position from the top
        'margin-right': '20px',  # Adjusts the margin from the right
    }),

    # flight info right under instructions
    html.Div(id='flight-info', style={
        'hidden': 'true',
        'position': 'absolute',
        'right': '0px',  # Adjusts the position to the right
        'top': '35vh',  # Adjusts the position from the top
        'margin-right': '20px',  # Adjusts the margin from the right
    }),
    html.Div(id='error-message', style={'color': 'red'}),

])


# Callback to store the routes data
@app.callback(
    [Output('stored-routes', 'data'),
     Output('error-message', 'children')],
    [Input('find-routes', 'n_clicks')],
    [State('start-iata', 'value'), State('end-iata', 'value')]
)
def update_stored_routes(n_clicks, start_iata, end_iata):
    if n_clicks > 0:
        if start_iata and end_iata:  # Validate IATA codes are entered
            start_id = next((airport.airportid for airport in graph.airports.values()
                             if airport.iata == start_iata), None)
            end_id = next((airport.airportid for airport in graph.airports.values()
                           if airport.iata == end_iata), None)
            if start_id and end_id:
                routes = find_multiple_routes(graph, start_id, end_id)
                return routes, ''  # No error message
            else:
                # Return a message for invalid IATA codes
                return [], 'Invalid IATA codes entered.'
        else:
            return [], 'Please enter both start and destination IATA codes.'
    return [], ''  # Default return if n_clicks is 0 or less


# Callback to update the map based on the routes data stored
@app.callback(
    [Output('flight-map', 'figure'),
     Output('route-instructions', 'children')],
    [Input('stored-routes', 'data')]
)
def update_map(routes_data):
    print("update_map called")  # Debug print statement
    if routes_data:
        figure = plot_routes(graph, routes_data)
        instructions = "Click on a route to see detailed information."
        return figure, instructions
    else:
        blank_figure = go.Figure(
            data=[go.Scattergeo()],
            layout=go.Layout(
                title='Enter Airport Codes to find routes.',
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
        return blank_figure, ""

# Callback to display information on a route when route is clicked on
@app.callback(
    [Output('flight-info', 'children'),
     Output('route-instructions', 'children', allow_duplicate=True)],
    [Input('flight-map', 'clickData')],
    [State('stored-routes', 'data')],
    prevent_initial_call=True
)
def display_click_data(clickData, routes_data):
    if clickData:
        points = clickData.get('points', [])
        print("clicked")
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

            print("info displayed")
            return html.Div(info, style={'white-space': 'pre-line'}), ""

    return []

# Callback to sort routes based on the factors available (WIP)


@app.callback(
    Output('stored-routes', 'data', allow_duplicate=True),
    Input('sort-by-dropdown', 'value'),
    Input('stored-routes', 'data'),
    prevent_initial_call=True  # dont callback if dropdown is not touched at the start
)
def sort_routes(chosen_value, routes_data):

    # choose sorting factor according to dropdown selection first (Enviro. impact yet to be added)
    if chosen_value == 'Distance':
        sort_factor = 'distance'
    elif chosen_value == 'Cost':
        sort_factor = 'cost'
    elif chosen_value == "Environmental Impact":
        sort_factor = 'environmental impact'
    else:
        return routes_data

    # quicksort: recursively sorts routes based on selected factor
    def quickSort(routes_data):
        size = len(routes_data)
        if size > 1:
            pivotIndex = partition(routes_data, sort_factor)
            routes_data[0:pivotIndex] = quickSort(routes_data[0:pivotIndex])
            routes_data[pivotIndex +
                        1:size] = quickSort(routes_data[pivotIndex+1:size])
        return routes_data

    # helper function for quicksort
    def partition(routes_data, sort_factor):
        pivot = routes_data[0]
        size = len(routes_data)
        pivotIndex = 0
        for index in range(1, size):
            if routes_data[index][sort_factor] < pivot[sort_factor]:
                pivotIndex += 1
                routes_data[pivotIndex], routes_data[index] = \
                    routes_data[index], routes_data[pivotIndex]
        return pivotIndex
    sorted_routes = quickSort(routes_data) 
    #error handling print
    # print("Sorted routes based on {}: {}".format(chosen_value, sorted_routes))
    return sorted_routes
    # return quickSort(routes_data)


# Initiate Dash app and run server
if __name__ == '__main__':
    try:
        app.run_server(debug=True)
    except Exception as e:
        print(f"Failed to start Dash app: {e}")
