# Flight Route Planner

## Introduction

This Flight Route Planner is a Python program designed to help users find and visualize multiple flight routes based on various parameters such as distance, cost, and CO2 emissions. It utilizes data on airports, routes, and CO2 emissions from aircraft to provide comprehensive insights into flight planning.

## Prerequisites

Before running this program, ensure you have the following prerequisites installed on your system:

* Python 3.8 or later
* Pip (Python package installer)

## Installation
1. Clone the Repository
First, clone this repository to your local machine using Git.

        git clone <repository-url>
        cd flight-route-planner

2. Set Up a Virtual Environment (Optional but recommended)
It's a good practice to create a virtual environment for your Python projects. This keeps dependencies required by different projects separate by creating isolated environments for them.

* Create a virtual environment:

        python -m venv venv

* Activate the virtual environment:
  * On Windows:

        .\venv\Scripts\activate

  * On macOS and Linux:

        source venv/bin/activate

3. Install Dependencies
Install all required packages using pip:

        pip install -r requirements.txt

This command reads the requirements.txt file and installs all the Python packages listed there. For this program, you'll need to ensure you have dash, plotly, and other dependencies installed.

## Data Preparation
Ensure you have the necessary CSV files (airports.csv, routes.csv, planes.csv) in the project directory. These files should contain data about airports, flight routes, and CO2 emissions related to different aircraft, respectively.

## Running the Program
1. Start the Application
With all dependencies installed, you can now run the Dash app:

        python a_star_flightroutes.py

2. Access the Dashboard
Once the server starts, you'll see a message in the console indicating the URL where the Dash app is running (typically http://127.0.0.1:8050/). Open this URL in a web browser to interact with the Flight Route Planner dashboard.

## Usage
* Use the dropdown menus to select the start and destination airports, exclude any airports if necessary, and choose a plane.
* Click the "Find Routes" button to display possible routes.
* The map will update to show the routes. Hover over the routes for more details.
* Use the "Sort routes by" dropdown to organize the displayed routes according to distance, cost, or environmental impact.

## Support
If you encounter any issues or have questions about running the program, please open an issue in the repository, and we'll do our best to address it.
