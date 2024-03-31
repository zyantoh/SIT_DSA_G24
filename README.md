#Flight Route Planner

##Introduction

This Flight Route Planner is a Python program designed to help users find and visualize multiple flight routes based on various parameters such as distance, cost, and CO2 emissions. It utilizes data on airports, routes, and CO2 emissions from aircraft to provide comprehensive insights into flight planning.

Prerequisites

Before running this program, ensure you have the following prerequisites installed on your system:

Python 3.8 or later
Pip (Python package installer)
Installation

Clone the Repository
First, clone this repository to your local machine using Git.
bash
Copy code
git clone <repository-url>
cd flight-route-planner
Set Up a Virtual Environment (Optional but recommended)
It's a good practice to create a virtual environment for your Python projects. This keeps dependencies required by different projects separate by creating isolated environments for them.
Create a virtual environment:
Copy code
python -m venv venv
Activate the virtual environment:
On Windows:
Copy code
.\venv\Scripts\activate
On macOS and Linux:
bash
Copy code
source venv/bin/activate
Install Dependencies
Install all required packages using pip:
Copy code
pip install -r requirements.txt
This command reads the requirements.txt file and installs all the Python packages listed there. For this program, you'll need to ensure you have dash, plotly, and other dependencies installed.
Data Preparation

Ensure you have the necessary CSV files (airports.csv, routes.csv, planes.csv) in the project directory. These files should contain data about airports, flight routes, and CO2 emissions related to different aircraft, respectively.
Running the Program

Start the Application
With all dependencies installed, you can now run the Dash app:
php
Copy code
python <name-of-the-python-file>.py
Replace <name-of-the-python-file> with the actual name of the Python script you wish to run.
Access the Dashboard
Once the server starts, you'll see a message in the console indicating the URL where the Dash app is running (typically http://127.0.0.1:8050/). Open this URL in a web browser to interact with the Flight Route Planner dashboard.
Usage

Use the dropdown menus to select the start and destination airports, exclude any airports if necessary, and choose a plane.
Click the "Find Routes" button to display possible routes.
The map will update to show the routes. Hover over the routes for more details.
Use the "Sort routes by" dropdown to organize the displayed routes according to distance, cost, or environmental impact.
