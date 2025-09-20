🚦 Traffic Pulse Real-Time Vehicle Classification and Traffic Analysis

💡 About The Project Traffic Pulse is a real-time traffic analysis system designed to address urban congestion problems. The system captures live video feeds, classifies and counts the number of vehicles, and identifies their types (e.g., cars, bikes, trucks). By comparing this real-time vehicle population with historical data, it provides actionable insights into traffic density.

The primary purpose of Traffic Pulse is to empower users to make informed decisions about their daily commute. By pinning live camera footage, users can directly observe traffic conditions on their planned routes. This project is still in active development, with a focus on creating a robust and accurate classification model.

🚀 Features Real-time Video Processing: Analyzes live video feeds to provide up-to-the-minute traffic information.

Vehicle Classification: Distinguishes between different types of vehicles, such as cars, bikes, and trucks.

Vehicle Counting: Accurately counts the number of vehicles in the live feed.

Database Integration: Stores vehicle population data for comparison with historical trends.

Traffic Trend Analysis: Compares current traffic density with normal day averages to highlight unusual congestion.

User Alerts: Educates users about traffic conditions, helping them choose alternative routes to avoid congested areas.

🛠️ Built With Python: The primary programming language used for development.

OpenCV: Used for real-time video capture and processing.

TensorFlow/PyTorch: for vehicle classification.

MySQL/PostgreSQL: for storing and managing vehicle data.

🤝 Contribution This project is a collaborative effort by our team. We welcome contributions and suggestions to improve its accuracy and functionality.

Team Members:

Anshul H(Team Lead)

Hasnain Mohammed Shariff(Teammate 2)

Yashwanth P(Teammate 3)

Mohammed Hassan(Teammate 4)

-->How to run:

A. Run the Streamlit Application

pip install streamlit==1.30.0 opencv-python==4.8.1.78 ultralytics==8.0.20 numpy==1.25.0 pandas==2.1.0 plotly==5.15.0

streamlit run streamlit_app.py

B. Run the Command-Line Scripts

--For vehicle tracking: python main.py

--For speed estimation: python speed.py

The speed.py script will process the video and save the output to output.mp4 and a log file to static/vehicle_log.csv.

( Optional: Use a virtual environment

This is cleaner and avoids most Windows permission issues)

python -m venv venv venv\Scripts\activate pip install --upgrade pip pip install -r requirements.txt

📈 Future Scope Predictive Analysis: Implement a machine learning model to predict future traffic congestion based on historical data.

Mobile Application: Develop a user-friendly mobile app to provide real-time traffic updates and route recommendations.

Extended Vehicle Classification: Broaden the classification model to identify a wider range of vehicle types (e.g., buses, taxis, emergency vehicles).

📄 License This project is licensed under the MIT License - see the LICENSE.md file for details.










Traffic Pulse is a real-time system that analyzes live video feeds to classify and count vehicles. It compares current traffic density with historical data to identify congestion. This project aims to help users avoid traffic by providing real-time insights and enabling them to choose alternative routes.
