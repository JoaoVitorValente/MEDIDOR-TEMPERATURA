
import sqlite3
import pandas as pd
import datetime
import random

from dash import Dash, dcc, html
import plotly.graph_objs as go
import dash

from sklearn.linear_model import LinearRegression
import numpy as np

DB_NAME = "temp_monitor_data_pro.db"

# -----------------------------
# Create database if not exists
# -----------------------------
conn = sqlite3.connect(DB_NAME)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS temperature_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    temperature REAL
)
""")
conn.commit()

# -----------------------------
# Simulate sensor data
# -----------------------------
def insert_fake_data():
    temp = round(random.uniform(20, 30), 2)
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cursor.execute(
        "INSERT INTO temperature_data (timestamp, temperature) VALUES (?, ?)",
        (now, temp)
    )
    conn.commit()

# -----------------------------
# Load data
# -----------------------------
def load_data():
    df = pd.read_sql_query("SELECT * FROM temperature_data", conn)
    return df

# -----------------------------
# AI Prediction (Linear Regression)
# -----------------------------
def predict_temperature(df):
    if len(df) < 5:
        return None

    df["time_index"] = range(len(df))

    X = df[["time_index"]]
    y = df["temperature"]

    model = LinearRegression()
    model.fit(X, y)

    future_index = np.array([[len(df) + 5]])
    prediction = model.predict(future_index)

    return round(prediction[0], 2)

# -----------------------------
# Dash App
# -----------------------------
app = Dash(__name__)

app.layout = html.Div([
    html.H1("🌡️ Monitor de Temperatura com IA"),

    dcc.Interval(id='interval', interval=2000, n_intervals=0),

    html.Div(id="current-temp", style={"fontSize": 30}),
    html.Div(id="prediction", style={"fontSize": 20}),

    dcc.Graph(id="temp-graph")
])

@app.callback(
    [
        dash.dependencies.Output("temp-graph", "figure"),
        dash.dependencies.Output("current-temp", "children"),
        dash.dependencies.Output("prediction", "children")
    ],
    [dash.dependencies.Input("interval", "n_intervals")]
)
def update(n):
    insert_fake_data()
    df = load_data()

    if df.empty:
        return {}, "Sem dados", ""

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["timestamp"],
        y=df["temperature"],
        mode='lines+markers',
        name="Temperatura"
    ))

    current_temp = df["temperature"].iloc[-1]

    pred = predict_temperature(df)

    pred_text = ""
    if pred:
        pred_text = f"Previsão (IA): {pred} °C"

    return fig, f"Temperatura Atual: {current_temp} °C", pred_text

# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
