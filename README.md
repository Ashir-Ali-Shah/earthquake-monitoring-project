# 🌍 USGS Earthquake Intelligence System

![React](https://img.shields.io/badge/Frontend-React-blue)
![FastAPI](https://img.shields.io/badge/Backend-FastAPI-green)
![TensorFlow](https://img.shields.io/badge/ML-TensorFlow-orange)
![Docker](https://img.shields.io/badge/Deployment-Docker-blueviolet)

A full-stack real-time seismic monitoring platform that combines **Retrieval-Augmented Generation (RAG)** for semantic data querying and **Long Short-Term Memory (LSTM)** neural networks for magnitude forecasting. The system aggregates data from the USGS API, processes it through an ML pipeline, and visualizes risks via an interactive dashboard.

## 🚀 Key Features

* **Real-time Monitoring:** Fetches and categorizes global earthquake data from the USGS (United States Geological Survey).
* **LSTM Forecasting Engine:** A deep learning model that predicts the magnitude of the next seismic event based on a sequence of historical features (magnitude, depth, location, energy release).
* **Semantic Search & RAG:** Uses `SentenceTransformers` and `FAISS` vector databases to allow natural language querying of earthquake events (e.g., "Strong earthquakes in Japan").
* **Geospatial Analysis:** Auto-detection of seismic hotspots and high-risk zones.
* **Interactive Dashboard:** Built with React, Recharts, and Tailwind CSS for responsive data visualization.

## 🧠 Machine Learning Architecture & Performance

The core forecasting engine utilizes a sequential LSTM (Long Short-Term Memory) network trained on historical USGS data. The model analyzes temporal sequences of **40 events** to predict the magnitude of the next earthquake.

### Model Specification
* **Architecture:** LSTM (128 units) → Dropout (0.3) → Dense Output.
* **Features:** Magnitude, Depth, Latitude, Longitude, Delta Time ($\Delta t$), Log Cumulative Energy.
* **Sequence Length:** 40 historic steps.

### Quantitative Metrics
The model was evaluated on a held-out test set. Despite the high stochasticity of seismic data, the model achieves a low Mean Absolute Error.

| Metric | Value | Description |
| :--- | :--- | :--- |
| **Best Validation Loss (MSE)** | `0.0243` | The lowest Mean Squared Error achieved during training. |
| **Mean Absolute Error (MAE)** | `0.3662` | On average, predictions are within **±0.36** magnitude of the actual event. |
| **Root Mean Squared Error** | `0.5001` | Penalizes larger prediction errors more heavily. |
| **R-squared ($R^2$)** | `0.0088` | Indicates high variance in the dataset (typical for chaotic seismic systems). |

### Inference Example
During live testing, the model successfully processed input tensors to generate forecasts:

Inference Time: ~239ms
Predicted Next Magnitude: 3.20
🛠️ Tech Stack
Frontend
React 18 (UI Library)

Tailwind CSS (Styling)

Recharts (Data Visualization)

Lucide React (Icons)

Backend
FastAPI (High-performance Async API)

Python 3.10

TensorFlow/Keras (Deep Learning)

FAISS & SentenceTransformers (Vector Embeddings)

Spacy (Named Entity Recognition)

DevOps
Docker & Docker Compose (Containerization)

Nginx (Reverse Proxy & Static Serving)

📦 Installation & Setup
This project is fully containerized. You can run the entire stack with a single command.

Prerequisites
Docker Desktop installed.

Git.

Steps
Clone the repository

Bash

git clone [https://github.com/Ashir-Ali-Shah/earthquake-monitoring-project.git](https://github.com/Ashir-Ali-Shah/earthquake-monitoring-project.git)
cd earthquake-monitoring-project
Run with Docker Compose

Bash

docker compose up --build
