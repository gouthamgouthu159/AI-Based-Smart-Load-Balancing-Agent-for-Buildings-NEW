# AI-Based Smart Load Balancing Agent for Buildings

> Powered by **IBM Granite** on **IBM WatsonX AI** · Built with IBM Bob + IBM Cloud

## Overview

This project is an AI-powered system that monitors building electricity consumption, detects peak load events, and provides intelligent load balancing recommendations using **IBM Granite** via WatsonX AI.

### Building Systems Monitored
| System | Typical Range |
|---|---|
| HVAC | 13 – 99 kW |
| Lighting | 2 – 30 kW |
| Elevators | 0.5 – 20 kW |
| Appliances | 6 – 27 kW |
| Server Room | 22 – 30 kW |

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the Flask Server
```bash
python app.py
```
Server runs at `http://localhost:5000`

### 3. Open the Dashboard
Open `dashboard.html` in your browser.

---

## Project Structure

```
├── app.py              # Flask REST API backend
├── load_monitor.py     # Load monitoring & analysis module
├── peak_detector.py    # Peak load detection (Z-score analysis)
├── granite_ai.py       # IBM Granite AI integration (WatsonX AI)
├── dashboard.html      # Interactive web dashboard (dark theme)
├── test_modules.py     # Module validation script
├── requirements.txt    # Python dependencies
└── config.env          # IBM credentials
```

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/health` | GET | Health check |
| `/api/dashboard?hours=24` | GET | All dashboard data |
| `/api/load-data?hours=24` | GET | Per-system load data |
| `/api/peaks?hours=24` | GET | Peak detection results |
| `/api/recommendations?hours=24` | GET | IBM Granite load balancing recommendations |
| `/api/insights?hours=24` | GET | IBM Granite energy optimization insights |
| `/api/explain-anomaly` | POST | IBM Granite anomaly explanation |

---

## Tech Stack

- **IBM Granite** (`ibm/granite-13b-instruct-v2`) via IBM WatsonX AI
- **IBM Cloud** (WatsonX endpoint: `us-south.ml.cloud.ibm.com`)
- **Flask** – REST API backend
- **Pandas / NumPy / Scikit-learn** – Data processing & statistics
- **Chart.js** – Interactive dashboard charts
- **IBM Bob** – Development assistant

---

## Features

1. **Load Monitoring** – Real-time hourly energy data for 5 building systems
2. **Peak Detection** – Z-score based anomaly detection with severity classification
3. **AI Recommendations** – IBM Granite generates 5 actionable load balancing strategies
4. **Energy Insights** – IBM Granite produces demand charge reduction strategies
5. **Anomaly Explanation** – Click any peak event to get an IBM Granite analysis
6. **Interactive Dashboard** – Dark-theme web UI with 6 real-time charts
