# 🛡️ DeepShield AI

### AI-Powered Cybersecurity Threat Detection & Security Monitoring Platform

<p align="center">
  <b>Detect. Analyze. Explain. Protect.</b>
</p>

<p align="center">
  DeepShield AI is a full-stack AI-powered cybersecurity platform designed to detect, analyze, score, explain, and visualize multiple types of cyber threats from one unified Security Operations Center.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/React-Frontend-61DAFB?logo=react&logoColor=black" />
  <img src="https://img.shields.io/badge/Vite-Frontend-646CFF?logo=vite&logoColor=white" />
  <img src="https://img.shields.io/badge/TailwindCSS-UI-06B6D4?logo=tailwindcss&logoColor=white" />
  <img src="https://img.shields.io/badge/Machine%20Learning-AI-purple" />
  <img src="https://img.shields.io/badge/Domain-Cybersecurity-red" />
  <img src="https://img.shields.io/badge/Status-In%20Development-orange" />
</p>

---

## 📌 Overview

**DeepShield AI** is being developed as a major portfolio-level Artificial Intelligence and Cybersecurity project.

The goal is to build an intelligent cybersecurity command center capable of combining:

- Machine Learning
- Natural Language Processing
- Anomaly Detection
- Behavioural Analysis
- Threat Scoring
- Explainable AI
- Security Rules
- Security Event Monitoring
- Full-Stack Development
- Security Analytics

Instead of creating isolated ML models, DeepShield AI aims to connect multiple security detection systems through one unified platform.

The final system will generate outputs such as:

```text
Threat Probability
Risk Score: 0–100
Threat Level
Threat Category
Detection Reasons
AI Explanation
Recommended Security Action
```

---

# 🎯 Problem Statement

Modern cybersecurity systems generate large amounts of data from URLs, messages, authentication systems, network traffic, devices, and user behaviour.

Analyzing these security signals manually can be difficult and slow.

DeepShield AI aims to provide a unified platform where AI/ML models can help identify potentially dangerous activity and present the results in a clear and understandable form.

---

# 💡 Proposed Solution

DeepShield AI acts as an **AI-powered Security Operations Center (SOC)**.

The platform will combine multiple cybersecurity detection modules and send their results to a central risk engine.

```text
Security Data
     ↓
Feature Extraction
     ↓
AI / ML Models
     ↓
Threat Probability
     ↓
DeepShield Risk Engine
     ↓
Risk Score 0–100
     ↓
Threat Classification
     ↓
Explainable Detection
     ↓
Alerts + Security Dashboard
```

---

# ✅ Current Project Status

## Phase 1 — Project Foundation

The first development phase has been completed.

Currently implemented:

```text
✅ Python 3.12 development environment
✅ Modular FastAPI backend foundation
✅ React + Vite frontend
✅ Tailwind CSS integration
✅ Professional dark cybersecurity dashboard
✅ Frontend → Backend API communication
✅ Environment variable configuration
✅ CORS configuration
✅ Backend health API
✅ Responsive sidebar foundation
✅ System health monitoring
✅ Basic backend API tests
✅ Git configuration
✅ Secure .gitignore
✅ GitHub repository
```

Current backend version:

```text
v0.1.0
```

---

# 🚀 Planned Core Modules

DeepShield AI is being developed phase by phase.

### 🔗 Malicious URL Detection

Analyze URLs and classify them as:

```text
Safe
Suspicious
Phishing
Malicious
```

Planned features include:

```text
URL length
Subdomains
Special characters
HTTPS usage
IP address detection
Suspicious keywords
URL entropy
Digit frequency
Redirect patterns
URL shortener detection
Domain-related features
```

Models will be compared using real evaluation metrics before selecting the final model.

---

### 📩 Phishing Message Detection

Analyze:

```text
Email
SMS
WhatsApp-like messages
Social-media messages
```

Possible classifications:

```text
Legitimate
Spam
Suspicious
Phishing
```

Initial NLP approach:

```text
Text preprocessing
TF-IDF
Logistic Regression
Naive Bayes
SVM
```

Deep learning or transformer models will only be added if they provide meaningful improvement.

---

### 🌐 Network Intrusion Detection

The Network Intrusion Detection System will analyze network traffic and identify:

```text
Normal Traffic
DoS / DDoS
Bot Activity
Brute Force
Port Scanning
Web Attacks
Other Suspicious Behaviour
```

Possible datasets:

```text
CIC-IDS2017
CIC-IDS2018
UNSW-NB15
```

---

### 🧠 AI Anomaly Detection

The anomaly detection engine will identify unusual security behaviour even when it does not exactly match a known attack.

Possible models:

```text
Isolation Forest
One-Class SVM
Autoencoder — only if justified
```

---

### 👤 User & Entity Behaviour Analytics

DeepShield AI will include a simplified UEBA system that can analyze:

```text
Login frequency
Failed logins
Login time
Device changes
IP changes
Location changes
Request frequency
Sudden behaviour changes
```

---

### ⚡ Brute-Force Detection

Authentication logs will be analyzed for patterns such as:

```text
Repeated failed logins
Rapid authentication attempts
Multiple attempts from one IP
Distributed login attempts
Multiple targeted accounts
```

---

### 📊 Unified DeepShield Risk Engine

A central risk engine will combine relevant signals from different AI modules.

Example:

```text
URL Risk          = 82
Message Risk      = 74
Network Risk      = 93
Behaviour Risk    = 65
Anomaly Risk      = 81
```

The system will calculate a logical final score rather than generating random values.

Risk levels:

| Risk Score | Level |
|---|---|
| 0–24 | SAFE |
| 25–49 | LOW |
| 50–69 | MEDIUM |
| 70–84 | HIGH |
| 85–100 | CRITICAL |

---

### 🔎 Explainable AI

DeepShield AI will not only provide a prediction.

It will explain **why** something was classified as dangerous.

Possible explanation methods:

```text
Feature contribution
Model probability
Feature importance
SHAP
Rule-based explanations
Security indicators
```

Example:

```text
Threat Score: 91/100

Threat Level:
CRITICAL

Threat Type:
Phishing Website

Detected Because:
• Suspicious URL structure
• Credential-related keyword detected
• Unusual number of subdomains
• ML phishing probability: 94%

Recommended Action:
Block and avoid visiting this URL.
```

---

### 🚨 Security Alert System

Alerts will contain information such as:

```text
Alert ID
Timestamp
Threat Type
Source
Risk Score
Severity
Status
Description
Recommended Action
```

Alert severity:

```text
INFO
LOW
MEDIUM
HIGH
CRITICAL
```

Alert status:

```text
New
Investigating
Resolved
Ignored
```

---

### 📜 Security Event Monitoring

Security events will eventually be stored and managed through a database.

Planned fields:

```text
Event ID
Timestamp
Source IP
Target
Event Type
Risk Score
Detection Model
Status
```

The interface will support:

```text
Search
Filtering
Sorting
Pagination
```

---

### 🕸️ Threat Network Visualization

DeepShield AI will visualize relationships between entities such as:

```text
IP Address
      ↓
User Account
      ↓
Suspicious URL
      ↓
Malicious Domain
      ↓
Security Incident
```

A graph visualization library such as React Flow may be integrated later.

---

### 🤖 DeepShield AI Security Assistant

The security assistant will help users understand detection results.

Users will eventually be able to ask questions such as:

```text
Explain this threat.

Why was this detected?

How dangerous is this event?

What security action should be taken?
```

The first version will work from structured security results.

An external LLM API will remain optional.

---

# 🖥️ Current Dashboard

The current frontend provides the initial DeepShield AI Security Operations Center interface.

Current dashboard sections include:

```text
Security Overview
System Status
Threat Metrics
Detection Module Status
Threat Activity Area
Recent Security Alerts
System Health
Backend Status
Responsive Sidebar
```

Because AI detection modules are not trained yet, DeepShield AI intentionally displays:

```text
Threats Detected: 0
Critical Alerts: 0
Active Events: 0
Security Score: —
```

No fake cybersecurity events, fake model predictions, or fake accuracy values are displayed.

---

# 🏗️ System Architecture

```text
┌───────────────────────────────────┐
│          React Frontend           │
│                                   │
│ Dashboard • Scanners • Analytics  │
└─────────────────┬─────────────────┘
                  │
                  │ REST API
                  ↓
┌───────────────────────────────────┐
│          FastAPI Backend          │
│                                   │
│ API Routes • Services • Security  │
└─────────────────┬─────────────────┘
                  │
                  ↓
┌───────────────────────────────────┐
│          AI / ML Layer            │
│                                   │
│ URL Detection                     │
│ Phishing Detection                │
│ Intrusion Detection               │
│ Anomaly Detection                 │
│ Behaviour Analytics               │
└─────────────────┬─────────────────┘
                  │
                  ↓
┌───────────────────────────────────┐
│      DeepShield Risk Engine       │
└─────────────────┬─────────────────┘
                  │
                  ↓
┌───────────────────────────────────┐
│       Security Data Layer         │
│                                   │
│ SQLite → PostgreSQL               │
│ Events • Alerts • Incidents       │
└───────────────────────────────────┘
```

---

# 🧰 Technology Stack

## Frontend

```text
React
Vite
Tailwind CSS
React Icons
Axios
React Router
Recharts
```

Future:

```text
React Flow
```

## Backend

```text
Python 3.12
FastAPI
Uvicorn
Pydantic
Pydantic Settings
Python Dotenv
```

## Machine Learning

```text
Scikit-learn
Pandas
NumPy
Joblib
Matplotlib
```

Optional when justified:

```text
XGBoost
SHAP
TensorFlow / Keras
Transformer Models
```

## Database

Development:

```text
SQLite
```

Future production option:

```text
PostgreSQL
```

## Testing

```text
Pytest
FastAPI TestClient
```

---

# 📂 Project Structure

```text
DeepShield-AI/
│
├── backend/
│   │
│   ├── app/
│   │   ├── main.py
│   │   ├── api/
│   │   ├── core/
│   │   ├── services/
│   │   ├── models/
│   │   ├── schemas/
│   │   ├── database/
│   │   └── utils/
│   │
│   ├── ml/
│   │   ├── url_detection/
│   │   ├── phishing_detection/
│   │   ├── intrusion_detection/
│   │   ├── anomaly_detection/
│   │   └── saved_models/
│   │
│   ├── tests/
│   ├── requirements.txt
│   └── .env.example
│
├── frontend/
│   │
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   ├── layouts/
│   │   ├── services/
│   │   ├── hooks/
│   │   ├── utils/
│   │   ├── App.jsx
│   │   ├── index.css
│   │   └── main.jsx
│   │
│   ├── public/
│   └── package.json
│
├── datasets/
├── notebooks/
├── docs/
├── screenshots/
│
├── .gitignore
└── README.md
```

---

# ⚙️ Installation

## 1. Clone Repository

```bash
git clone https://github.com/Aryan2624/DeepShield-AI.git
```

```bash
cd DeepShield-AI
```

---

# 🐍 Backend Setup

Go to the backend:

```bash
cd backend
```

Create a Python 3.12 environment on Windows:

```bash
py -3.12 -m venv .venv
```

Activate it using Command Prompt:

```bash
.venv\Scripts\activate.bat
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create your environment file from:

```text
.env.example
```

Current development values:

```env
APP_NAME=DeepShield AI API
APP_VERSION=0.1.0
ENVIRONMENT=development
FRONTEND_URL=http://localhost:5173
```

Start FastAPI:

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Backend:

```text
http://127.0.0.1:8000
```

API documentation:

```text
http://127.0.0.1:8000/docs
```

Health endpoint:

```text
http://127.0.0.1:8000/api/health
```

---

# ⚛️ Frontend Setup

Open another terminal.

```bash
cd frontend
```

Install dependencies:

```bash
npm install
```

Create:

```text
.env
```

with:

```env
VITE_API_BASE_URL=http://127.0.0.1:8000
```

Start the frontend:

```bash
npm run dev
```

Open:

```text
http://localhost:5173
```

---

# 🔌 Current API

## Health Check

```http
GET /api/health
```

Example response:

```json
{
  "success": true,
  "data": {
    "service": "DeepShield AI API",
    "status": "healthy",
    "version": "0.1.0",
    "environment": "development"
  }
}
```

---

# 🧪 Testing

Activate the backend environment and run:

```bash
python -m pytest -q
```

Current Phase 1 result:

```text
2 passed
```

Tests currently verify:

```text
Root API endpoint
Health API endpoint
Response status
Expected response structure
```

Testing coverage will grow as AI modules are implemented.

---

# 📊 Machine Learning Quality Standards

DeepShield AI will not evaluate models using accuracy alone.

Depending on the problem, evaluation will include:

```text
Accuracy
Precision
Recall
F1-score
ROC-AUC
Confusion Matrix
Class Distribution
False Positive Analysis
False Negative Analysis
```

Cybersecurity datasets may be highly imbalanced, so model evaluation will consider class imbalance carefully.

The project follows these rules:

```text
No training on test data
No data leakage
Preprocessing fitted only on training data
Separate test datasets
Reproducible random states
No fake metrics
No hard-coded predictions
No random risk scores
```

---

# 📁 Dataset Policy

Large cybersecurity datasets are **not committed directly to GitHub**.

Datasets will be stored locally inside:

```text
datasets/
```

Examples:

```text
datasets/
├── phishing_urls/
├── phishing_messages/
├── intrusion_detection/
└── behavioural_logs/
```

Dataset download information and references will be documented separately.

---

# 🔐 Security Practices

DeepShield AI is focused on:

```text
Detection
Monitoring
Analysis
Protection
Incident Response
Security Education
```

Application security practices include:

```text
Environment variables
Input validation
CORS configuration
Safe API error handling
Protected secrets
Secure authentication when implemented
Password hashing when implemented
Rate limiting where appropriate
No secret API keys committed to GitHub
```

DeepShield AI is not intended to provide offensive hacking capabilities.

---

# 🗺️ Development Roadmap

```text
✅ Phase 1  — Project Foundation

⏳ Phase 2  — Malicious URL Detection AI

⬜ Phase 3  — Phishing Message Detection AI

⬜ Phase 4  — Unified Risk Engine

⬜ Phase 5  — Main SOC Dashboard

⬜ Phase 6  — Network Intrusion Detection

⬜ Phase 7  — Anomaly & Behaviour Detection

⬜ Phase 8  — Alert & Incident System

⬜ Phase 9  — Explainable AI

⬜ Phase 10 — Threat Network Visualization

⬜ Phase 11 — Model Analytics

⬜ Phase 12 — Authentication & Security

⬜ Phase 13 — Testing & Optimization

⬜ Phase 14 — Deployment

⬜ Phase 15 — Portfolio & Documentation
```

---

# 🎯 Final Project Goal

The completed DeepShield AI platform is intended to demonstrate practical knowledge of:

```text
Python
Machine Learning
Natural Language Processing
Anomaly Detection
Cybersecurity Data
Feature Engineering
Model Evaluation
Explainable AI
REST APIs
FastAPI
React
Data Visualization
Databases
Full-Stack Integration
Software Architecture
Testing
Git & GitHub
Deployment
```

The final vision is:

> **An AI-powered Security Operations Center capable of detecting, analyzing, scoring, explaining, and visualizing multiple types of cyber threats from one unified platform.**

---

# 👨‍💻 Developer

**Aryan Dubey**

B.Tech Artificial Intelligence & Machine Learning Student  
Aspiring AI Engineer

GitHub:

[github.com/Aryan2624](https://github.com/Aryan2624)

---

# 🤝 Contributions

DeepShield AI is currently under active development.

Suggestions, issues, and future contributions are welcome as the project evolves.

---

# ⭐ Support

If you find DeepShield AI interesting, consider giving the repository a ⭐.

It helps support the project as it continues to grow.

---

<p align="center">
  <b>🛡️ DeepShield AI</b>
</p>

<p align="center">
  Detect • Analyze • Explain • Protect
</p>
