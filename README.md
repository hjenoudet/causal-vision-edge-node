# 🛰️ Causal Vision Edge Node

An edge-deployed autonomous data pipeline combining **Causal Inference Math** and **Local Vision-Language Models (VLMs)** to detect, verify, and diagnose biological anomalies. 

This repository serves as the backend / edge-compute layer for the **Autonomous Fleet Command** dashboard. 

## 🧠 Core Architecture
This pipeline does not rely on simple thresholds. It uses live thermodynamic data and statistical models to establish causal baselines before triggering AI inference.
1. **Thermodynamic Data Fusion:** Fetches **live weather conditions** and 14-day rolling trends (Open-Meteo API) to calculate the current local Vapor Pressure Deficit (VPD).
2. **Causal Gatekeeper:** Fits a Mixed Linear Model (`statsmodels`) to predict expected biological stress. If the live stress deviates significantly from the baseline (p < 0.05), it triggers an anomaly state.
3. **Local VLM Inference:** A completely air-gapped, offline LLM (Llama 3.2-Vision via Ollama) processes the visual anomaly to diagnose pathogens and recommend actions.
4. **Real-time Telemetry:** Diagnostic results are pushed to a PostgreSQL database (Supabase) and streamed instantly to the Next.js/Three.js frontend via WebSockets.

## 🛠️ Tech Stack
* **Orchestration:** Apache Airflow (Astronomer Astro CLI) — *Scheduled autonomously every day at 5:00 PM.*
* **Math/Stats:** Python, Pandas, Statsmodels, SciPy
* **AI/Inference:** Ollama (Local Llava)
* **Database:** Supabase (PostgreSQL)

## 🚀 Running Locally

### Prerequisites
* Docker Desktop installed and running.
* [Astro CLI](https://docs.astronomer.io/astro/cli/install-cli) installed.
* [Ollama](https://ollama.com/) installed with Llava (`ollama run llava`).
* A Supabase project with a `telemetry_logs` table.

### Setup
1. Clone this repository.
2. Create a `.env` file in the root directory and add your Supabase credentials:
   ```env
   SUPABASE_URL=your_supabase_url
   SUPABASE_SERVICE_KEY=your_service_role_key
   ```
3. Start the local Airflow environment:
   ```bash
   astro dev start
   ```
4. Start your local Ollama server:
   ```bash
   brew services start ollama
   ```
5. Access the Airflow UI at `http://localhost:8080` (default credentials: `admin` / `admin`). The `edge_diagnostics_pipeline` DAG will run automatically at 5:00 PM, or you can trigger it manually!