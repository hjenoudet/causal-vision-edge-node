# 🛰️ Causal Vision Edge Node
### Sovereign, Hardware-Accelerated Autonomous Fleet Command

An industrial-grade, edge-deployed data pipeline combining **Causal Inference Math** and **Native Local Vision-Language Models (VLMs)** to detect and diagnose biological anomalies in real-time. This project operates as a **Zero-Marginal-Cost Digital Twin**, leveraging local silicon (Apple M5 NPU) and open-source orchestration to manage a 15-zone global fleet.

---

## 🛠️ The "Cutting Edge" Tech Stack

### **Hardware & Core Inference**
*   **Apple Silicon M5 NPU:** Direct hardware acceleration via the Neural Processing Unit, bypassing CPU/GPU bottlenecks for dedicated 4-bit quantized tensor operations.
*   **MLX (Apple Native):** Utilizing Apple's open-source array framework for high-performance machine learning on Silicon.
*   **Pixtral 12B (VLM):** A 12-billion parameter multi-modal model analyzing high-resolution leaf imagery alongside thermodynamic causal vectors.
*   **FastAPI Engine:** Native macOS server routing Dockerized Airflow traffic to the host NPU for sub-10s inference.

### **Data Science & Causal Math**
*   **Linear Mixed Models (LMM):** Using `statsmodels` to isolate "Biological Anomalies" from environmental noise by calculating zone-specific random effects.
*   **Tetens Thermodynamics:** Real-time calculation of **Vapor Pressure Deficit (VPD)** to measure plant transpiration stress.
*   **Open-Meteo API:** High-resolution, global historical weather data (30-day lookback) without API key overhead.

### **Orchestration & Cloud-Native Edge**
*   **Apache Airflow:** Enterprise-grade DAG orchestration managing the daily lifecycle of global data fusion, math processing, and VLM triggering.
*   **Supabase (Postgres + Realtime):** Using PostgreSQL as a "Causal Ledger" and WebSockets to push NPU diagnoses to the UI in milliseconds.
*   **Next.js & React Three Fiber:** A spatial "Command Center" using WebGL/Three.js to visualize the 15-zone global fleet in a 3D coordinate system.

---

## 🔄 The "Level 12" Workflows

1.  **The Causal Gatekeeper:** A "Low-Energy" math layer that stays asleep during nominal states. It only "wakes up" the power-hungry VLM NPU when a $p < 0.05$ causal deviation is detected.
2.  **The Recursive Edge-to-Cloud Loop:** Local NPU pulls from Cloud S3 $\rightarrow$ Performs Inference $\rightarrow$ Pushes JSON back to Cloud DB $\rightarrow$ UI re-renders via Realtime subscription.
3.  **Physically-Informed Simulation:** A 15-zone "Global Twin" modeling real-world climate offsets (e.g., cool/moist Seattle vs. hot/arid Lerida) to provide the statistical variance required for LMM convergence.

---

## 🧪 Tests, Doubts, and Solutions

| The Doubt | The Engineering Solution |
| :--- | :--- |
| **"The Apple Scab Bias"** | Discovered the VLM was "hallucinating" Scab due to simulation defaults in dry weather. Fixed by **randomizing the simulation fallback** to test the VLM's actual visual discrimination. |
| **"Convergence Failure"** | The LMM failed to converge on small data with zero zone variance. Scaled to **15 global zones** and implemented **"Zone Personalities"** (unique climate offsets), stabilizing the Hessian matrix. |
| **"VLM Instruction Failure"** | The VLM ignored Pydantic categories. Implemented a **Vision Sharpening Guide** (morphological markers like "frog-eye spots") and **Schema Enforcement** to force scientific accuracy. |
| **"The Ghost Error"** | Diagnosed 500 errors as a **Database Queue** issue where the Agent processed old, broken URLs. Implemented a "Self-Healing" logic where the Agent logs errors and advances. |

---

## 💎 Final Verdict
This project is **100% OSS, Free, and Compliant**. It demonstrates how a sovereign architecture can build an industrial-grade, climate-aware robotic vision fleet that costs **$0.00/month** to operate by leveraging local silicon and modern open-source orchestration.

---

## 🚀 Running Locally

### Prerequisites
* **Hardware:** Apple Silicon (M1-M5) highly recommended for NPU acceleration.
* **Orchestration:** [Astro CLI](https://docs.astronomer.io/astro/cli/install-cli).
* **AI Engine:** Python 3.12+ with `mlx-vlm` installed.

### Setup
1. Clone this repository.
2. Initialize the Local NPU Server:
   ```bash
   python3 vlm_training/native_mlx_server.py
   ```
3. Start the Airflow Environment:
   ```bash
   astro dev start
   ```
4. Access the Command Center UI at `http://localhost:3000` to watch the global fleet in real-time.
