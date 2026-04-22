import os
from airflow.decorators import dag, task
from pendulum import datetime
from supabase import create_client, Client
from include.data_fusion import fetch_weather_panel, map_biological_state
from include.causal_math import detect_anomalies
from include.local_vlm_inference import run_agentic_diagnostics

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

@dag(
    start_date=datetime(2024, 1, 1),
    schedule="@daily",
    catchup=False,
    tags=["AgTech", "MLOps", "Causal Inference"]
)
def edge_diagnostics_pipeline():

    @task(multiple_outputs=True)
    def run_causal_gatekeeper():
        df = fetch_weather_panel(days=14)
        results = detect_anomalies(df)
        today_data = results["today_data"]
        
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        anomalies_detected = False
        
        for row in today_data:
            is_anomalous = bool(row['p_value'] < 0.05 and row['residual'] > 0)
            if is_anomalous:
                anomalies_detected = True
                
            image_url = map_biological_state(
                is_anomalous, row['temp_c'], row['humidity_pct'], row['precip_mm']
            )
            
            payload = {
                "zone_id": row['zone_id'], "temperature_c": row['temp_c'],
                "humidity_percent": row['humidity_pct'], "precip_mm": row['precip_mm'],
                "vpd_kpa": row['vpd_kpa'], "stress_index": row['stress_index'],
                "residual": row['residual'], "p_value": row['p_value'],
                "is_anomaly": is_anomalous, "image_url": image_url
            }
            
            supabase.table("telemetry_logs").insert(payload).execute()
            if is_anomalous:
                print(f"🚨 ANOMALY in {row['zone_id']} | P-Val: {row['p_value']:.4f} | Image: {image_url}")

        return {"trigger_vlm": anomalies_detected, "beta_vpd": results["beta_vpd"]}

    @task.branch
    def conditional_vlm_routing(gatekeeper_res: dict):
        if gatekeeper_res["trigger_vlm"]: return "trigger_ollama_agent"
        return "log_healthy_state"
        
    @task
    def trigger_ollama_agent(gatekeeper_res: dict):
        print("Causal Gatekeeper OPEN. Waking up local Ollama VLM agent...")
        run_agentic_diagnostics()
        # To be built in Phase 2
        
    @task
    def log_healthy_state():
        print("All zones normal. Edge compute remains asleep to save battery.")
        
    # Wire the Graph
    gatekeeper = run_causal_gatekeeper()
    branch = conditional_vlm_routing(gatekeeper)
    gatekeeper >> branch >> [trigger_ollama_agent(gatekeeper), log_healthy_state()]

edge_diagnostics_pipeline()