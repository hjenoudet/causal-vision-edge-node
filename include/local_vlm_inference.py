import os
import requests
from pydantic import BaseModel, Field, ValidationError
from supabase import create_client, Client
from typing import Dict, Any

class RobotAction(BaseModel):
    task: str = Field(description="'monitor', 'targeted_spray', 'isolate', 're-inspect', or 'none'")
    priority: str = Field(description="'low', 'medium', or 'high'")
    params: Dict[str, Any] = Field(description="Dynamic ROS parameters")

class DiagnosticReport(BaseModel):
    disease: str = Field(description="Exact disease name or 'Healthy'")
    severity: str = Field(description="'none', 'early', 'moderate', or 'severe'")
    confidence: float = Field(description="Confidence float from 0.0 to 1.0")
    weather_risk: str = Field(description="Explanation of how Temp/Humidity/VPD triggered this")
    reasoning: str = Field(description="Detailed phytopathology + weather correlation")
    prediction: str = Field(description="3-7 day progression forecast based on thermodynamics")
    robot_action: RobotAction

def run_agentic_diagnostics():
    supabase: Client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_KEY"))
    
    res = supabase.table("telemetry_logs")\
        .select("*")\
        .eq("is_anomaly", True)\
        .is_("llm_diagnostic", "null")\
        .execute()
        
    anomalous_records = res.data
    if not anomalous_records:
        print("No pending anomalies require AI diagnosis.")
        return

    # Route Docker traffic to the native macOS host
    NATIVE_MLX_API_URL = "http://host.docker.internal:8000/diagnose"

    for record in anomalous_records:
        print(f"🧠 Routing {record['zone_id']} causal vector to Native MLX NPU...")
        
        image_url = record.get('image_url')
        if not image_url: continue

        payload = {
            "image_url": image_url,
            "zone_id": record['zone_id'],
            "p_val": float(record.get('p_value', 1.0)),
            "vpd": float(record.get('vpd_kpa', 0.0)),
            "humidity": int(record.get('humidity_percent', 0)),
            "temp": float(record.get('temperature_c', 0.0))
        }
        
        try:
            response = requests.post(NATIVE_MLX_API_URL, json=payload, timeout=120)
            response.raise_for_status()
            
            clean_json_str = response.json().get("json_payload", "{}")
            diagnostic_data = DiagnosticReport.model_validate_json(clean_json_str)
            
            supabase.table("telemetry_logs").update({
                "llm_diagnostic": diagnostic_data.model_dump()
            }).eq("id", record["id"]).execute()
            
            print(f"✅ Diagnosed {record['zone_id']}: {diagnostic_data.disease} (Conf: {diagnostic_data.confidence:.2f})")
            print(f"🤖 Action Queued: {diagnostic_data.robot_action.task.upper()}")
            
        except requests.exceptions.ConnectionError:
            print("🚨 ERROR: Cannot reach Native MLX API. Ensure native_mlx_server.py is running on the macOS host.")
        except ValidationError as e:
            print(f"🚨 ERROR: Schema validation failed. Model hallucinations detected. \n{e}")
        except Exception as e:
            print(f"❌ Pipeline Failure: {e}")
