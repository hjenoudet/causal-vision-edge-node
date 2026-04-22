import os
import base64
import requests
from pydantic import BaseModel, Field, ValidationError
from supabase import create_client, Client

# 1. Enforce strict JSON schemas using Pydantic
class DiagnosticReport(BaseModel):
    pathogen_detected: str = Field(description="Exact disease name (e.g., Apple Scab, Black Rot, Rust) or 'Healthy'")
    confidence_score: float = Field(description="Confidence float from 0.0 to 1.0")
    recommended_action: str = Field(description="Action: 'QUARANTINE_ZONE', 'SPRAY_FUNGICIDE', or 'MONITOR'")
    severity: str = Field(description="Must be 'low', 'medium', or 'high'")

def fetch_and_encode_image(image_url: str) -> str:
    """Downloads image into RAM and converts to Base64 for the edge VLM."""
    response = requests.get(image_url)
    response.raise_for_status()
    return base64.b64encode(response.content).decode('utf-8')

def run_agentic_diagnostics():
    """Finds undiagnosed anomalies, queries the local VLM, and updates Cloud HQ."""
    supabase: Client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_KEY"))
    
    # 2. Idempotent fetch: Find anomalies that haven't been diagnosed yet
    res = supabase.table("telemetry_logs")\
        .select("*")\
        .eq("is_anomaly", True)\
        .is_("llm_diagnostic", "null")\
        .execute()
        
    anomalous_records = res.data
    if not anomalous_records:
        print("No pending anomalies require AI diagnosis.")
        return

    # Docker-to-Host Networking Fix
    OLLAMA_API_URL = "http://host.docker.internal:11434/api/generate"

    for record in anomalous_records:
        print(f"🧠 Routing {record['zone_id']} image to Edge AI (LLaVA)...")
        b64_image = fetch_and_encode_image(record['image_url'])
        
        prompt = f"""
        You are the diagnostic AI for an autonomous orchard rover. 
        The causal engine detected a biological anomaly in {record['zone_id']}.
        Context: P-Value={record['p_value']:.4f}, Humidity={record['humidity_percent']}%, Temp={record['temperature_c']}C.
        Analyze this apple leaf image. Respond ONLY with a JSON object matching this schema. No markdown formatting.
        {{
            "pathogen_detected": "string",
            "confidence_score": 0.0,
            "recommended_action": "string",
            "severity": "string"
        }}
        """
        
        payload = {
            "model": "llava",
            "prompt": prompt,
            "images": [b64_image],
            "format": "json", # Forces Ollama to strictly output valid JSON
            "stream": False,
            "options": {"temperature": 0.1} # Low temperature for deterministic robotics
        }
        
        try:
            # 3. Execute Edge Inference
            ai_response = requests.post(OLLAMA_API_URL, json=payload, timeout=120).json()
            raw_json_str = ai_response.get("response", "{}")
            
            # Clean potential markdown ticks from the LLM
            clean_json_str = raw_json_str.strip().strip('`').removeprefix('json').strip()
            
            # 4. Enforce Schema Validation via Pydantic
            diagnostic_data = DiagnosticReport.model_validate_json(clean_json_str)
            
            # 5. Sync the verified diagnostic back to Cloud HQ (Supabase)
            supabase.table("telemetry_logs").update({
                "llm_diagnostic": diagnostic_data.model_dump()
            }).eq("id", record["id"]).execute()
            
            print(f"✅ Diagnosed {record['zone_id']}: {diagnostic_data.pathogen_detected}")
            
        except requests.exceptions.ConnectionError:
            print("🚨 ERROR: Cannot reach Ollama. Is 'ollama run llava' running on the host machine?")
        except ValidationError as e:
            print(f"🚨 ERROR: LLM hallucinated bad JSON format for {record['zone_id']}. {e}")
        except Exception as e:
            print(f"❌ VLM Inference Failed for {record['zone_id']}: {e}")