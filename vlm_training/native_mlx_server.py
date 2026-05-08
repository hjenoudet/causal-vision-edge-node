import os
import json
import requests
import tempfile
import uvicorn
import traceback
import mlx.core as mx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from mlx_vlm import load, generate
from PIL import Image

app = FastAPI(title="RoboLeaf Native MLX NPU Engine")

# Force MLX to use the GPU/NPU explicitly
mx.set_default_device(mx.gpu)

print("🔌 Booting Native MLX Server on M5 NPU...")
try:
    # Load model once at startup in the main thread
    MODEL, PROCESSOR = load("mlx-community/pixtral-12b-4bit")
    print("✅ Model & Processor Loaded Successfully.")
except Exception as e:
    print(f"❌ FATAL: Failed to load model: {e}")
    traceback.print_exc()

class InferenceRequest(BaseModel):
    image_url: str
    zone_id: str
    p_val: float
    vpd: float
    humidity: int
    temp: float

SCHEMA_BLUEPRINT = {
    "disease": "string", "severity": "none|early|moderate|severe", "confidence": "float (0.0-1.0)",
    "weather_risk": "explanation string", "reasoning": "string", "prediction": "string",
    "robot_action": {"task": "string", "priority": "low|medium|high", "params": "dict"}
}

# 🚨 CHANGE: Use 'async def' to stay on the main event loop thread
@app.post("/diagnose")
async def generate_diagnostics(req: InferenceRequest):
    temp_img_path = None
    try:
        print(f"📸 Incoming request for {req.zone_id}...")
        
        # Download image
        img_response = requests.get(req.image_url, stream=True, timeout=30)
        img_response.raise_for_status()
        
        tf = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        for chunk in img_response.iter_content(chunk_size=8192):
            tf.write(chunk)
        tf.close()
        temp_img_path = tf.name

        system_prompt = (
            "You are the RoboLeaf Edge Perception Node. "
            f"The causal engine detected a potential anomaly in {req.zone_id}. "
            f"Context: P-Value={req.p_val:.4f}, VPD={req.vpd} kPa, Humidity={req.humidity}%, Temp={req.temp}C.\n\n"
            "=== ROBOLEAF PROTOCOL RULEBOOK ===\n"
            "- Apple Scab: Task='targeted_spray', Priority='high', Params={'product': 'captan_fungicide', 'boom_height_cm': 120}\n"
            "- Black Rot: Task='isolate', Priority='high', Params={'flag_location': true, 'schedule_pruning': true}\n"
            "- Cedar Apple Rust: Task='re-inspect', Priority='medium', Params={'re_inspect_hours': 48, 'drone_dispatch': true}\n"
            "- Healthy: Task='none', Priority='low', Params={'scan_interval_days': 7}\n\n"
            "=== VISUAL IDENTIFICATION GUIDE ===\n"
            "- Apple Scab: Look for velvety, olive-green to black spots with fuzzy, fringed margins.\n"
            "- Black Rot: Look for 'frog-eye' spots (brown centers with dark rings); may show concentric circles.\n"
            "- Cedar Apple Rust: Look for bright orange-yellow lesions; check for tiny black dots in the center.\n"
            "- Healthy: Look for uniform green texture with no necrotic spots or discolorations.\n\n"
            "Analyze the visual leaf data. Cross-reference it with the causal context and the rulebook.\n"
            "SCIENTIFIC GUIDELINES:\n"
            "1. If visual symptoms are absent, categorize as 'Healthy' (False Positive anomaly).\n"
            "2. If visual symptoms conflict with thermodynamic context (e.g. Scab in dry VPD), lower your 'confidence'.\n"
            "3. 'severity' must be one of: 'none', 'early', 'moderate', or 'severe'.\n"
            "4. 'weather_risk' must explain how VPD/Humidity influenced the diagnosis.\n\n"
            "Output ONLY a valid JSON payload matching this exact schema:\n"
            f"{json.dumps(SCHEMA_BLUEPRINT, indent=4)}\n"
            "Do not include any introductory text, markdown formatting, or explanations."
        )
        
        prompt = f"<s>[INST] {system_prompt}\n<image> [/INST]\n{{\n"
        
        print(f"🧠 NPU Inference starting for {req.zone_id}...")
        
        # The actual inference call
        output = generate(
            model=MODEL, processor=PROCESSOR, prompt=prompt,
            image=[temp_img_path], max_tokens=800, temperature=0.0
        )
        
        raw_text = output.text if hasattr(output, "text") else str(output)
        clean_json = raw_text.strip().removeprefix("```json").removesuffix("```").strip()
        if not clean_json.startswith("{"): clean_json = "{\n" + clean_json
            
        print(f"✅ Diagnosis Complete for {req.zone_id}")
        return {"status": "success", "json_payload": clean_json}
        
    except Exception as e:
        print(f"❌ ERROR DURING INFERENCE: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_img_path and os.path.exists(temp_img_path):
            os.remove(temp_img_path)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
