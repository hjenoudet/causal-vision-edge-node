import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from supabase import create_client, Client
import os 
import re 

# These should be in your .env file
url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_SERVICE_KEY") # That sb_secret_ key!
supabase: Client = create_client(url, key)

BUCKET_NAME = "PlantVillage-images"

def get_images_by_state(state_prefix):
    res = supabase.storage.from_(BUCKET_NAME).list(
        path='', 
        options={"search": f"apple__apple_{state_prefix}_"}
    )
    
    # Filter for only .JPG files
    files = [f for f in res if f['name'].endswith('.JPG')]

    # Smooth Sort: This sorts by the actual number at the end
    # It finds the digits (\d+) right before the .JPG
    files.sort(key=lambda x: int(re.search(r'_(\d+)\.JPG$', x['name']).group(1)) 
               if re.search(r'_(\d+)\.JPG$', x['name']) else 0)

    base_url = f"{os.environ.get('SUPABASE_URL')}/storage/v1/object/public/{BUCKET_NAME}"
    return [f"{base_url}/{f['name']}" for f in files]

# Now your space is fully automated and "chill"
IMAGE_STATE_SPACE = {
    "healthy": get_images_by_state("healthy"),
    "scab": get_images_by_state("scab"),
    "black_rot": get_images_by_state("black_rot"),
    "cedar_apple_rust": get_images_by_state("cedar_apple_rust")
}

ZONES = {
    "Zone_1_Fresno": {"lat": 36.7782, "lon": -119.4179},
    "Zone_2_Seattle": {"lat": 47.6062, "lon": -122.3321},
    "Zone_3_Bakersfield": {"lat": 35.3733, "lon": -119.0187},
    "Zone_4_Asheville": {"lat": 35.5951, "lon": -82.5515},
    "Zone_5_Charlottesville": {"lat": 38.0293, "lon": -78.4767},
    "Zone_6_Yakima": {"lat": 46.6021, "lon": -120.5059},
    "Zone_7_Wenatchee": {"lat": 47.4235, "lon": -120.3103},
    "Zone_8_Traverse_City": {"lat": 44.7631, "lon": -85.6206},
    "Zone_9_Hudson_Valley": {"lat": 41.7004, "lon": -73.9210},
    "Zone_10_Nagano": {"lat": 36.6485, "lon": 138.1947},
    "Zone_11_South_Tyrol": {"lat": 46.4983, "lon": 11.3548},
    "Zone_12_Elgin": {"lat": -34.1492, "lon": 19.0125},
    "Zone_13_Kent": {"lat": 51.2704, "lon": 0.5227},
    "Zone_14_Lerida": {"lat": 41.6176, "lon": 0.6229},
    "Zone_15_Hawkes_Bay": {"lat": -39.6394, "lon": 176.8392}
}

def fetch_weather_panel(days=30) -> pd.DataFrame:
    """Fetches real historical weather and calculates thermodynamic VPD."""
    start_str = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    end_str = datetime.now().strftime('%Y-%m-%d')
    
    data = []
    for zone_id, coords in ZONES.items():
        url = (f"https://api.open-meteo.com/v1/forecast?latitude={coords['lat']}"
               f"&longitude={coords['lon']}&past_days={days}&forecast_days=0"
               f"&daily=temperature_2m_mean,relative_humidity_2m_mean,precipitation_sum"
               f"&timezone=America%2FLos_Angeles")
        
        try:
            res = requests.get(url).json()
            daily = res.get('daily', {})
            if not daily: continue
            
            for i in range(len(daily.get('time', []))):
                t = daily['temperature_2m_mean'][i]
                h = daily['relative_humidity_2m_mean'][i]
                p = daily['precipitation_sum'][i]
                
                # Tetens Equation for VPD (kPa)
                svp = 0.6108 * np.exp((17.27 * t) / (t + 237.3))
                avp = svp * (h / 100.0)
                
                data.append({
                    "date": daily['time'][i], "zone_id": zone_id,
                    "temp_c": t, "humidity_pct": h, "precip_mm": p, "vpd_kpa": svp - avp
                })
        except Exception as e:
            print(f"Failed to fetch {zone_id}: {e}")
            
    return pd.DataFrame(data).dropna()

# ... (Keep all your imports, Supabase setup, BUCKET_NAME, and ZONES the same) ...

def map_biological_state(is_anomalous: bool, temp_c: float, humidity: float, precip: float) -> str:
    """Maps physical thermodynamics to biological image URLs."""
    
    # 🚨 FIX: Supabase-native fallbacks using the standard first image of your 12-image sets
    base_url = f"{os.environ.get('SUPABASE_URL')}/storage/v1/object/public/{BUCKET_NAME}"
    
    FALLBACK_IMAGES = {
        "healthy": f"{base_url}/apple__apple_healthy_1.JPG",
        "scab": f"{base_url}/apple__apple_scab_1.JPG",
        "black_rot": f"{base_url}/apple__apple_black_rot_1.JPG",
        "cedar_apple_rust": f"{base_url}/apple__apple_cedar_apple_rust_1.JPG" 
    }

    # Helper to safely pick an image, or use your guaranteed local fallback
    def safe_choice(state_key):
        images = IMAGE_STATE_SPACE.get(state_key, [])
        if images:
            return random.choice(images)
        else:
            print(f"⚠️ WARNING: Bucket fetch failed for '{state_key}'. Falling back to standard Supabase image: _1.JPG")
            return FALLBACK_IMAGES.get(state_key, FALLBACK_IMAGES["healthy"])

    if not is_anomalous:
        return safe_choice("healthy")
    
    if temp_c < 20.0 and humidity > 80.0:
        return safe_choice("scab")
    elif temp_c > 25.0 and humidity > 75.0:
        return safe_choice("black_rot")
    elif precip > 0:
        return safe_choice("cedar_apple_rust")
    
    # If anomalous but weather is dry, randomize disease to test VLM discrimination and break Scab bias
    return safe_choice(random.choice(["scab", "black_rot", "cedar_apple_rust"]))