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
    "cedar_apple_rust": get_images_by_state("rust")
}

ZONES = {
    "Zone_1_Fresno": {"lat": 36.7782, "lon": -119.4179},
    "Zone_2_Visalia": {"lat": 36.3302, "lon": -119.2921},
    "Zone_3_Bakersfield": {"lat": 35.3733, "lon": -119.0187},
    "Zone_4_Madera": {"lat": 36.9222, "lon": -120.0658},
    "Zone_5_Merced": {"lat": 37.3022, "lon": -120.4830}
}

def fetch_weather_panel(days=14) -> pd.DataFrame:
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

def map_biological_state(is_anomalous: bool, temp_c: float, humidity: float, precip: float) -> str:
    """Maps physical thermodynamics to biological image URLs."""
    
    # Helper to safely pick an image, or return a chill placeholder if the bucket is empty
    def safe_choice(state_key):
        images = IMAGE_STATE_SPACE.get(state_key, [])
        if images:
            return random.choice(images)
        else:
            print(f"WARNING: No images found for {state_key} in Supabase. Proceeding without image.")
            return None

    if not is_anomalous:
        return safe_choice("healthy")
    
    if temp_c < 20.0 and humidity > 80.0:
        return safe_choice("scab")
    elif temp_c > 25.0 and humidity > 75.0:
        return safe_choice("black_rot")
    elif precip > 0:
        return safe_choice("cedar_apple_rust")
    
    return safe_choice("scab") # Fallback