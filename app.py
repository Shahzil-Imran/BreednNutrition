import os
import io
import json
import uvicorn
from typing import Annotated
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from groq import Groq
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import inference logic from your TensorFlow model loader
from model_loader import predict_breed

app = FastAPI(
    title="Smart Paws AI Backend",
    description="Breed Identification and Personalized Nutrition API",
    version="1.0.0"
)

# Initialize Groq client
try:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("WARNING: GROQ_API_KEY not found in environment variables.")
        groq_client = None
    else:
        groq_client = Groq(api_key=api_key)
        print("Groq Client initialized successfully.")
except Exception as e:
    groq_client = None
    print(f"ERROR: Failed to initialize Groq client: {e}")

# --- DATA MODELS ---
class PetDetails(BaseModel):
    age_months: str
    weight_kg: str

# --- NUTRITION GENERATOR ---
def generate_nutrition_plan(breed: str, details: PetDetails) -> dict:
    """Generates a structured nutrition plan using Groq matching the Flutter UI."""
    if groq_client is None:
        return {
            "daily_calories": "N/A",
            "protein_requirements": "N/A",
            "recommended_food": "N/A",
            "feeding_schedule": "N/A"
        }
    
    # SYSTEM PROMPT: Enforces the exact JSON keys for your UI icons
    system_prompt = (
        "You are an expert veterinary nutritionist. "
        "Analyze the dog breed and details provided. "
        "Respond ONLY with a raw JSON object (no markdown, no explanations). "
        "You MUST use this exact schema: "
        "{"
        "  \"daily_calories\": \"String (e.g., '1,200 - 1,500 kcal/day')\", "
        "  \"protein_requirements\": \"String (e.g., '22-25% of daily intake')\", "
        "  \"recommended_food\": \"String (Short recommendation, e.g., 'High-quality dry kibble','meat','fish')\", "
        "  \"feeding_schedule\": \"String (e.g., '2-3 meals per day')\" "
        "}"
    )

    user_prompt = (
        f"Create a nutrition plan for a {breed}. "
        f"Age: {details.age_months} months, Weight: {details.weight_kg}kg."
    )

    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model="llama-3.3-70b-versatile",
            response_format={"type": "json_object"} 
        )
        return json.loads(chat_completion.choices[0].message.content)
    except Exception as e:
        print(f"Groq Error: {e}")
        return {
            "daily_calories": "Error",
            "protein_requirements": "Error",
            "recommended_food": "Error",
            "feeding_schedule": "Error"
        }

# --- API ENDPOINT ---
@app.post("/analyze-pet")
async def analyze_pet(
    file: Annotated[UploadFile, File()],
    age_months: Annotated[str, Form()],
    weight_kg: Annotated[str, Form()]
):
    try:
        image_content = await file.read()
        image_stream = io.BytesIO(image_content)

        # Step 1: Predict Breed (Returns name and float confidence)
        breed_name, confidence_float = predict_breed(image_stream)
        
        # Format confidence for UI (e.g., 0.982 -> "98% Match")
        # Handle edge cases where confidence might be 0.0
        if confidence_float > 0:
            confidence_str = f"{int(confidence_float * 100)}% Match"
        else:
            confidence_str = "Unknown"

        # Step 2: Get Nutrition Plan
        pet_info = PetDetails(
            age_months=age_months, 
            weight_kg=weight_kg
        )
        nutrition_data = generate_nutrition_plan(breed_name, pet_info)

        # Step 3: Return Structure matching Flutter 'PetDietPlan' model
        return {
            "breed_name": breed_name,
            "confidence_score": confidence_str,
            "nutrition_plan": nutrition_data
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "detail": str(e)}
        )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)