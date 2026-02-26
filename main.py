from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI(title="Plant Disease Detection API")

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Class Names ───────────────────────────────────────────────────────────────
CLASS_NAMES = [
    'Apple__Apple_scab', 'Apple__Black_rot', 'Apple__Cedar_apple_rust', 'Apple__healthy',
    'Blueberry__healthy', 'Cherry_(including_sour)__Powdery_mildew', 'Cherry_(including_sour)__healthy',
    'Corn_(maize)__Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)__Common_rust',
    'Corn_(maize)__Northern_Leaf_Blight', 'Corn_(maize)__healthy',
    'Grape__Black_rot', 'Grape__Esca_(Black_Measles)', 'Grape__Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape__healthy', 'Orange__Haunglongbing_(Citrus_greening)', 'Peach__Bacterial_spot',
    'Peach__healthy', 'Pepper,_bell__Bacterial_spot', 'Pepper,_bell__healthy',
    'Potato__Early_blight', 'Potato__Late_blight', 'Potato__healthy',
    'Raspberry__healthy', 'Soybean__healthy', 'Squash__Powdery_mildew',
    'Strawberry__Leaf_scorch', 'Strawberry__healthy', 'Tomato__Bacterial_spot',
    'Tomato__Early_blight', 'Tomato__Late_blight', 'Tomato__Leaf_Mold',
    'Tomato__Septoria_leaf_spot', 'Tomato__Spider_mites Two-spotted_spider_mite',
    'Tomato__Target_Spot', 'Tomato__Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato__Tomato_mosaic_virus', 'Tomato__healthy'
]

# ── Disease Info ──────────────────────────────────────────────────────────────
DISEASE_INFO = {
    "Apple__Apple_scab": {"suggestion": "Remove infected leaves and apply fungicide spray. Avoid overhead watering.", "severity": "Moderate"},
    "Apple__Black_rot": {"suggestion": "Prune infected branches, remove mummified fruits, apply copper-based fungicide.", "severity": "High"},
    "Apple__Cedar_apple_rust": {"suggestion": "Apply fungicide early in the season. Remove nearby juniper trees if possible.", "severity": "Moderate"},
    "Cherry_(including_sour)__Powdery_mildew": {"suggestion": "Apply sulfur-based fungicide. Improve air circulation around the plant.", "severity": "Moderate"},
    "Corn_(maize)__Cercospora_leaf_spot Gray_leaf_spot": {"suggestion": "Use resistant varieties. Apply fungicide at early stages of infection.", "severity": "High"},
    "Corn_(maize)__Common_rust": {"suggestion": "Apply fungicide if infection is severe. Use rust-resistant hybrid seeds next season.", "severity": "Moderate"},
    "Corn_(maize)__Northern_Leaf_Blight": {"suggestion": "Apply fungicide and use resistant varieties. Rotate crops annually.", "severity": "High"},
    "Grape__Black_rot": {"suggestion": "Remove infected plant parts. Apply fungicide before and after flowering.", "severity": "High"},
    "Grape__Esca_(Black_Measles)": {"suggestion": "Prune infected wood. Protect pruning wounds with fungicide paste.", "severity": "High"},
    "Grape__Leaf_blight_(Isariopsis_Leaf_Spot)": {"suggestion": "Apply copper-based fungicide. Remove and destroy infected leaves.", "severity": "Moderate"},
    "Orange__Haunglongbing_(Citrus_greening)": {"suggestion": "No cure exists. Remove infected trees to prevent spread. Control psyllid insects.", "severity": "Critical"},
    "Peach__Bacterial_spot": {"suggestion": "Apply copper-based bactericide. Avoid wetting foliage during irrigation.", "severity": "Moderate"},
    "Pepper,_bell__Bacterial_spot": {"suggestion": "Use disease-free seeds. Apply copper bactericide spray regularly.", "severity": "Moderate"},
    "Potato__Early_blight": {"suggestion": "Apply fungicide and remove infected leaves. Avoid overhead irrigation.", "severity": "Moderate"},
    "Potato__Late_blight": {"suggestion": "Apply fungicide immediately. Destroy infected plants to prevent spread.", "severity": "Critical"},
    "Squash__Powdery_mildew": {"suggestion": "Apply potassium bicarbonate or neem oil spray. Improve air circulation.", "severity": "Moderate"},
    "Strawberry__Leaf_scorch": {"suggestion": "Remove infected leaves. Apply fungicide and avoid overcrowding plants.", "severity": "Moderate"},
    "Tomato__Bacterial_spot": {"suggestion": "Apply copper bactericide. Use certified disease-free seeds.", "severity": "Moderate"},
    "Tomato__Early_blight": {"suggestion": "Remove lower infected leaves. Apply fungicide and mulch around base.", "severity": "Moderate"},
    "Tomato__Late_blight": {"suggestion": "Apply fungicide immediately and remove infected plants. Very contagious.", "severity": "Critical"},
    "Tomato__Leaf_Mold": {"suggestion": "Improve ventilation. Apply fungicide and reduce humidity in greenhouse.", "severity": "Moderate"},
    "Tomato__Septoria_leaf_spot": {"suggestion": "Remove infected leaves. Apply fungicide and avoid wetting leaves.", "severity": "Moderate"},
    "Tomato__Spider_mites Two-spotted_spider_mite": {"suggestion": "Apply miticide or neem oil. Increase humidity around plants.", "severity": "Moderate"},
    "Tomato__Target_Spot": {"suggestion": "Apply fungicide and remove infected debris from soil.", "severity": "Moderate"},
    "Tomato__Tomato_Yellow_Leaf_Curl_Virus": {"suggestion": "No cure. Remove infected plants. Control whitefly population to prevent spread.", "severity": "Critical"},
    "Tomato__Tomato_mosaic_virus": {"suggestion": "Remove infected plants. Disinfect tools. Use virus-resistant varieties.", "severity": "High"},
}

SEVERITY_ICONS = {
    "Critical": "🔴",
    "High":     "🟠",
    "Moderate": "🟡",
    "Healthy":  "🟢"
}

EXPERT_DISCLAIMER = (
    "⚠️ This is an AI-based prediction and may not be 100% accurate. "
    "Please consult a certified agricultural expert or plant pathologist "
    "before taking any action on your crops."
)

# ── Load Model ────────────────────────────────────────────────────────────────
# compile=False skips loading optimizer state — fixes the add_slot error
MODEL_PATH = "models/plant_leaf_disease_detector"
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("✅ Model loaded successfully!")

# ── Helpers ───────────────────────────────────────────────────────────────────
def preprocess(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((256, 256))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, 0).astype(np.float32)


def format_result(label: str, confidence: float) -> dict:
    plant, condition = label.split("__") if "__" in label else (label, label)
    is_healthy = "healthy" in condition.lower()

    if is_healthy:
        return {
            "plant": plant.replace("_", " "),
            "condition": "Healthy",
            "is_healthy": True,
            "severity": "Healthy",
            "severity_icon": SEVERITY_ICONS["Healthy"],
            "confidence": round(float(confidence) * 100, 2),
            "suggestion": "Your plant looks healthy! Keep monitoring regularly and maintain good farming practices.",
            "disclaimer": EXPERT_DISCLAIMER
        }

    info = DISEASE_INFO.get(label, {
        "suggestion": "Monitor the plant closely and reduce moisture around leaves.",
        "severity": "Moderate"
    })
    severity = info["severity"]

    return {
        "plant": plant.replace("_", " "),
        "condition": condition.replace("_", " "),
        "is_healthy": False,
        "severity": severity,
        "severity_icon": SEVERITY_ICONS.get(severity, "🟡"),
        "confidence": round(float(confidence) * 100, 2),
        "suggestion": info["suggestion"],
        "disclaimer": EXPERT_DISCLAIMER
    }

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "Plant Disease Detection API is running 🌿"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Only JPG/PNG images are supported")

    image_bytes = await file.read()
    img_array = preprocess(image_bytes)

    predictions = model.predict(img_array)[0]

    top_indices = predictions.argsort()[-3:][::-1]

    return {
        "top_prediction": format_result(CLASS_NAMES[top_indices[0]], predictions[top_indices[0]]),
        "alternatives": [
            format_result(CLASS_NAMES[i], predictions[i]) for i in top_indices[1:]
        ]
    }