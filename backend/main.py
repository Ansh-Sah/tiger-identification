import io
import os
import json

import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image

from fastapi import (
    FastAPI,
    File,
    UploadFile
)

from fastapi.middleware.cors import CORSMiddleware

from torchvision import transforms

from model import TigerEmbeddingModel


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "tiger_embedding_model.pth"
)

DATABASE_PATH = os.path.join(
    MODEL_DIR,
    "tiger_database.npz"
)

CONFIG_PATH = os.path.join(
    MODEL_DIR,
    "config.json"
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("========================================")
print("Tiger Identification Backend")
print("========================================")
print(f"Device: {device}")


# ============================================================
# CHECK MODEL FILES
# ============================================================

print("\nChecking model files...")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file not found:\n{MODEL_PATH}"
    )

if not os.path.exists(DATABASE_PATH):
    raise FileNotFoundError(
        f"Database file not found:\n{DATABASE_PATH}"
    )

print("✓ tiger_embedding_model.pth found")
print("✓ tiger_database.npz found")

if os.path.exists(CONFIG_PATH):
    print("✓ config.json found")
else:
    print("⚠ config.json not found, using default settings")


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

print("\nLoading tiger model...")

model = TigerEmbeddingModel()

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)


# Support both possible saving formats
if (
    isinstance(checkpoint, dict)
    and "model_state_dict" in checkpoint
):

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

else:

    model.load_state_dict(
        checkpoint
    )


model.to(device)

# IMPORTANT:
# This is inference only.
# No training happens here.
model.eval()

print("✓ Tiger embedding model loaded")


# ============================================================
# LOAD TIGER DATABASE
# ============================================================

print("\nLoading tiger database...")

database = np.load(
    DATABASE_PATH,
    allow_pickle=True
)

prototype_embeddings = database[
    "prototype_embeddings"
]

tiger_ids = database[
    "tiger_ids"
]

# Convert NumPy embeddings to PyTorch tensor
prototype_embeddings = torch.tensor(
    prototype_embeddings,
    dtype=torch.float32,
    device=device
)

# Normalize database embeddings
prototype_embeddings = F.normalize(
    prototype_embeddings,
    p=2,
    dim=1
)

# Convert IDs to strings
tiger_ids = [
    str(tiger_id)
    for tiger_id in tiger_ids
]

print("✓ Tiger database loaded")

print("\nKnown tigers:")

for tiger in tiger_ids:
    print(f"  • {tiger}")


# ============================================================
# CONFIDENCE THRESHOLD
# ============================================================

confidence_threshold = 0.55


if os.path.exists(CONFIG_PATH):

    try:

        with open(
            CONFIG_PATH,
            "r"
        ) as file:

            config = json.load(file)

        confidence_threshold = float(
            config.get(
                "confidence_threshold",
                0.55
            )
        )

    except Exception as error:

        print(
            f"⚠ Could not read config.json: {error}"
        )


print(
    f"\nConfidence threshold: "
    f"{confidence_threshold}"
)


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

transform = transforms.Compose([

    transforms.Resize(
        (224, 224)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],

        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(

    title="Tiger Identification API",

    description=(
        "AI-based individual tiger "
        "identification system"
    ),

    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(

    CORSMiddleware,

    allow_origins=["*"],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"]
)


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():

    return {

        "status": "online",

        "message": (
            "Tiger Identification API "
            "is running"
        )
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {

        "status": "healthy",

        "model_loaded": True,

        "device": str(device),

        "known_tigers": len(tiger_ids)
    }


# ============================================================
# TIGER IDENTIFICATION
# ============================================================

@app.post("/identify")
async def identify_tiger(
    file: UploadFile = File(...)
):

    try:

        # ----------------------------------------------------
        # 1. Read uploaded image
        # ----------------------------------------------------

        image_bytes = await file.read()

        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")


        # ----------------------------------------------------
        # 2. Preprocess image
        # ----------------------------------------------------

        image_tensor = transform(
            image
        )

        image_tensor = image_tensor.unsqueeze(
            0
        )

        image_tensor = image_tensor.to(
            device
        )


        # ----------------------------------------------------
        # 3. Generate tiger embedding
        # ----------------------------------------------------

        with torch.no_grad():

            embedding = model(
                image_tensor
            )

            embedding = F.normalize(
                embedding,
                p=2,
                dim=1
            )


        # ----------------------------------------------------
        # 4. Compare with tiger database
        # ----------------------------------------------------

        similarities = torch.matmul(
            embedding,
            prototype_embeddings.T
        )

        similarities = similarities[0]


        # ----------------------------------------------------
        # 5. Find best match
        # ----------------------------------------------------

        best_index = torch.argmax(
            similarities
        ).item()

        best_score = similarities[
            best_index
        ].item()

        best_tiger = tiger_ids[
            best_index
        ]


        # ----------------------------------------------------
        # 6. Check unknown threshold
        # ----------------------------------------------------

        confidence = best_score * 100


        if best_score < confidence_threshold:

            return {

                "success": True,

                "identity": "Unknown",

                "tiger_id": None,

                "confidence": round(
                    confidence,
                    2
                ),

                "message": (
                    "Tiger identity "
                    "could not be confirmed."
                )
            }


        # ----------------------------------------------------
        # 7. Return identified tiger
        # ----------------------------------------------------

        return {

            "success": True,

            "identity": best_tiger,

            "tiger_id": best_tiger,

            "confidence": round(
                confidence,
                2
            ),

            "message": (
                "Tiger identified "
                "successfully."
            )
        }


    except Exception as error:

        print(
            f"Identification error: {error}"
        )

        return {

            "success": False,

            "identity": None,

            "tiger_id": None,

            "confidence": 0,

            "error": str(error)
        }