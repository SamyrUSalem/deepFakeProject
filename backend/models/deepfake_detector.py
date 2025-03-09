import os
from models.deepfake_ai import predict_deepfake

def verify_deepfake(image_path):

    if not os.path.exists(image_path):
        return None, "Error: Image file not found"

    try:
        deepfake, score = predict_deepfake(image_path)

        if deepfake is None:
            return None, "Error: Unable to process the image"

        return deepfake, score
    
    except Exception as e:
        return None, f"Unexpected error: {str(e)}"
