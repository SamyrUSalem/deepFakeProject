from flask import Flask, request, jsonify
from flask_cors import CORS  
import os
from models.face_detection import detectar_face
from models.deepfake_detector import verificar_deepfake

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads/"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({"erro": "Nenhuma imagem enviada"}), 400

    file = request.files['file']
    caminho = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(caminho)

    if not detectar_face(caminho):
        return jsonify({"erro": "Nenhuma face detectada"}), 400

    deepfake, score = verificar_deepfake(caminho)

    return jsonify({"deepfake": deepfake, "score": round(score, 2)})  
    
if __name__ == '__main__':
    app.run(debug=True)
