import React, { useState } from "react";
import axios from "axios";
import "bootstrap/dist/css/bootstrap.min.css";

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (event) => {
    const selectedFile = event.target.files[0];
    setFile(selectedFile);

    if (selectedFile) {
      setPreview(URL.createObjectURL(selectedFile));
    }
  };

  const handleUpload = async () => {
    if (!file) {
      alert("Please select an image before uploading.");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    setLoading(true);

    try {
      const response = await axios.post("http://127.0.0.1:5000/upload", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      console.log("API Response:", response.data);
      setResult(response.data);
    } catch (error) {
      console.error("Error uploading image:", error);
      alert("Error processing the image.");
    }

    setLoading(false);
  };

  return (
    <div className="container mt-5">
      <div className="card shadow p-4">
        <h1 className="text-center text-danger">Deepfake Detector</h1>

        <div className="form-group mt-3">
          <label className="form-label">Choose an image file</label>
          <div className="input-group">
            <input
              type="file"
              id="fileUpload"
              className="form-control d-none"
              onChange={handleFileChange}
              accept="image/*"
            />
            <label htmlFor="fileUpload" className="btn btn-outline-primary">
              Select File
            </label>
            <span className="form-control">{file ? file.name : "No file chosen"}</span>
          </div>
        </div>

        {preview && (
          <div className="text-center mt-3">
            <h5>Selected Image:</h5>
            <img
              src={preview}
              alt="Preview"
              className="img-fluid rounded shadow"
              style={{ maxWidth: "300px" }}
            />
          </div>
        )}

        <button className="btn btn-success w-100 mt-3" onClick={handleUpload} disabled={loading}>
          {loading ? "Analyzing..." : "Upload Image"}
        </button>

        {result && (
          <div className="mt-4 text-center">
            <h2>Result:</h2>
            {console.log("Rendered Data in React:", result)}
            <p>
              <strong>Deepfake:</strong>{" "}
              {result.deepfake ? (
                <span className="badge bg-danger ms-2">Yes</span>
              ) : (
                <span className="badge bg-success ms-2">No</span>
              )}
            </p>
            <p className="lead">
              <strong>Score:</strong> {result.score !== undefined ? result.score.toFixed(2) : "N/A"}
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
