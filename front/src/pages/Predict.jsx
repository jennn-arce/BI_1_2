import React, { useState } from "react";

function Predict() {
    // Mode: "json", "form", or "upload"
    const [mode, setMode] = useState("json");

    // Preloaded JSON (shortened for brevity)
    const [jsonInput, setJsonInput] = useState(`[
  {
      "ID": "ID",
      "Titulo": "La mesa del congreso censura un encuentro internacional de parlamentarios prosáhara en el parlamento",
      "Descripcion": "Portavoces de Ciudadanos, PNV, UPN, PSOE, Unidos PP y EQUO denuncian juntos esta censura que consideran injustificable.",
      "Fecha": "30/10/2018"
  },
  {
      "ID": "ID",
      "Titulo": "La brecha digital que dificulta el acceso de ayudas a las personas vulnerables: 'Llega un momento en el que uno se rinde'",
      "Descripcion": "No es la primera vez que los ciudadanos vulnerables se topan con obstáculos a la hora de solicitar ayudas debido al lenguaje burocrático, el obligatoriedad de la Cl@ve Pin o la conexión a internet.",
      "Fecha": "15/03/2023"
  }
]`);

    // Form state: an array of objects, one per item
    const [formData, setFormData] = useState([
        { ID: "", Titulo: "", Descripcion: "", Fecha: "" },
    ]);

    // File upload state
    const [fileData, setFileData] = useState(null);

    const [predictions, setPredictions] = useState([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    // Add a new empty form
    const handleAddForm = () => {
        setFormData([...formData, { ID: "", Titulo: "", Descripcion: "", Fecha: "" }]);
    };

    // Handle change in any form field
    const handleFormChange = (index, field, value) => {
        const newData = [...formData];
        newData[index][field] = value;
        setFormData(newData);
    };

    // Handle file input change
    const handleFileChange = (e) => {
        const file = e.target.files[0];
        if (!file) return;
        const reader = new FileReader();
        reader.onload = (evt) => {
            setFileData(evt.target.result);
        };
        reader.onerror = () => {
            setError("Failed to read file");
        };
        reader.readAsText(file);
    };

    // On form submission, prepare the data based on the mode and send it to the endpoint
    const handleSubmit = async (e) => {
        e.preventDefault();
        setLoading(true);
        setError(null);

        let bodyData;
        if (mode === "json") {
            bodyData = jsonInput;
        } else if (mode === "form") {
            // Convert the form data to a JSON string
            bodyData = JSON.stringify(formData);
        } else if (mode === "upload") {
            // Use the fileData if available
            if (!fileData) {
                setError("Please select a file first");
                setLoading(false);
                return;
            }
            bodyData = fileData;
        }

        try {
            const response = await fetch("http://localhost:8000/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: bodyData,
            });
            if (!response.ok) {
                throw new Error("Network response was not ok");
            }
            const data = await response.json();
            setPredictions(data);
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="container mt-5">
            <h2 className="mb-4">Predict</h2>

            {/* Mode selection */}
            <div className="mb-4">
                <label className="me-3">
                    <input
                        type="radio"
                        name="mode"
                        value="json"
                        checked={mode === "json"}
                        onChange={() => setMode("json")}
                    />
                    Preloaded JSON
                </label>
                <label className="me-3">
                    <input
                        type="radio"
                        name="mode"
                        value="form"
                        checked={mode === "form"}
                        onChange={() => setMode("form")}
                    />
                    Form Input
                </label>
                <label>
                    <input
                        type="radio"
                        name="mode"
                        value="upload"
                        checked={mode === "upload"}
                        onChange={() => setMode("upload")}
                    />
                    Upload File
                </label>
            </div>

            <form onSubmit={handleSubmit}>
                {mode === "json" && (
                    <div className="mb-3">
                        <label htmlFor="jsonInput" className="form-label">
                            JSON Input
                        </label>
                        <textarea
                            id="jsonInput"
                            className="form-control"
                            rows="10"
                            value={jsonInput}
                            onChange={(e) => setJsonInput(e.target.value)}
                        ></textarea>
                    </div>
                )}

                {mode === "form" && (
                    <div>
                        {formData.map((item, index) => (
                            <div key={index} className="mb-4 border p-3 rounded">
                                <h5>Item {index + 1}</h5>
                                <div className="mb-3">
                                    <label className="form-label">ID</label>
                                    <input
                                        type="text"
                                        className="form-control"
                                        value={item.ID}
                                        onChange={(e) => handleFormChange(index, "ID", e.target.value)}
                                    />
                                </div>
                                <div className="mb-3">
                                    <label className="form-label">Titulo</label>
                                    <input
                                        type="text"
                                        className="form-control"
                                        value={item.Titulo}
                                        onChange={(e) => handleFormChange(index, "Titulo", e.target.value)}
                                    />
                                </div>
                                <div className="mb-3">
                                    <label className="form-label">Descripcion</label>
                                    <textarea
                                        className="form-control"
                                        rows="3"
                                        value={item.Descripcion}
                                        onChange={(e) =>
                                            handleFormChange(index, "Descripcion", e.target.value)
                                        }
                                    ></textarea>
                                </div>
                                <div className="mb-3">
                                    <label className="form-label">Fecha</label>
                                    <input
                                        type="text"
                                        className="form-control"
                                        value={item.Fecha}
                                        onChange={(e) => handleFormChange(index, "Fecha", e.target.value)}
                                    />
                                </div>
                            </div>
                        ))}
                        <button
                            type="button"
                            className="btn btn-secondary mb-3"
                            onClick={handleAddForm}
                        >
                            Add Another Item
                        </button>
                    </div>
                )}

                {mode === "upload" && (
                    <div className="mb-3">
                        <label htmlFor="fileUpload" className="form-label">
                            Upload JSON File
                        </label>
                        <input
                            id="fileUpload"
                            type="file"
                            accept=".json,application/json"
                            className="form-control"
                            onChange={handleFileChange}
                        />
                        {fileData && (
                            <small className="text-success">File loaded successfully.</small>
                        )}
                    </div>
                )}

                <button type="submit" className="btn btn-primary">
                    Upload Data
                </button>
            </form>

            {loading && <p className="mt-3">Loading...</p>}
            {error && <p className="mt-3 text-danger">Error: {error}</p>}

            {predictions.length > 0 && (
                <div className="mt-5">
                    <h3>Predictions</h3>
                    <div className="list-group">
                        {predictions.map((item, index) => (
                            <div key={index} className="list-group-item">
                                <p>
                                    <strong>Prediction:</strong> {item.prediction}
                                </p>
                                <p>
                                    <strong>Probability:</strong> {item.probability}
                                </p>
                            </div>
                        ))}
                    </div>
                </div>
            )}
        </div>
    );
}

export default Predict;
