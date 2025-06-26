# SpiritRAG Backend

This is the backend for the SpiritRAG project, which provides APIs for document parsing, search, and response generation.

## Environment Configuration

### 1. **Install Python**
Ensure you have Python 3.8 or later installed on your system. You can check your Python version with:
```bash
python --version
```

### 2. **Set Up a Virtual Environment**
Create and activate a virtual environment to isolate dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Linux/Mac
venv\Scripts\activate     # On Windows
```

### 3. **Install Dependencies**
Install the required Python packages using `pip`:
```bash
pip install -r requirements.txt
```

### 4. **Install CUDA (Optional)**
If you want to use GPU acceleration, ensure you have CUDA installed and configured. The backend will automatically detect and use the GPU if available.

---

## Running the Backend

### 1. **Activate the Virtual Environment**
Before running the backend, activate the virtual environment:
```bash
source venv/bin/activate  # On Linux/Mac
venv\Scripts\activate     # On Windows
```

### 2. **Run the Backend**
Start the backend server by running `api-server.py`:
```bash
python api-server.py
```

The server will start and listen on `http://0.0.0.0:5000` by default.

---

## API Endpoints

### 1. **Parse API**
- **Endpoint**: `/api/parse`
- **Method**: `POST`
- **Description**: Parses and cleans the content of an uploaded PDF file.
- **Request**:
  - `file`: The PDF file to parse (multipart/form-data).
- **Response**:
  ```json
  {
    "parsed_content": "Raw parsed content",
    "cleaned_content": "Cleaned and processed content"
  }
  ```

### 2. **Search API**
- **Endpoint**: `/api/search`
- **Method**: `POST`
- **Description**: Searches for relevant documents based on a query.
- **Request**:
  ```json
  {
    "query": "Your search query",
    "dataset": "education" or "health"
  }
  ```
- **Response**:
  ```json
  {
    "results": ["Document 1", "Document 2", "Document 3"]
  }
  ```

### 3. **Generate API**
- **Endpoint**: `/api/generate`
- **Method**: `POST`
- **Description**: Generates a response based on the query and retrieved documents.
- **Request**:
  ```json
  {
    "query": "Your query",
    "retrieved_docs": ["Document 1", "Document 2"],
    "parsed_pdf": "Parsed content from the PDF"
  }
  ```
- **Response**:
  ```json
  {
    "generated_text": "Generated response text"
  }
  ```

---

## Notes

- **CORS**: The backend is configured to allow requests from `http://localhost:5173` (frontend development server). Update the CORS configuration in `api-server.py` if needed.
- **GPU Support**: The backend will use GPU if CUDA is available. Otherwise, it will fall back to CPU.

---

## Troubleshooting

### 1. **CORS Issues**
If the frontend cannot communicate with the backend due to CORS errors, ensure the `CORS` configuration in `api-server.py` is correct:
```python
CORS(app, resources={r"/*": {"origins": ["http://localhost:5173", "http://127.0.0.1:5173"]}})
```

### 2. **Port Conflicts**
If port `5000` is already in use, modify the `app.run()` call in `api-server.py` to use a different port:
```python
app.run(host='0.0.0.0', port=5001)
```

### 3. **CUDA Issues**
If CUDA is not installed or configured correctly, the backend will fall back to CPU. Ensure your environment supports GPU acceleration if required.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.