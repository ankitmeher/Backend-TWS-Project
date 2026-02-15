# Buy Wait TWS Project (Backend)

This repository contains the **Buy Wait TWS Project** backend, built using FastAPI. The service provides prediction endpoints consumed by a frontend (not included here).

---

## Project Structure

```
Backend - TWS Project/
    FAST API/
        API.py
        Dockerfile
        requirements.txt
        __pycache__/
```

## Getting Started

### Prerequisites

- Python 3.10+
- Docker (optional, for containerization)

### Backend Setup

1. Navigate to the backend folder:
   ```bash
   cd "Backend - TWS Project/FAST API"
   ```
2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   # Windows PowerShell
   .\venv\Scripts\Activate.ps1
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the API:
   ```bash
   uvicorn API:app --reload
   ```
5. The API should be available at `http://127.0.0.1:8000`.

#### Docker (optional)

Build and run the container:
```bash
cd "Backend - TWS Project/FAST API"
docker build -t buywait-backend .
docker run -p 8000:8000 buywait-backend
```

### API Interaction

The backend exposes endpoints under `/` that return prediction results. Document these endpoints in `API.py` or in an OpenAPI spec if desired.

## Testing

*There are currently no automated tests configured.*

## Deployment

Describe any deployment steps here (e.g., deploying backend to a cloud service).

## Contributing

Feel free to submit pull requests or open issues. Follow the existing code style and include tests when available.

## License

Specify the license under which this project is distributed.
