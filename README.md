# Project Setup

## Backend (FastAPI)

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install fastapi uvicorn
uvicorn main:app --reload
```

## Frontend (Vite + React)

```bash
cd frontend
npm install
npm run dev
```

## Environment Notes

- Backend runs on `http://localhost:8000` by default.
- Frontend runs on `http://localhost:5173` by default.
- Sample data lives in `data_test/911.csv`.
