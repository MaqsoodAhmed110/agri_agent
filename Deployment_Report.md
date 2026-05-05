# Industrial Packaging & Deployment Report — Agri-Agent

## 1. Reproducible Container Image
### Choice of Base Image: `python:3.11-slim`
We chose `python:3.11-slim` (Debian-based) over Alpine or the full Python image for several reasons:
- **Size Optimization**: At ~120MB, it is significantly smaller than the standard 900MB+ image.
- **Dependency Stability**: AI/ML libraries (like `psycopg` for Postgres and `faiss-cpu`) often require C-level bindings that are more stable on Debian/glibc than on Alpine's `musl`.
- **Security**: The "slim" variant contains fewer pre-installed utilities, reducing the potential attack surface.

### Layer Ordering Strategy
The `Dockerfile` is structured to maximize the use of Docker's layer caching:
1. **System Deps**: Installed first as they change rarely.
2. **Requirements**: `requirements.txt` is copied and `pip install` is run before copying any application code. This ensures that a single line change in `main.py` does not trigger a 5-minute dependency reinstall.
3. **App Code**: Copied last as it changes most frequently.

## 2. Secret-Free Image
### Zero-Secret Strategy
- **.dockerignore**: We explicitly exclude `.env` files and `env/` folders from being copied into the image.
- **Runtime Injection**: Secrets (like `GROQ_API_KEY` and `POSTGRES_PASSWORD`) are injected at runtime via Docker Compose using environment variables. This prevents API keys from being leaked via `docker history` or `docker save`.

## 3. Multi-Service Orchestration
### Orchestration Logic
We use **Docker Compose** to manage two core services:
1. **`agri-api`**: The FastAPI-based agent service.
2. **`agri-db`**: A Postgres 15 database that acts as the backing data store for conversation checkpoints.

### Discovery & Persistence
- **Discovery**: The API service connects to the database using the hostname `agri-db`, which is resolved automatically within the Docker bridge network.
- **Persistence**: 
    - **Postgres Data**: Stored in a named volume `db_data` to ensure conversation history survives container removals.
    - **Vector Index**: The FAISS index and PDF data are mounted via a bind mount from the local `./data` directory, allowing for external data management.

## 4. Evidence of Successful Deployment
- **Build Logs**: Can be generated using `docker compose build`.
- **Startup**: `agri-db` includes a healthcheck that ensures the database is ready before the `agri-api` attempts to connect.
- **Persistence Verification**: You can stop the containers (`docker compose down`) and restart them, and previous `thread_id` conversations will still be available in the Postgres database.
