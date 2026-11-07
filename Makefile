.PHONY: help install test run-cli run-api docker-build docker-up docker-down clean

help:
	@echo "Available commands:"
	@echo "  make install      Install package and dependencies in editable mode"
	@echo "  make test         Run test suite with pytest"
	@echo "  make run-cli      Start interactive terminal CLI agent"
	@echo "  make run-api      Start FastAPI server at http://localhost:8000"
	@echo "  make docker-build Build Docker container image"
	@echo "  make docker-up    Start Docker Compose service in background"
	@echo "  make docker-down  Stop Docker Compose service"
	@echo "  make clean        Remove build artifacts and caches"

install:
	pip install -e .
	pip install fastapi uvicorn sse-starlette pytest pytest-mock

test:
	pytest -v

run-cli:
	python -m multitool_agent

run-api:
	uvicorn multitool_agent.api:app --reload --port 8000

docker-build:
	docker build -t multitool-agent:latest .

docker-up:
	docker compose up -d

docker-down:
	docker compose down

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	rm -rf build dist *.egg-info
