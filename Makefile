.PHONY: help up down restart logs logs-follow build clean test test-unit test-integration deploy-flows deploy-training deploy-predictions status ps

# Default target
help:
	@echo "ER Patient Forecast MLOps - Makefile Commands"
	@echo "=============================================="
	@echo ""
	@echo "Container Management:"
	@echo "  make up              - Start all services"
	@echo "  make down            - Stop all services"
	@echo "  make restart         - Restart all services"
	@echo "  make build           - Build all Docker images"
	@echo "  make logs            - View logs from all services"
	@echo "  make logs-follow     - Follow logs in real-time"
	@echo "  make status          - Show status of all containers"
	@echo "  make ps              - List running containers"
	@echo "  make clean           - Stop and remove all containers, networks, volumes"
	@echo ""
	@echo "Testing:"
	@echo "  make test            - Run all tests"
	@echo "  make test-unit       - Run unit tests only"
	@echo "  make test-integration - Run integration tests only"
	@echo ""
	@echo "Deployment:"
	@echo "  make deploy-flows    - Deploy all Prefect flows"
	@echo "  make deploy-training - Deploy training flow"
	@echo "  make deploy-predictions - Deploy prediction flow"
	@echo ""
	@echo "Utilities:"
	@echo "  make install         - Install Python dependencies with uv"
	@echo "  make format          - Format code with black"
	@echo "  make lint            - Lint code with ruff"
	@echo ""

# Container management
up:
	@echo "Starting all services..."
	docker-compose up -d
	@echo "Services started! Access points:"
	@echo "  - MLflow: http://localhost:5050"
	@echo "  - Prefect: http://localhost:4200"
	@echo "  - Grafana: http://localhost:3000"
	@echo "  - pgAdmin: http://localhost:16543"

down:
	@echo "Stopping all services..."
	docker-compose down

restart:
	@echo "Restarting all services..."
	docker-compose restart

build:
	@echo "Building all Docker images..."
	docker-compose build

logs:
	docker-compose logs

logs-follow:
	docker-compose logs -f

status:
	@echo "Service Status:"
	@docker-compose ps

ps:
	docker-compose ps

clean:
	@echo "WARNING: This will remove all containers, networks, and volumes!"
	@echo "Press Ctrl+C to cancel, or wait 5 seconds..."
	@sleep 5
	docker-compose down -v
	@echo "Cleanup complete"

# Testing
test:
	@echo "Running all tests..."
	uv run pytest tests/ -v --cov=src --cov-report=html --cov-report=term

test-unit:
	@echo "Running unit tests..."
	uv run pytest tests/unit/ -v

test-integration:
	@echo "Running integration tests..."
	uv run pytest tests/integration/ -v

# Deployment
deploy-flows: deploy-training deploy-predictions
	@echo "All flows deployed!"

deploy-training:
	@echo "Deploying training flow..."
	uv run python scripts/deploy_training_flow.py

deploy-predictions:
	@echo "Deploying prediction flow..."
	uv run python scripts/deploy_prediction_flow.py

# Utilities
install:
	@echo "Installing Python dependencies with uv..."
	uv sync

format:
	@echo "Formatting code with black..."
	uv run black src/ tests/ flows/

lint:
	@echo "Linting code with ruff..."
	uv run ruff check src/ tests/ flows/
