# Variables
DOCKER_COMPOSE = docker compose
MLFLOW_PORT = 5000
TENSORBOARD_PORT = 6006
EXPERIMENT_DIR = experiments
RUNS_DIR = runs
MLRUNS_DIR = mlruns

.PHONY: init format lint type-check test coverage clean docker-check create-dirs \
        services-start services-stop services-restart services-status services-logs \
        docker-clean all-clean help

# Colors for help output
BLUE = \033[1;34m
YELLOW = \033[1;33m
GREEN = \033[1;32m
RED = \033[1;31m
NC = \033[0m # No Color

# Development Environment Setup
init: ## Initialize development environment
	rye sync
	pre-commit install
	@make create-dirs

# Code Quality
format: ## Format code using ruff and black
	rye run ruff format .
	rye run ruff check --fix .

lint: ## Run linting checks
	rye run ruff check .

type-check: ## Run type checking
	rye run mypy src tests

test: ## Run tests
	rye run pytest

coverage: ## Run tests with coverage report
	rye run pytest --cov=email_classifier --cov-report=html

# Cleaning
clean: ## Clean Python cache files
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type d -name ".pytest_cache" -exec rm -r {} +
	find . -type d -name ".ruff_cache" -exec rm -r {} +
	find . -type d -name ".mypy_cache" -exec rm -r {} +
	find . -type d -name "*.egg-info" -exec rm -r {} +
	find . -type f -name ".coverage" -delete
	find . -type d -name "htmlcov" -exec rm -r {} +
	find . -type f -name "coverage.xml" -delete

docker-clean: ## Clean Docker resources
	$(DOCKER_COMPOSE) down -v
	docker system prune -f

all-clean: clean docker-clean ## Clean everything (Python cache and Docker resources)

# Docker Service Management
docker-check: ## Check if Docker is running
	@echo "Checking Docker status..."
	@docker info > /dev/null 2>&1 || (echo "$(RED)Error: Docker is not running. Please start Docker and try again.$(NC)" && exit 1)

create-dirs: ## Create necessary directories
	@mkdir -p $(EXPERIMENT_DIR) $(RUNS_DIR) $(MLRUNS_DIR)
	@echo "$(GREEN)Created necessary directories$(NC)"

services-start: docker-check create-dirs ## Start MLflow and TensorBoard services
	@echo "$(BLUE)Starting MLflow and TensorBoard services...$(NC)"
	@$(DOCKER_COMPOSE) up -d
	@echo "$(YELLOW)Waiting for services to be ready...$(NC)"
	@for i in $$(seq 1 30); do \
		if curl -s http://localhost:$(MLFLOW_PORT) > /dev/null && \
		   curl -s http://localhost:$(TENSORBOARD_PORT) > /dev/null; then \
			echo "$(GREEN)Services are ready!$(NC)"; \
			echo "$(BLUE)MLflow UI: $(NC)http://localhost:$(MLFLOW_PORT)"; \
			echo "$(BLUE)TensorBoard: $(NC)http://localhost:$(TENSORBOARD_PORT)"; \
			exit 0; \
		fi; \
		echo "Attempt $$i/30: Waiting for services to start..."; \
		sleep 2; \
	done; \
	echo "$(RED)Error: Services did not start properly$(NC)"; \
	make services-stop; \
	exit 1

services-stop: docker-check ## Stop MLflow and TensorBoard services
	@echo "$(BLUE)Stopping MLflow and TensorBoard services...$(NC)"
	@$(DOCKER_COMPOSE) down
	@echo "$(GREEN)Services stopped$(NC)"

services-restart: ## Restart MLflow and TensorBoard services
	@make services-stop
	@sleep 2
	@make services-start

services-status: docker-check ## Check status of services
	@echo "$(BLUE)Checking service status...$(NC)"
	@$(DOCKER_COMPOSE) ps

services-logs: docker-check ## View service logs
	@$(DOCKER_COMPOSE) logs

# Training Workflow
train: services-start ## Run training workflow
	@echo "$(BLUE)Starting training workflow...$(NC)"
	python src/email_classifier/train.py
	@echo "$(GREEN)Training completed$(NC)"

# Help
help: ## Show this help message
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(BLUE)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "Examples:"
	@echo "  $(YELLOW)make init$(NC)          # Initialize the development environment"
	@echo "  $(YELLOW)make services-start$(NC) # Start MLflow and TensorBoard services"
	@echo "  $(YELLOW)make train$(NC)         # Run the training workflow"

# Default target
.DEFAULT_GOAL := help
