# BETC: Bi-directional encoders for text classification

# "Betsy"

## Usage

```bash
# First time setup
make init

# Start services and run training
make train

# Monitor training
make services-logs

# Clean up after training
make services-stop
```

## Command Interface

```bash
# Show all available commands
make help

# Development commands
make init          # Initialize environment
make format        # Format code
make lint          # Run linting
make type-check    # Run type checking
make test          # Run tests
make coverage      # Generate coverage report

# Service management
make services-start    # Start MLflow and TensorBoard
make services-stop     # Stop services
make services-restart  # Restart services
make services-status   # Check service status
make services-logs     # View service logs

# Cleaning
make clean         # Clean Python cache
make docker-clean  # Clean Docker resources
make all-clean     # Clean everything
```

## Run Experimentation
```bash
rye run experiments
```
