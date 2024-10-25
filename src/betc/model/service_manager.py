"""
service_manager.py - Management of MLflow and TensorBoard services with proper path handling.
"""
import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import docker
import requests
from docker.errors import DockerException

from betc.model.config import ServiceMode
from betc.model.config import ServicesConfig

logger = logging.getLogger(__name__)


class ServiceManager:
    """Manages MLflow and TensorBoard services."""

    def __init__(self, config: ServicesConfig, project_dir: Optional[Path] = None):
        """
        Initialize service manager.

        Args:
            config: Services configuration
            project_dir: Project directory for local services
        """
        self.config = config
        self.project_dir = Path(project_dir or Path.cwd()).resolve()  # Get absolute path
        self.docker_client = None
        self.services_status = {}

    def _init_docker(self) -> None:
        """Initialize Docker client if needed."""
        if self.docker_client is None:
            try:
                self.docker_client = docker.from_env()
            except DockerException as e:
                logger.error(f"Failed to initialize Docker client: {e}")
                raise

    def _check_service_health(self, url: str, timeout: int = 30, interval: int = 2) -> bool:
        """
        Check if a service is healthy.

        Args:
            url: Service URL to check
            timeout: Maximum time to wait in seconds
            interval: Time between checks in seconds

        Returns:
            bool: True if service is healthy
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    return True
            except requests.RequestException:
                pass
            time.sleep(interval)
        return False

    def _create_directories(self) -> None:
        """Create necessary directories for local services."""
        # Create MLflow directories
        if self.config.mlflow.mode == ServiceMode.LOCAL:
            mlflow_dir = self.project_dir / "mlruns"
            mlflow_dir.mkdir(exist_ok=True)
            mlflow_db_dir = self.project_dir / "mlflow.db"
            if not mlflow_db_dir.exists():
                mlflow_db_dir.touch()

        # Create TensorBoard directories
        if self.config.tensorboard.mode == ServiceMode.LOCAL:
            tensorboard_dir = self.project_dir / "runs"
            tensorboard_dir.mkdir(exist_ok=True)

        logger.info(f"Created service directories in {self.project_dir}")

    def start_services(self) -> dict[str, bool]:
        """
        Start required services based on configuration.

        Returns:
            Dict[str, bool]: Status of each service
        """
        self._create_directories()

        # Start MLflow if needed
        if self.config.mlflow.mode == ServiceMode.LOCAL:
            self._init_docker()
            try:
                # Convert paths to absolute string paths
                mlruns_path = str(self.project_dir / "mlruns")
                mlflow_db_path = str(self.project_dir / "mlflow.db")

                # Define container name with timestamp to avoid conflicts
                container_name = f"mlflow_server_{int(time.time())}"

                mlflow_container = self.docker_client.containers.run(
                    "ghcr.io/mlflow/mlflow:v2.10.0",
                    command="mlflow server --host 0.0.0.0 --serve-artifacts",
                    ports={"5000/tcp": self.config.mlflow.port or 5000},
                    volumes={
                        mlruns_path: {"bind": "/mlruns", "mode": "rw"},
                        mlflow_db_path: {"bind": "/mlflow.db", "mode": "rw"},
                    },
                    environment={
                        "MLFLOW_TRACKING_URI": self.config.mlflow.tracking_uri or "sqlite:///mlflow.db"
                    },
                    detach=True,
                    remove=True,
                    name=container_name,
                )
                logger.info(f"Started MLflow container: {container_name}")
                self.services_status["mlflow"] = True

            except Exception as e:
                logger.error(f"Failed to start MLflow: {e}")
                self.services_status["mlflow"] = False

        # Start TensorBoard if needed
        if self.config.tensorboard.mode == ServiceMode.LOCAL:
            self._init_docker()
            try:
                # Convert path to absolute string path
                runs_path = str(self.project_dir / "runs")

                # Define container name with timestamp
                container_name = f"tensorboard_server_{int(time.time())}"

                tensorboard_container = self.docker_client.containers.run(
                    "tensorflow/tensorflow:2.14.0",
                    command="tensorboard --logdir /runs --bind_all --port 6006",
                    ports={"6006/tcp": self.config.tensorboard.port or 6006},
                    volumes={runs_path: {"bind": "/runs", "mode": "rw"}},
                    detach=True,
                    remove=True,
                    name=container_name,
                )
                logger.info(f"Started TensorBoard container: {container_name}")
                self.services_status["tensorboard"] = True

            except Exception as e:
                logger.error(f"Failed to start TensorBoard: {e}")
                self.services_status["tensorboard"] = False

        # Check remote services if configured
        if self.config.mlflow.mode == ServiceMode.REMOTE:
            mlflow_healthy = self._check_service_health(self.config.mlflow.uri)
            self.services_status["mlflow"] = mlflow_healthy

        if self.config.tensorboard.mode == ServiceMode.REMOTE:
            tensorboard_healthy = self._check_service_health(self.config.tensorboard.uri)
            self.services_status["tensorboard"] = tensorboard_healthy

        return self.services_status

    def stop_services(self) -> None:
        """Stop all running local services."""
        if self.docker_client is None:
            return

        try:
            for container in self.docker_client.containers.list():
                if container.name.startswith(("mlflow_server_", "tensorboard_server_")):
                    container.stop()
                    logger.info(f"Stopped container: {container.name}")
        except Exception as e:
            logger.error(f"Error stopping services: {e}")

    @contextmanager
    def service_context(self):
        """Context manager for service lifecycle."""
        try:
            self.start_services()
            yield self
        finally:
            self.stop_services()

    def get_mlflow_uri(self) -> Optional[str]:
        """Get MLflow tracking URI."""
        if self.config.mlflow.mode == ServiceMode.DISABLED:
            return None
        return self.config.mlflow.tracking_uri or self.config.mlflow.uri

    def get_tensorboard_logdir(self) -> Optional[str]:
        """Get TensorBoard log directory."""
        if self.config.tensorboard.mode == ServiceMode.DISABLED:
            return None
        return (
            str(self.project_dir / "runs")
            if self.config.tensorboard.mode == ServiceMode.LOCAL
            else self.config.tensorboard.logdir
        )
