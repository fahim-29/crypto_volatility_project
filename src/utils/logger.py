import logging
from pathlib import Path

def get_logger(name: str, log_file: str = "logs/project.log") -> logging.Logger:
    """Initialize and return a logger."""
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        fh = logging.FileHandler(log_path)
        ch = logging.StreamHandler()

        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger


if __name__ == "__main__":
    log = get_logger(__name__)
    log.info("Logger is working!")
