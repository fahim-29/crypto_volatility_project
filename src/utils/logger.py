import logging
from pathlib import Path


def get_logger(name: str, log_file: str = "logs/project.log") -> logging.Logger:
    """Creates and configures a logger that writes to both console and file."""
    Path("logs").mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers if logger already exists
    if not logger.handlers:
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # File handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        ch.setFormatter(formatter)
        fh.setFormatter(formatter)

        # Add handlers
        logger.addHandler(ch)
        logger.addHandler(fh)

    return logger


if __name__ == "__main__":
    logger = get_logger("test_logger")
    print("🔹 Running logger test...")
    logger.info("Logger is working!")
<<<<<<< HEAD


=======
>>>>>>> 89da243 (Updated prediction pipeline and logger)
