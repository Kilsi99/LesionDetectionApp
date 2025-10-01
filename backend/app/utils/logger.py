import logging


logging.basicConfig(
    filename="logs/LesionApp.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)

def get_logger(name="LesionApp"):
    return logging.getLogger(name)