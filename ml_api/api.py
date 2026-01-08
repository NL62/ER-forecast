import docker
from flask import Flask, jsonify
from time import sleep
import os

app = Flask(__name__)

# Initialize the Docker client
client = docker.from_env()

# Endpoint to trigger train docker
@app.route('/train_old', methods=['GET'])
def train_old():
    try:
        # Start training docker
        container = client.containers.run("er-forecast-training", detach=True)
        container_name = container.name

        sleep(10)

	# Check if the container is running
        container = client.containers.get(container_name)
        if container.status == 'running':
            return jsonify({"message": "Training started..."})
        else:
            return jsonify({"message": "Training failed to start!"}), 500
    except Exception as e:
        return jsonify({"message": f"Error starting training: {str(e)}"}), 500

@app.route('/predict_old', methods=['GET'])
def predict_old():
    try:
        container = client.containers.run("er-forecast-prediction", detach=True)
        container_name = container.name

        sleep(10)

        # Check if the container is running
        container = client.containers.get(container_name)
        if container.status == 'running':
            return jsonify({"message": "Prediction started..."})
        else:
            return jsonify({"message": "Prediction failed to start!"}), 500
    except Exception as e:
        return jsonify({"message": f"Error starting prediction: {str(e)}"}), 500


import logging
import subprocess

LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs.txt")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@app.route('/train', methods=['GET'])
def train():
    logger.info("Received /train request")

    # Run a command and capture stdout and stderr
    result = subprocess.run(
        ["docker", "compose", "-f", "../docker-compose.jobs.yml", "run", "training"],
        stdout=subprocess.PIPE,   # capture standard output
        stderr=subprocess.PIPE,   # capture error output
        text=True                 # decode bytes to string
    )


    logger.info("Training STDOUT:\n" + result.stdout)
    logger.error("Training STDERR:\n" + result.stderr)
    # Access output and errors
    print("STDOUT:")
    print(result.stdout)

    print("STDERR:")
    print(result.stderr)

    # Check exit code
    if result.returncode != 0:
        logger.error(f"Training command failed with exit code {result.returncode}")
        print(f"Command failed with exit code {result.returncode}")

    if result.returncode == 0:
        logger.info("Training completed successfully")
#        return jsonify({ "message": f"Command failed with exit code {result.returncode}"}, 500)

    return jsonify({"message": "Training completed!"}, 200)

@app.route('/predict', methods=['GET'])
def predict():
    logger.info("Received /predict request")

    result = subprocess.run(
        ["docker", "compose", "-f", "../docker-compose.jobs.yml", "run", "prediction"],
        stdout=subprocess.PIPE,   # capture standard output
        stderr=subprocess.PIPE,   # capture error output
        text=True                 # decode bytes to string
    )

    logger.info("Prediction STDOUT:\n" + result.stdout)
    logger.error("Prediction STDERR:\n" + result.stderr)

    print("STDOUT:")
    print(result.stdout)

    print("STDERR:")
    print(result.stderr)

    # Check exit code
    if result.returncode != 0:
        logger.error(f"Prediction command failed with exit code {result.returncode}")
        print(f"Command failed with exit code {result.returncode}")
 #       return jsonify({"message": f"Command failed with exit code {result.returncode}"}, 500)

    if result.returncode == 0:
        logger.info("Prediction completed successfully")

    return jsonify({"message": "Prediction completed!"}, 200)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
