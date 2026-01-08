from flask import Flask, jsonify
import subprocess

app = Flask(__name__)

@app.get("/train")
def train():
	try:
		cmd = [
			"docker", "compose",
			"-f", "docker-compose.jobs.yml",
			"run", "--rm", "training"
		]

		result = subprocess.run(cmd, capture_output=True, text=True, check=True)

		return jsonify({
			"status": "sucess",
			"message": "Training completed",
			"error": result.stdout,
			"returncode": result.returncode
		})

	except subprocess.CalledProcessError as e:
		return jsonify({
			"status": "error",
			"message": "training job failed.",
			"output": e.stdout,
			"error": e.stderr,
		}), 500


@app.get("/predict")
def predict():
	try:
		cmd = [
			"docker", "compose",
			"-f", "docker-compose.jobs.yml",
			"run", "--rm", "training"
		]

		result = subprocess.run(cmd, capture_output=True, text=True, check=True)

		return jsonify({
			"status": "success",
			"stdout": result.stdout,
			"stderr": result.stderr,
			"returncode": result.returncode
		})

	except subprocess.CalledProcessError as e:
		return jsonify({
			"status": "error",
			"stdout": e.stdout,
			"stderr": e.stderr,
			"returncode": e.returncode
		}), 500

if __name__ == "__main__":
	app.run(host="0.0.0.0", port=5000)
