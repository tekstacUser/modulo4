from flask import Flask, request, jsonify
import requests
import os
import time

app = Flask(__name__)

OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "http://host.minikube.internal:11434"
)

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "phi3:mini"
)

CPU_WORK_MS = int(os.getenv("CPU_WORK_MS", "250"))


def cpu_work(milliseconds):
    """
    Small CPU workload used to make HPA behavior observable.
    """
    end_time = time.perf_counter() + (milliseconds / 1000.0)

    value = 0

    while time.perf_counter() < end_time:
        value = (value * 13 + 7) % 1000003

    return value


@app.get("/health")
def health():
    return jsonify({
        "status": "ok"
    })


@app.get("/ready")
def ready():
    return jsonify({
        "status": "ready"
    })


@app.post("/generate")
def generate():
    data = request.get_json(silent=True) or {}

    prompt = data.get("prompt")

    if not prompt:
        return jsonify({
            "error": "prompt is required"
        }), 400

    # Small CPU workload for HPA demonstration.
    cpu_work(CPU_WORK_MS)

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    }

    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json=payload,
            timeout=600
        )

        response.raise_for_status()

        ollama_response = response.json()

        return jsonify({
            "model": OLLAMA_MODEL,
            "response": ollama_response.get("response", "")
        })

    except requests.exceptions.RequestException as exc:
        return jsonify({
            "error": "Ollama request failed",
            "details": str(exc)
        }), 502


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=8070
    )
