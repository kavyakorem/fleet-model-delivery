# Fleet Model Delivery

A local MLOps prototype for releasing a fault detector on a **simulated robot fleet**. The dashboard exposes the model quality checks, telemetry and rollout decisions so you can explain how a candidate becomes the active model.

### Run locally

Use Python 3.12 from the project root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m fleetops.train
python -m unittest discover -s tests -v
uvicorn fleetops.app:from_env --factory --host 127.0.0.1 --port 8000
```

On Windows, activate with `.venv\Scripts\Activate.ps1` in PowerShell. Open [the dashboard](http://127.0.0.1:8000/) or [the API docs](http://127.0.0.1:8000/docs). The ZIP also includes the trained ONNX artifacts and validation report, so retraining is optional for a demo.

1. Start the 40% canary. v1 remains active; v2 receives a deterministic subset of robot windows.
2. Try **Promote** before replaying data to see the evidence gate block it.
3. Click **Replay 96 windows**. If the canary still needs evidence, replay another batch. Labels come from fault injection.
4. Promote when the offline and online gates pass. Click **Roll back** to restore v1. The release history stays in SQLite.

An independent telemetry producer can also send individual inference requests:

```bash
python -m fleetops.replay --count 96
```

Optional Docker workflow:

```bash
docker compose up --build --wait
docker compose run --rm replay
```

The Compose setup provides a health-checked inference service, a persistent SQLite volume and a separate replay process. The HTTP port is bound to the local machine. The supplied GitHub Actions workflow retrains, runs the tests, checks the dashboard script, builds the container and runs an HTTP replay. The workflow is included for use after publishing the repository; it was not run on GitHub.

### How it works

- **Data and labels:** Each 24-sample window has current, vibration, temperature, speed and load. The simulator injects drag, excess vibration or overheating and records a binary fault label. Mean, variability, peak and trend features summarize each channel. Train, validation and test sets use disjoint robot IDs.
- **Training and export:** Train two NumPy neural classifiers, select thresholds on validation robots, then export normalization and the neural network to ONNX. The report tracks held-out precision, recall, F1, accuracy, export parity, model hashes and warmed CPU inference latency.
- **Serving and rollout:** FastAPI validates windows, runs ONNX Runtime and stores predictions, labels and inference latency in SQLite. A checksum check rejects damaged models at startup. Four offline gates check held-out F1, recall, ONNX parity and CPU p95. Canary promotion also requires 16 labeled windows, at least two faults, F1 of at least 0.85 and p95 of at most 30 ms. Canary requests also log the baseline model score on the same input.
- **Operations:** The dashboard shows gate results, recent predictions, alerts and release history. `/metrics` exposes Prometheus-format counters by model version.

### Verified result

| Release | Held-out windows | F1 | Recall | Max ONNX parity error |
| --- | ---: | ---: | ---: | ---: |
| v1 | 780 | 0.972 | 0.963 | 1.2 × 10⁻⁷ |
| v2 | 780 | 0.988 | 0.975 | 1.2 × 10⁻⁷ |

Seven tests passed, covering robot splits and parity, telemetry replay, malformed/duplicate inputs, offline gate rejection, checksum validation, canary promotion and rollback, and unlabeled evidence rejection. A live HTTP smoke test served the dashboard and exercised start, blocked early promotion, replay, promotion and rollback. The Docker runtime was not available for local verification. CPU inference latency measures model execution only; it does not include the API, storage or network.

This is a single-worker prototype with simulated inputs and labels. It does not establish performance or safety on physical robots. The release controls are intended for a local demo, without production authentication or infrastructure.
