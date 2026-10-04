"""
DNA-QBio Platform: Production Flask Web Application Runner
Launches the Flask Web Dashboard on http://localhost:5000
"""
import sys
from pathlib import Path

# Configure utf-8 encoding for Windows console redirection
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flask_app.app import app

if __name__ == "__main__":
    print("=" * 70, flush=True)
    print("[DNA-QBio] Quantum Multi-Modal Fusion Platform (Flask Edition)", flush=True)
    print("=" * 70, flush=True)
    print("[LAUNCH] Starting Flask Web Dashboard...", flush=True)
    print("[URL] Running at: http://localhost:5000", flush=True)
    print("[PAGES]:", flush=True)
    print("   - Dashboard Home & 3D Lab:     http://localhost:5000/", flush=True)
    print("   - Unified Workflow Studio:     http://localhost:5000/workflow", flush=True)
    print("   - Biological & Quantum Theory: http://localhost:5000/concepts", flush=True)
    print("   - Genomic Data Studio:         http://localhost:5000/data-studio", flush=True)
    print("   - Tri-Engine & QMFN Modeling:  http://localhost:5000/models", flush=True)
    print("   - Live Mutation Inference:     http://localhost:5000/inference", flush=True)
    print("   - Quantum vs Classical Graphs: http://localhost:5000/visualizations", flush=True)
    print("   - Thesis Research Dossier:     http://localhost:5000/report", flush=True)
    print("[REST API]:", flush=True)
    print("   - POST /api/workflow/run", flush=True)
    print("   - GET  /api/workflow/dag", flush=True)
    print("   - POST /api/detect (Multi-Model Backend -> Quantum Best)", flush=True)
    print("   - POST /api/dataset/upload (FASTA/FASTQ/CSV)", flush=True)
    print("   - POST /api/dataset/curated/<key> (ClinVar Pan-Cancer/Hereditary)", flush=True)
    print("   - POST /api/dataset/fetch_live (Ensembl REST API)", flush=True)
    print("   - POST /api/dataset/train_test (Live In-Memory Training)", flush=True)
    print("   - GET  /api/dataset/active", flush=True)
    print("   - POST /api/classify", flush=True)
    print("   - POST /api/localize", flush=True)
    print("   - GET  /api/sample/<id>", flush=True)
    print("   - GET  /api/plot/<type>", flush=True)
    print("   - GET  /api/report", flush=True)
    print("=" * 70, flush=True)
    app.run(host="0.0.0.0", port=5000, debug=False)
