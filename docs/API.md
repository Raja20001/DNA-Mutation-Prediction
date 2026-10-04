# DNA Variant Intelligence Platform (DNA_v2) — REST API Documentation

## 1. Overview

The DNA_v2 platform exposes a secure, high-throughput RESTful API built on Flask. The central endpoint is **`POST /api/analyze`**, which orchestrates the complete 14-stage variant analysis pipeline and returns the full `VariantRecord` payload.

Base URL: `http://localhost:5000`

---

## 2. Master Analysis Endpoint: `POST /api/analyze`

Executes end-to-end variant intelligence including quality audit, alignment, feature extraction, multi-model consensus (Classical, TCN, Quantum, QMFN), conformal uncertainty, annotation, and ClinVar evidence retrieval.

### Request Headers
```http
Content-Type: application/json
```

### Request Body Schema
```json
{
  "sample_sequence": "ATGCGATCGATCGTTCGATCGATCGATCGATCGATC",
  "reference_sequence": "ATGCGATCGATCGATCGATCGATCGATCGATCGATC",
  "gene": "TP53",
  "chromosome": "chr17",
  "position": 7577120,
  "analysis_mode": "reference_aware"
}
```

#### Field Definitions
| Parameter | Type | Required | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `sample_sequence` | `string` | **Yes** | — | Query DNA sequence string (strict ACGT alphabet). |
| `reference_sequence` | `string` | No | `null` | Reference DNA sequence. If omitted, pipeline triggers `reference_blind` mode. |
| `gene` | `string` | No | `"TP53"` | HGNC gene symbol for coordinate and transcript resolution. |
| `chromosome` | `string` | No | `"chr17"` | Chromosomal identifier (e.g. `chr17`, `chr7`). |
| `position` | `integer` | No | `null` | 1-based genomic start coordinate. |
| `analysis_mode` | `string` | No | `"reference_aware"` | `"reference_aware"` or `"reference_blind"`. |

---

### Response Schema (200 OK)
```json
{
  "status": "SUCCESS",
  "execution_time_seconds": 0.428,
  "variant": {
    "variant_id": "VAR_TP53_7577120_C_T",
    "gene": "TP53",
    "chromosome": "chr17",
    "position": 7577120,
    "reference": "A",
    "alternate": "T",
    "mutation_type": "SNV",
    "mutation_subtype": "Transversion",
    "hgvs_c": "c.14A>T",
    "hgvs_p": "p.Lys5Ter",
    "consequence": "stop_gained"
  },
  "alignment": {
    "edit_distance": 1,
    "identity_percent": 97.2,
    "confidence": 0.985,
    "mutation_positions": [14]
  },
  "predictions": {
    "Random Forest": {
      "call": "MUTATION",
      "probability": 0.942,
      "family": "classical"
    },
    "Gradient Boosting": {
      "call": "MUTATION",
      "probability": 0.927,
      "family": "classical"
    },
    "TCN": {
      "call": "MUTATION",
      "probability": 0.884,
      "family": "deep_learning"
    },
    "Quantum Kernel": {
      "call": "MUTATION",
      "probability": 0.812,
      "family": "quantum"
    },
    "QMFN Hybrid": {
      "call": "MUTATION",
      "probability": 0.967,
      "family": "hybrid"
    }
  },
  "ensemble": {
    "consensus_prediction": "MUTATION",
    "consensus_probability": 0.941,
    "confidence": "HIGH",
    "model_agreement": "100.0%",
    "model_agreement_summary": "5/5 models agree",
    "is_disagreement_warning": false,
    "champion_model": "QMFN Hybrid"
  },
  "uncertainty": {
    "conformal_prediction_set": ["MUTATION"],
    "conformal_uncertainty_level": "LOW",
    "conformal_abstention": false,
    "alpha": 0.10
  },
  "explainability": {
    "top_contributing_features": [
      {"feature": "rknp_k3_novelty_fraction", "importance": 0.384},
      {"feature": "ref_similarity_jaccard", "importance": 0.245},
      {"feature": "gc_content", "importance": 0.178}
    ]
  },
  "annotation": {
    "assembly": "GRCh38",
    "transcript": "ENST00000269305.9",
    "consequence": "stop_gained"
  },
  "evidence": {
    "clinvar_association": "Li-Fraumeni Syndrome",
    "evidence_source": "NCBI ClinVar API",
    "evidence_provenance": "retrieved",
    "evidence_confidence": "HIGH"
  },
  "report": {
    "markdown_summary": "# DNA VARIANT RESEARCH DOSSIER..."
  },
  "limitations": {
    "disclaimer": "Experimental research framework. In-silico benchmark. Not validated for clinical patient diagnosis."
  }
}
```

---

## 3. Dedicated Modular Endpoints

### 3.1. Mutation Detection: `POST /api/detect`
Evaluates binary alteration probability using a specific model family.
```bash
curl -X POST http://localhost:5000/api/detect \
  -H "Content-Type: application/json" \
  -d '{"sequence": "ATGCGATCGATC", "model": "QMFN Hybrid", "threshold": 0.5}'
```

### 3.2. Mutation Classification: `POST /api/classify`
Classifies variant typology (SNV, Indel, Duplication, DELINS).
```bash
curl -X POST http://localhost:5000/api/classify \
  -H "Content-Type: application/json" \
  -d '{"reference": "A", "alternate": "G"}'
```

### 3.3. ClinVar Disease Query: `GET /api/clinvar/<gene>`
Queries NCBI ClinVar live endpoint with automated fallback to local validated cache.
```bash
curl http://localhost:5000/api/clinvar/TP53
```

### 3.4. Multi-Engine Batch Detection: `POST /api/detect_multi_model`
Executes parallel inference across Classical, Deep Learning, and Quantum models.
```bash
curl -X POST http://localhost:5000/api/detect_multi_model \
  -H "Content-Type: application/json" \
  -d '{"sequence": "ATGCGATCGATCGATCGATC", "gene": "TP53"}'
```

---

## 4. Error Handling & Status Codes

All errors return standardized structured JSON responses:

```json
{
  "status": "ERROR",
  "error_code": "INVALID_DNA_SEQUENCE",
  "message": "Sequence contains non-IUPAC nucleotide characters: ['X', '9'].",
  "timestamp": "2026-10-04T15:20:00Z"
}
```

| HTTP Code | Error Condition |
| :--- | :--- |
| `400 Bad Request` | Missing required parameters, sequence contains illegal non-nucleotide characters, sequence length below minimum threshold (10 bp). |
| `413 Payload Too Large` | Uploaded file or sequence payload exceeds 16MB `MAX_CONTENT_LENGTH`. |
| `422 Unprocessable Entity` | Reference and query sequence length disparity too severe for pairwise alignment (>50% difference). |
| `500 Internal Error` | Internal model execution failure or missing runtime checkpoint. |

---

## 5. Security Architecture

1. **File Upload Hardening:** Filenames are strictly sanitized via `werkzeug.utils.secure_filename()`.
2. **Memory Protection:** Maximum payload size capped at 16MB via Flask configuration.
3. **Alphabet Isolation:** Input sequences strictly checked against IUPAC DNA sets prior to memory allocation.
4. **Environment Isolation:** External NCBI ClinVar API keys, proxy parameters, and host ports managed via `.env` file.
