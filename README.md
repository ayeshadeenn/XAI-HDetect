# XAI-HDetect

### A Post-Hoc Explainable Framework for Token-Level Hallucination Detection and Taxonomy Analysis in Vision-Language Outputs

XAI-HDetect is a post-hoc explainability framework for detecting and analysing hallucinations in Vision-Language Model (VLM) outputs at the **token level**.

The system operates on a **frozen LLaVA-1.5-7B model**, requiring no retraining or modification of the underlying VLM. It combines multimodal signals to identify potentially hallucinated tokens, classify their hallucination type, and provide explanations for the model's predictions.

---

## Overview

Given an image and a prompt, XAI-HDetect:

* Generates a response using LLaVA-1.5-7B
* Extracts multimodal features from the generated tokens
* Predicts token-level hallucination probabilities
* Groups consecutive hallucinated tokens into spans
* Classifies hallucinations into four categories:

  * **Object**
  * **Attribute**
  * **Relationship**
  * **Scene**
* Provides multimodal explanations using SHAP and Grad-CAM
* Estimates visual dependence through controlled input perturbation

### System Pipeline

```text
Image + Prompt
      │
      ▼
LLaVA-1.5-7B
(Frozen VLM)
      │
      ▼
Feature Extraction
Entropy • Log Probability • CLIP
Hidden States • Attention
      │
      ▼
Hallucination Detection
Calibrated LightGBM
      │
      ▼
Span Formation
      │
      ▼
Taxonomy Classification
Object • Attribute • Relationship • Scene
      │
      ▼
Explainability
SHAP • Grad-CAM • Visual Dependence
```

---

## Key Features

### Token-Level Detection

Identifies potentially hallucinated tokens rather than only assigning a hallucination score to an entire response.

### Hallucination Taxonomy

Categorises detected hallucinations into object, attribute, relationship, and scene-level errors.

### Multimodal Explanations

Combines:

* **SHAP** — feature-level contribution to the hallucination prediction
* **Grad-CAM** — visual regions associated with selected tokens
* **Visual dependence analysis** — comparison of model behaviour with and without visual input

### Calibrated Predictions

The hallucination detector uses probability calibration to produce more reliable token-level confidence estimates.

---

## Technology Stack

**Machine Learning**

* Python
* PyTorch
* LLaVA-1.5-7B
* CLIP
* LightGBM
* Sentence Transformers

**Explainability**

* SHAP
* Grad-CAM
* Controlled input perturbation

**Backend**

* FastAPI
* Python

**Frontend**

* React
* JavaScript

**Deployment**

* Modal
* Vercel

---

## Model & Dataset

The detection model operates on features extracted from a **frozen LLaVA-1.5-7B** model.

The detection model uses a combination of:

* Token entropy
* Token log probability
* Global and regional CLIP similarity
* LLaVA hidden-state representations
* Attention statistics

The hallucination detector was trained and evaluated using grouped splits by image to reduce image-level data leakage.

Taxonomy classification was performed separately across four hallucination categories:

| Category     | Description                                         |
| ------------ | --------------------------------------------------- |
| Object       | Incorrect or nonexistent objects                    |
| Attribute    | Incorrect properties such as colour, size, or count |
| Relationship | Incorrect relationships between entities            |
| Scene        | Incorrect scene-level descriptions                  |

---

## Performance

Evaluation was performed on a held-out grouped test split.

| Metric                     |      Score |
| -------------------------- | ---------: |
| AUROC                      | **0.8597** |
| F1 Score                   |  **0.636** |
| Expected Calibration Error | **0.0081** |

These results represent offline evaluation on the project's test data and should not be interpreted as guaranteed performance on arbitrary images or VLMs.

---

## Project Structure

```text
xai-hdetect/
│
├── xai-hdetect-backend/
│   ├── main.py
│   ├── modal_app.py
│   └── requirements.txt
│
├── xai-hdetect-frontend/
│   ├── src/
│   │   └── App.js
│   └── .env.example
│
└── README.md
```

---

## Running the Project

### Backend

Install the required dependencies:

```bash
cd xai-hdetect-backend
pip install -r requirements.txt
```

The backend exposes the analysis pipeline through FastAPI.

### Frontend

```bash
cd xai-hdetect-frontend
npm install
```

Create an environment file based on the provided template:

```bash
cp .env.example .env.local
```

Set the backend API URL:

```env
REACT_APP_API_URL=<your-backend-url>
```

Then start the frontend:

```bash
npm start
```

---

## API

### `POST /analyze`

Accepts an image and optional text prompt and returns the model's analysis.

The response includes:

* Generated caption
* Token-level hallucination probabilities
* Detected hallucination spans
* Taxonomy classifications
* SHAP feature contributions
* Visual dependence scores
* Grad-CAM visualization

### `GET /health`

Returns the current pipeline status.

---

## Limitations

* Currently evaluated on **LLaVA-1.5-7B** only.
* Taxonomy performance varies between categories, with relationship and scene hallucinations being more challenging.
* Domain generalisation beyond the evaluated datasets has not been extensively validated.
* Visual dependence analysis provides a heuristic measure and is **not formal causal inference**.
* Predictions should be interpreted as model-assisted analysis rather than definitive proof that a token is hallucinated.

---

## Academic Context

XAI-HDetect was developed as a **Final Year Project** investigating post-hoc explainability for hallucination detection in Vision-Language Models.

The project focuses on combining token-level detection, hallucination taxonomy, and multimodal explanations into a single analysis pipeline without modifying the underlying VLM.

---

## Author

**Ayesha Issadeen**

Final Year Undergraduate — IIT / University of Westminster

**Project:** XAI-HDetect
