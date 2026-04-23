<div align="center">
  <h1>🛡️ SRDC: Semantics-Based Ransomware Detection and Classification</h1>
  <p><b>Advanced Zero-Day Ransomware Detection using LLM-assisted Pre-training (GPT-2 Semantic Analysis)</b></p>
</div>

## 📖 Overview

Welcome to the **SRDC Ransomware Detection System**. This project implements an advanced cybersecurity AI that identifies and classifies zero-day ransomware. Unlike traditional models that look for static signatures, SRDC analyzes the *semantics of sandbox behaviors* (API calls, file modifications, registry edits, and dropped files). 

By feeding system telemetry into a **fine-tuned GPT-2 Classification Model** (`zhouce/RDC-GPT`), it understands what a program is trying to do at a deeper semantic level, allowing it to accurately flag obfuscated and never-before-seen (zero-day) ransomware threats.

### Key Features
1. **Zero-Day Ransomware Detection**: Identifies whether unknown dynamic behavior is Goodware or Ransomware.
2. **Family Classification**: Once Ransomware is detected, the second model categorizes the threat into one of 12 known ransomware families (e.g., *CryptoWall, TeslaCrypt, Reveton, CryptLocker*).
3. **Interactive Demo**: Comes with a real-time sandbox simulation script (`finally demo/srdc_demo_fixed.py`) to visualize the model scanning API behaviors and making live predictions.

---

## 🛠️ Project Structure

- **`/finally demo/srdc_demo_fixed.py`**: An interactive sandbox simulation that loads up `.csv` sample data, runs the behavior text through the GPT-2 model, and provides a polished real-time console output mapping behavior to the correct ransomware family.
- **`/project/SRDC/`**: The core research code, including datasets, preprocessing steps, and raw `pytorch` AI models.

---

## 🚀 How to Run the Demo

The interactive demo will queue 3 samples (Ransomware and Goodware), parse their runtime behaviors (API features, extensions accessed, etc.), and process them live through the SRDC-GPT pipeline.

### Prerequisites

Ensure you have Python installed along with `PyTorch`, `transformers`, and `pandas`.

```bash
pip install torch transformers pandas
