# 🛡️ SRDC: Semantics-Based Ransomware Detection and Classification with LLM-Assisted Pre-Training

[![AAAI 2025](https://img.shields.io/badge/AAAI%202025-Accepted-brightgreen.svg)](https://aaai.org/)
[![HuggingFace Model](https://img.shields.io/badge/HuggingFace-zhouce%2FRDC--GPT-orange.svg)](https://huggingface.co/zhouce/RDC-GPT)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Framework-PyTorch](https://img.shields.io/badge/Framework-PyTorch-red.svg)](https://pytorch.org/)

An advanced, production-grade cybersecurity AI system designed to detect and classify **Zero-Day Ransomware** in real-time. Instead of relying on fragile, static signatures, **SRDC** analyzes the core **semantics of sandbox behaviors** (API calls, file modifications, registry edits, and memory operations) using a fine-tuned GPT-2 classification model.

This project bridges academic research (accepted at **AAAI 2025**) and practical defense systems, featuring an interactive sandbox CLI simulation, a local Flask scanning backend, and a real-time Chrome Extension client.

---

## 📖 Overview & Architecture

Modern ransomware continuously obfuscates its binary code, rendering traditional antivirus signatures obsolete. **SRDC** resolves this by translating system execution sequences into natural language-like behavior representations and passing them into a cognitive pre-trained LLM (`zhouce/RDC-GPT`).

```
                              [ Download Interception ]
                                         │
                                         ▼
                             [ SRDC Shield Extension ]
                                         │  (Forward Download URL)
                                         ▼
                        [ srdc_shield_backend.py (Flask) ]
                                         │
                    ┌────────────────────┴────────────────────┐
                    ▼ (Layer 1)                               ▼ (Layer 2)
           [ VirusTotal Scan ]                      [ SRDC Cognitive AI Scan ]
          (Known Hash Check)                   (Statically Extracted PE Semantics)
                                                              │
                                            ┌─────────────────┴─────────────────┐
                                            ▼ (Step A)                          ▼ (Step B)
                                 [ Zero-Day Detection ]               [ Family Classification ]
                                  (Goodware vs Ransomware)             (Identifies 12 Families)
```

### Key System Components
1. **Zero-Day Ransomware Detection (Layer 2A)**: Evaluates whether unknown program behavior exhibits malicious ransomware traits (98.7%+ confidence).
2. **Malware Family Classification (Layer 2B)**: Classifies flagged threats into 12 distinct ransomware families (e.g., *CryptoWall, TeslaCrypt, Reveton, CryptLocker*).
3. **Cognitive Flask Backend (`srdc_shield_backend.py`)**: Preloads the pre-trained weights (`srdc_zero_day_BEST.pth` & `srdc_family_BEST.pth`) on CPU startup for rapid execution, providing full API, Registry, and String parsing for PE files (and `.zip` archives).
4. **SRDC Shield Extension (`srdc-shield-extension/`)**: A Chrome extension frontend client that intercepts browser downloads and queries the local AI engine for safe-allow/block verdicts before files can execute on your system.
5. **Interactive Console Demo (`finally_demo/`)**: A rich, standalone CLI simulation showcasing how telemetry is loaded, tokenized, evaluated by the AI, and contextualized.

---

## 🛠️ Project Structure

```directory
DOMAIN-PRO-SRDC/
├── srdc_shield_backend.py          # Flask backend, preloads models, extracts PE semantics
├── srdc-shield-extension/           # Chrome Extension frontend (HTML, CSS, JS)
│   ├── background.js               # Download interception, talks to Flask backend
│   ├── popup.html / popup.js       # Extension popup UI & configuration
│   └── popup.css                   # Modern dark-mode Glassmorphism CSS UI
├── finally_demo/                   # Interactive Sandbox Simulation Suite
│   ├── srdc_demo_fixed.py          # Main CLI simulation with rich terminal UI
│   ├── srdc_custom_test.py         # Test custom behavior strings/CSV rows
│   ├── srdc_family_test.py         # Standalone test for family classification
│   └── custom_samples.csv          # Example behavioral datasets
├── project/SRDC/                   # Core research code, notebook, and offline scripts
│   ├── result/                     # Put your fine-tuned model .pth files here
│   ├── FineTuned_SRDC.ipynb        # Jupyter notebook used for fine-tuning
│   └── fix_splits.py               # Data splits manager
├── README.md                       # Comprehensive documentation (this file)
└── .gitignore                      # Configured to ignore models (.pth), caches, and envs
```

---

## 🚀 Getting Started

### 1. Installation & Environment Setup

Clone the repository and install the required machine learning and web server dependencies:

```bash
pip install torch transformers pandas flask flask-cors pefile requests
```

*Note: Make sure your fine-tuned model files (`srdc_zero_day_BEST.pth` and `srdc_family_BEST.pth`) are placed in the `project/SRDC/result/` directory as configured in the backend.*

---

### 2. Running the Interactive Sandbox CLI Demo

The standalone console demo loads sample telemetry data from `finally_demo/custom_samples.csv`, processes it through the models, and showcases a real-time behavioral audit.

```bash
# Navigate to the demo directory
cd finally_demo

# Run the interactive simulator
python srdc_demo_fixed.py
```

#### Expected CLI Output Example:
```text
============================================================
   SRDC Ransomware Detection System 🛡️
   Powered by GPT-2 Semantic Analysis
============================================================

[*] Sample 1/3 entering sandbox...

[*] Captured API behavior (preview):
    → LdrLoadDll LdrGetProcedureAddress NtCreateSection NtMapViewOfSection...
[*] Registry activity:
    → REG_OPEN_KEY REG_QUERY_VALUE...

[*] Feeding behavior into SRDC-GPT model...
[*] Running semantic analysis...

⚠️  ══════════════════════════════════════
🚨  RANSOMWARE DETECTED!
    Confidence : 98.7%
⚠️  ══════════════════════════════════════

[*] Running Family Classification...
🔍  Family Identified : CryptoWall
    Confidence        : 95.2%
    True Family       : CryptoWall
    Result            : ✅ CORRECT

🛑  ACTION: ISOLATE SYSTEM IMMEDIATELY
    Threat: CryptoWall ransomware confirmed.
```

---

### 3. Deploying the Scanning Backend & Chrome Extension

#### Step A: Launch the Local Flask Server
The Flask server runs on port `5000` and preloads the PyTorch weights so that scanning takes fraction-of-a-second times.

```bash
# From the root project directory
python srdc_shield_backend.py
```

*The console will print out success messages confirming models have successfully loaded onto CPU memory and that the server is ready at `http://127.0.0.1:5000`.*

#### Step B: Install the Chrome Extension
1. Open Google Chrome and navigate to `chrome://extensions/`.
2. Toggle on **Developer mode** in the top right-hand corner.
3. Click on **Load unpacked** in the top left-hand corner.
4. Select the `srdc-shield-extension` folder from this project directory.
5. *Success!* You will see the **SRDC Shield** icon in your toolbar.
6. Click the extension icon to add your **VirusTotal API Key** (optional) and toggle **Strict Mode**. The extension will now automatically block malicious downloads in real-time.

---

## 📊 Experimental Evaluation (AAAI 2025 Paper)

We evaluated our fine-tuned model against state-of-the-art general-purpose LLMs (via APIs) using Zero-Day detection datasets. General-purpose LLMs rely on generic reasoning, which triggers high rates of hallucination, whereas **SRDC's** localized training yields rapid and highly precise threat blocking.

### 1. Zero-Day Ransomware Detection Benchmarks
| Method | Accuracy | Recall | F1-Score | Inference Speed (Sec/Sample) |
| :--- | :---: | :---: | :---: | :---: |
| **gpt-4-turbo** | 0.4950 | 0.1000 | 0.1653 | 0.786s |
| **claude-3.5-sonnet** | 0.4700 | 0.6600 | 0.5546 | 5.960s |
| **SRDC (Ours - GPT-2 based)** | **0.8860** | **0.9160** | **0.9130** | **0.0866s** |

### 2. Ransomware Family Classification Benchmarks
| Method | Balanced Accuracy | Inference Speed (Sec/Sample) |
| :--- | :---: | :---: |
| **gpt-4-turbo** | 0.1075 | 0.808s |
| **claude-3.5-sonnet** | 0.1045 | 6.070s |
| **SRDC (Ours - GPT-2 based)** | **0.5483** | **0.0836s** |

---

## 📝 LATAP Training Data Generation

The custom semantic corpus was generated using a customized LLM prompts strategy (detailed in `project/SRDC/Pretraining_Corpus/PromptDesign.md`).

| Corpus Source | Manually Collected | Generated by GPT-3.5 Turbo |
| :--- | :---: | :---: |
| Windows System APIs | 230 | 1,903 |
| Windows System Registry | 1,005 | 7,380 |
| Introduction to Ransomware | 59 | 0 |

*Every entry in the corpus was manually audited and approved by cybersecurity experts prior to model pre-training.*

---

## 🎓 Citation & Acknowledgments

This implementation leverages the methodology and weights described in:
> **SRDC: Semantics-based Ransomware Detection and Classification with LLM-assisted Pre-training** (Accepted by AAAI 2025 AISI Track).  
> *Authors: Ce Zhou, Yilun Liu, Weibin Meng, Shimin Tao, Weinan Tian, Feiyu Yao, Xiaochun Li, Tao Han, Boxing Chen, Hao Yang.*

* HuggingFace pre-trained base model: [zhouce/RDC-GPT](https://huggingface.co/zhouce/RDC-GPT)
* Original pre-training corpus dataset: [GitHub Pretraining Corpus](https://github.com/Michael-zhouce/RDCS/tree/main/Pretraining_Corpus)
