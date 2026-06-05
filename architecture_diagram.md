# SRDC Shield Architecture Diagram & Design Specification

This document provides a comprehensive blueprint of **SRDC Shield**—a Chrome extension paired with a local Flask backend implementing a dual-layer ransomware detection engine. 

---

## 1. High-Level Architecture Flowchart

The following flowchart visualizes the sequence of operations from a user initiating a download to the browser either resuming or canceling the download based on the security verdict.

```mermaid
graph TD
    %% Define styles
    classDef chrome fill:#1e293b,stroke:#38bdf8,stroke-width:2px,color:#f8fafc;
    classDef flask fill:#0f172a,stroke:#34d399,stroke-width:2px,color:#f8fafc;
    classDef model fill:#311042,stroke:#c084fc,stroke-width:2px,color:#f8fafc;
    classDef external fill:#1c1917,stroke:#fb923c,stroke-width:2px,color:#f8fafc;

    subgraph extension ["Chrome Extension (Frontend)"]
        A["User initiates download"] --> B["background.js (intercepts download)"]
        B --> C["Pause download pre-disk"]
        C --> D["Query Flask Backend (/scan)"]
        E["popup.html (Glassmorphic UI)"] <-->|Long-polling / updates| B
        B -->|Verdict: Allowed| F["Resume download"]
        B -->|Verdict: Blocked| G["Cancel download & warn user"]
    end

    subgraph backend ["Flask Backend (Local Scanning Coordinator)"]
        D --> H["srdc_shield_backend.py (/scan)"]
        H --> I{"Layer 1: VirusTotal Check"}
        
        %% VT Path
        I -->|Hash Match Found| J["Retrieve Verdict"]
        
        %% Deep Scan Path
        I -->|Hash Not Found / Unknown| K["Layer 2: Local PE Semantic Engine"]
        K --> L["Download to sandbox_temp/"]
        L --> M["Read 256KB Optimization"]
        M --> N["pefile static analysis"]
        N --> O["Map APIs & strings to semantic sentences"]
        O --> P["Feed sentences to preloaded models"]
    end

    subgraph cognitive ["Cognitive AI Layer (PyTorch CPU)"]
        P --> Q["Zero-Day Model (srdc_zero_day_BEST.pth)"]
        P --> R["Family Classifier (srdc_family_BEST.pth)"]
        Q --> S["Ransomware vs Goodware Verdict"]
        R --> T["Ransomware Family Class (Citroni, Kovter, etc.)"]
        S & T --> U["Compile scanning report"]
    end

    subgraph ext_apis ["External APIs"]
        I <-->|Query file SHA-256| V["VirusTotal Database"]
    end

    %% Apply CSS classes to nodes
    class A,B,C,D,E,F,G chrome;
    class H,I,K,L,M,N,O,P flask;
    class Q,R,S,T,U model;
    class V external;
```

---

## 2. Sequence Diagram: Step-by-Step Data Flow

The diagram below details the chronological request-response lifecycle when a file is intercepted.

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant Browser as Chrome Browser
    participant Ext as Chrome Extension (background.js)
    participant Popup as Extension Dashboard (popup.js)
    participant Flask as Flask Server (srdc_shield_backend.py)
    participant VT as VirusTotal API
    participant Model as Preloaded GPT-2 Models

    User->>Browser: Click Download Link
    Browser->>Ext: triggers chrome.downloads.onCreated
    activate Ext
    Ext->>Browser: chrome.downloads.pause
    Note over Ext: Download is safely held pre-disk
    Ext->>Popup: Notify scanning started
    activate Popup
    Popup->>User: Display glassmorphic "Scanning..." animation
    
    Ext->>Flask: POST /scan (file metadata)
    activate Flask
    
    Flask->>VT: Check file hash
    alt File known on VirusTotal
        VT-->>Flask: Return malicious/clean votes
        Note over Flask: Short-circuits deep scan for speed
    else File unknown / new (Zero-day)
        Flask->>Browser: Stream file content to sandbox_temp/
        Note over Flask: Reads only first 256KB to optimize CPU load
        Flask->>Flask: Static parse (imports, sections, strings)
        Flask->>Flask: Translate raw symbols to semantic text
        Flask->>Model: Tokenize & forward semantic text
        activate Model
        Model-->>Flask: Zero-Day Verdict & Family classification probabilities
        deactivate Model
        Flask->>Flask: Delete temporary sandbox file
    end
    
    Flask-->>Ext: JSON payload (verdict, details, confidence)
    deactivate Flask
    
    Ext->>Popup: Send final report JSON
    Popup->>User: Update UI (Green: Clean, Red: Threat Alert + Breakdown)
    deactivate Popup
    
    alt Verdict is Allowed
        Ext->>Browser: chrome.downloads.resume
    else Verdict is Blocked
        Ext->>Browser: chrome.downloads.cancel
        Ext->>Browser: Alert warning notification to user
    end
    deactivate Ext
```

---

## 3. Detailed Component Breakdown

### 3.1 Chrome Extension (MV3)
* **Manifest (`manifest.json`)**: Configured with MV3, requesting `downloads`, `downloads.shelf`, and `declarativeNetRequest` permissions. It registers `background.js` as a background service worker.
* **Service Worker (`background.js`)**: Captures downloads, pauses them, performs the HTTP POST to localhost:5000, and resumes/cancels based on the scanning report.
* **Dashboard (`popup.html` / `popup.js` / `popup.css`)**: Built with a sleek, premium dark-theme layout using CSS glassmorphism, responsive flex layouts, and custom fonts. It renders plain-English descriptions of why particular APIs are suspicious (e.g. mapping `CryptEncrypt` to ransomware behaviors).

### 3.2 Flask Backend Coordination
* **Secure Sandbox (`sandbox_temp/`)**: Temporary directory configured at root where files are temporarily streamed during Layer 2 analysis.
* **256KB Scan Optimization**: To ensure near-instantaneous CPU scans, the backend reads only the first 256KB of the target file. This is sufficient to capture headers, import tables, and initial resource strings where PE malware features are stored.
* **VirusTotal API Check (Layer 1)**: Computes the SHA-256 hash of the incoming file and queries the VirusTotal database. If the file is already flagged/cleared, it returns immediately without running the deep model.

### 3.3 Semantic Translation & Model Inference (Layer 2)
* **Semantic Parser**: Translates raw camelCase Windows APIs to natural, lowercased, space-separated word combinations (e.g. `RegSetValueEx` ➡️ `reg set value`). Suspicious strings and file system changes are mapped to readable text.
* **Preloaded PyTorch Models**:
  - `srdc_zero_day_BEST.pth`: A binary classifier fine-tuned on the `zhouce/RDC-GPT` architecture to detect ransomware vs goodware.
  - `srdc_family_BEST.pth`: A 12-class classifier evaluating the exact ransomware family signature (Citroni, CryptLocker, Kovter, Reveton, TeslaCrypt, etc.).
  - Running on CPU with cached preloading to complete scans in under **0.25 seconds** once initialized.
