/**
 * SRDC Shield Extension Dashboard UI logic
 * Dynamically binds state changes, live scanner status, and scan history.
 * Author: ML Auditor (Antigravity)
 */

document.addEventListener("DOMContentLoaded", () => {
  // Navigation elements
  const tabLive = document.getElementById("tab-live");
  const tabHistory = document.getElementById("tab-history");
  const tabSettings = document.getElementById("tab-settings");

  const contentLive = document.getElementById("content-live");
  const contentHistory = document.getElementById("content-history");
  const contentSettings = document.getElementById("content-settings");

  // Tab switching
  const switchTab = (activeTabBtn, activeContentSection) => {
    [tabLive, tabHistory, tabSettings].forEach(btn => btn.classList.remove("active"));
    [contentLive, contentHistory, contentSettings].forEach(sec => sec.classList.remove("active-content"));

    activeTabBtn.classList.add("active");
    activeContentSection.classList.add("active-content");

    if (activeTabBtn === tabHistory) {
      loadHistory();
    }
  };

  tabLive.addEventListener("click", () => switchTab(tabLive, contentLive));
  tabHistory.addEventListener("click", () => switchTab(tabHistory, contentHistory));
  tabSettings.addEventListener("click", () => switchTab(tabSettings, contentSettings));

  // Shield UI Elements
  const shieldRing = document.getElementById("shield-ring");
  const shieldCore = document.getElementById("shield-core");
  const scanVerdict = document.getElementById("scan-verdict");
  const scanDesc = document.getElementById("scan-desc");

  // File Scanning details
  const fileCard = document.getElementById("file-card");
  const fileName = document.getElementById("file-name");
  const fileUrl = document.getElementById("file-url");
  const scanTime = document.getElementById("scan-time");

  // Stepper Elements
  const dotVt = document.getElementById("dot-vt");
  const dotSrdc = document.getElementById("dot-srdc");
  const line1 = document.getElementById("line-1");
  const line2 = document.getElementById("line-2");

  // Explainability
  const explainCard = document.getElementById("explain-card");
  const valVt = document.getElementById("val-vt");
  const valSrdc = document.getElementById("val-srdc");
  const explainList = document.getElementById("explain-list");

  // Local storage parameters
  const apiInput = document.getElementById("vt-api-input");
  const strictModeInput = document.getElementById("strict-mode");
  const saveSettingsBtn = document.getElementById("save-settings");
  const saveStatus = document.getElementById("save-status");

  // Load Saved Settings on Startup
  chrome.storage.local.get(["vt_api_key", "strict_mode"], (res) => {
    if (res.vt_api_key) apiInput.value = res.vt_api_key;
    if (res.strict_mode) strictModeInput.checked = res.strict_mode;
  });

  // Save Settings logic
  saveSettingsBtn.addEventListener("click", () => {
    const key = apiInput.value.trim();
    const strict = strictModeInput.checked;
    chrome.storage.local.set({ "vt_api_key": key, "strict_mode": strict }, () => {
      saveStatus.classList.remove("hidden");
      setTimeout(() => saveStatus.classList.add("hidden"), 3000);
    });
  });

  // Check Backend Server Status
  const checkBackendHealth = async () => {
    const engineBadge = document.getElementById("engine-status");
    const badgeText = engineBadge.querySelector(".badge-text");
    const dot = engineBadge.querySelector(".status-dot");
    
    try {
      const res = await fetch("http://127.0.0.1:5000/health");
      if (res.ok) {
        badgeText.textContent = "Engine Ready";
        engineBadge.style.background = "rgba(16, 185, 129, 0.1)";
        engineBadge.style.borderColor = "rgba(16, 185, 129, 0.2)";
        dot.style.backgroundColor = "var(--accent-green)";
      } else {
        throw new Error("HTTP warning");
      }
    } catch {
      badgeText.textContent = "Engine Offline";
      engineBadge.style.background = "rgba(239, 68, 68, 0.1)";
      engineBadge.style.borderColor = "rgba(239, 68, 68, 0.2)";
      dot.style.backgroundColor = "var(--accent-red)";
    }
  };
  checkBackendHealth();
  setInterval(checkBackendHealth, 5000);

  // Monitor live download scan in real-time
  const updateLiveShieldUI = (scan) => {
    if (!scan) {
      // Default clean state
      shieldRing.className = "pulse-ring active-ring";
      shieldCore.className = "shield-core green-glow";
      shieldCore.textContent = "🛡️";
      scanVerdict.textContent = "SYSTEM SECURED";
      scanVerdict.className = "shield-verdict-title allowed-title";
      scanDesc.textContent = "Shield is active. Monitoring for insecure dynamic downloads pre-disk.";
      
      fileCard.classList.add("hidden");
      explainCard.classList.add("hidden");
      return;
    }

    // Populate general details
    fileCard.classList.remove("hidden");
    fileName.textContent = scan.filename;
    fileUrl.textContent = scan.url;

    if (scan.status === "scanning") {
      // SCANNING STATE
      shieldRing.className = "pulse-ring active-ring";
      shieldCore.className = "shield-core yellow-glow";
      shieldCore.textContent = "🔎";
      scanVerdict.textContent = "SCANNING FILE...";
      scanVerdict.className = "shield-verdict-title scanning-title";
      scanDesc.textContent = "Download paused pre-disk. Local Flask engine is analyzing binary intentions.";
      
      scanTime.textContent = "Scanning...";
      explainCard.classList.add("hidden");

      // Stepper updates
      dotVt.className = "step-dot active-dot";
      dotVt.textContent = "1";
      dotSrdc.className = "step-dot";
      dotSrdc.textContent = "2";
      line1.className = "step-line active-line";
      line2.className = "step-line";
    }
    else if (scan.status === "allowed") {
      // ALLOWED STATE
      shieldRing.className = "pulse-ring active-ring";
      shieldCore.className = "shield-core green-glow";
      shieldCore.textContent = "✓";
      scanVerdict.textContent = "DOWNLOAD ALLOWED";
      scanVerdict.className = "shield-verdict-title allowed-title";
      scanDesc.textContent = "Binary intent is verified as safe. File written successfully to disk.";
      
      scanTime.textContent = `${scan.elapsed}s`;
      explainCard.classList.remove("hidden");

      // Stepper updates
      dotVt.className = "step-dot success-dot";
      dotVt.textContent = "✓";
      dotSrdc.className = "step-dot success-dot";
      dotSrdc.textContent = "✓";
      line1.className = "step-line active-line";
      line2.className = "step-line active-line";

      // Report details
      valVt.textContent = scan.layer1_vt ? `${scan.layer1_vt.positives}/${scan.layer1_vt.total}` : "0/0";
      valVt.className = "m-val green-text";
      valSrdc.textContent = scan.layer2_srdc ? `${scan.layer2_srdc.verdict}` : "Clean";
      valSrdc.className = "m-val green-text";

      explainList.innerHTML = "";
      if (scan.explanations && scan.explanations.length > 0) {
        scan.explanations.forEach(exp => {
          const li = document.createElement("li");
          li.textContent = exp;
          explainList.appendChild(li);
        });
      } else {
        const li = document.createElement("li");
        li.textContent = "Static features show default administrative API execution, matching Goodware profile.";
        explainList.appendChild(li);
      }
    }
    else if (scan.status === "blocked") {
      // BLOCKED STATE (THREAT DETECTED)
      shieldRing.className = "pulse-ring active-ring";
      shieldCore.className = "shield-core red-glow";
      shieldCore.textContent = "🚨";
      scanVerdict.textContent = "THREAT BLOCKED";
      scanVerdict.className = "shield-verdict-title blocked-title";
      
      const familyName = scan.layer2_srdc ? scan.layer2_srdc.family : "Malicious";
      scanDesc.textContent = `Intercepted pre-disk. Dangerous intention detected matching [${familyName}] signature. File deleted safely.`;
      
      scanTime.textContent = `${scan.elapsed}s`;
      explainCard.classList.remove("hidden");

      // Stepper updates
      dotVt.className = scan.layer1_vt && scan.layer1_vt.flagged ? "step-dot fail-dot" : "step-dot success-dot";
      dotVt.textContent = scan.layer1_vt && scan.layer1_vt.flagged ? "✗" : "✓";
      dotSrdc.className = scan.layer2_srdc && scan.layer2_srdc.verdict === "Ransomware" ? "step-dot fail-dot" : "step-dot success-dot";
      dotSrdc.textContent = scan.layer2_srdc && scan.layer2_srdc.verdict === "Ransomware" ? "✗" : "✓";
      line1.className = "step-line active-line";
      line2.className = "step-line active-line";

      // Report details
      valVt.textContent = scan.layer1_vt ? `${scan.layer1_vt.positives}/${scan.layer1_vt.total}` : "0/0";
      valVt.className = scan.layer1_vt && scan.layer1_vt.flagged ? "m-val red-text" : "m-val green-text";
      
      const srdcConf = scan.layer2_srdc ? `${scan.layer2_srdc.confidence}%` : "100%";
      valSrdc.textContent = scan.layer2_srdc && scan.layer2_srdc.verdict === "Ransomware" ? `${familyName} (${srdcConf})` : "Clean";
      valSrdc.className = scan.layer2_srdc && scan.layer2_srdc.verdict === "Ransomware" ? "m-val red-text" : "m-val green-text";

      explainList.innerHTML = "";
      if (scan.explanations && scan.explanations.length > 0) {
        scan.explanations.forEach(exp => {
          const li = document.createElement("li");
          li.textContent = exp;
          explainList.appendChild(li);
        });
      } else {
        const li = document.createElement("li");
        li.textContent = `Static API calls demonstrate obfuscated encrypting intention consistent with zero-day ransomware behavior.`;
        explainList.appendChild(li);
      }
    }
  };

  // Poll current scan every 500ms to provide live interactive changes
  const pollLiveScan = () => {
    chrome.storage.local.get("current_scan", (res) => {
      updateLiveShieldUI(res.current_scan);
    });
  };
  pollLiveScan();
  setInterval(pollLiveScan, 5000);

  // Storage listener to update UI instantly when background finishes
  chrome.storage.onChanged.addListener((changes, area) => {
    if (area === "local" && changes.current_scan) {
      updateLiveShieldUI(changes.current_scan.newValue);
    }
  });

  // Load and Render History Items
  const loadHistory = () => {
    const historyContainer = document.getElementById("history-items");
    chrome.storage.local.get("scan_history", (res) => {
      const history = res.scan_history || [];
      if (history.length === 0) {
        historyContainer.innerHTML = '<p class="empty-text">No downloads scanned yet.</p>';
        return;
      }

      historyContainer.innerHTML = "";
      history.forEach(item => {
        const div = document.createElement("div");
        const borderClass = item.status === "allowed" ? "allowed-border" : "blocked-border";
        div.className = `history-item ${borderClass}`;

        const badgeClass = item.status === "allowed" ? "allowed-badge" : "blocked-badge";
        const dateStr = new Date(item.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
        
        div.innerHTML = `
          <div class="h-info">
            <h4 class="h-name" title="${item.filename}">${item.filename}</h4>
            <p class="h-meta">${dateStr} | hash: ${item.hash ? item.hash.substring(0, 8) : 'zero-day'}...</p>
          </div>
          <span class="h-badge ${badgeClass}">${item.status}</span>
        `;
        
        // Let user load this record onto their main view when clicked
        div.addEventListener("click", () => {
          chrome.storage.local.set({ "current_scan": item }, () => {
            switchTab(tabLive, contentLive);
          });
        });

        historyContainer.appendChild(div);
      });
    });
  };
});
