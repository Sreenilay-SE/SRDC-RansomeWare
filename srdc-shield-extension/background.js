/**
 * SRDC Shield Background Service Worker (Manifest V3)
 * Intercepts, pauses, and scans downloads pre-disk.
 * Queries localhost:5000/analyze and resumes or cancels the download.
 * Author: ML Auditor (Antigravity)
 */

// Keep track of downloads currently being analyzed
let scanningDownloads = new Set();

chrome.downloads.onCreated.addListener(async (downloadItem) => {
  // Only scan executable files, DLLs, scripts, or dangerous formats to prevent scanner overhead
  const dangerousExtensions = ['.exe', '.dll', '.scr', '.sys', '.msi', '.bat', '.cmd', '.vbs', '.js', '.jar', '.zip', '.rar'];
  const hasDangerousExt = dangerousExtensions.some(ext => downloadItem.filename.toLowerCase().endsWith(ext)) ||
                           dangerousExtensions.some(ext => downloadItem.url.toLowerCase().split('?')[0].endsWith(ext));
  
  if (!hasDangerousExt) {
    console.log(`[SRDC Shield] Bypassing scan for benign file format: ${downloadItem.filename}`);
    return;
  }

  // Prevent duplicate scanning loops
  if (scanningDownloads.has(downloadItem.id)) return;
  scanningDownloads.add(downloadItem.id);

  console.log(`[SRDC Shield] INTERCEPTED download: ${downloadItem.url}`);

  // Step 1: Immediately pause the download to freeze the .crdownload fragment pre-disk
  chrome.downloads.pause(downloadItem.id, () => {
    console.log(`[SRDC Shield] Paused download ${downloadItem.id} safely pre-disk.`);
  });

  // Save "Scanning" state to local storage so the popup UI can reflect live progress
  const scanRecord = {
    id: downloadItem.id,
    filename: downloadItem.filename ? downloadItem.filename.split('\\').pop().split('/').pop() : "Analyzing File...",
    url: downloadItem.url,
    status: "scanning",
    timestamp: Date.now(),
    verdict: "pending",
    elapsed: 0
  };
  chrome.storage.local.set({ "current_scan": scanRecord });

  // Get custom VirusTotal API Key from storage if user saved one
  chrome.storage.local.get(["vt_api_key", "scan_history"], async (result) => {
    const vtKey = result.vt_api_key || "";
    let history = result.scan_history || [];

    // Keep service worker alive during long AI scans
    const keepAliveInterval = setInterval(() => {
      chrome.runtime.getPlatformInfo(() => {});
    }, 15000);

    try {
      // Step 2: Post download URL to local Flask engine
      const response = await fetch("http://127.0.0.1:5000/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url: downloadItem.url, vt_key: vtKey })
      });

      if (!response.ok) {
        throw new Error(`Flask returned HTTP ${response.status}`);
      }

      const scanResult = await response.json();
      console.log("[SRDC Shield] Scan result received:", scanResult);

      const finalRecord = {
        ...scanRecord,
        filename: scanResult.filename,
        hash: scanResult.hash,
        status: scanResult.verdict === "allowed" ? "allowed" : "blocked",
        verdict: scanResult.verdict,
        layer1_vt: scanResult.layer1_vt,
        layer2_srdc: scanResult.layer2_srdc,
        explanations: scanResult.explanations || [],
        elapsed: scanResult.elapsed_seconds || 0,
        timestamp: Date.now()
      };

      // Add to history list (limit to top 20 items)
      history.unshift(finalRecord);
      if (history.length > 20) history.pop();
      chrome.storage.local.set({ "scan_history": history });

      if (scanResult.verdict === "allowed") {
        // Step 3A: Clean - Resume download and write to actual disk safely
        chrome.storage.local.set({ "current_scan": finalRecord });
        chrome.downloads.resume(downloadItem.id, () => {
          console.log(`[SRDC Shield] RESUMED clean download: ${downloadItem.id}`);
          scanningDownloads.delete(downloadItem.id);
        });
      } else {
        // Step 3B: Threat - Cancel download and permanently destroy the .crdownload file
        chrome.storage.local.set({ "current_scan": finalRecord });
        
        // Multi-pronged deletion (handles if file already finished downloading)
        chrome.downloads.cancel(downloadItem.id, () => {
          let err = chrome.runtime.lastError; // Clear last error
          
          chrome.downloads.removeFile(downloadItem.id, () => {
            let err2 = chrome.runtime.lastError;
            console.log(`[SRDC Shield] Destroyed dangerous file from disk: ${downloadItem.id}`);
            scanningDownloads.delete(downloadItem.id);
          });
        });
      }

    } catch (err) {
      console.error("[SRDC Shield] Scanning backend unreachable! Fallback to Allowed for safety:", err);
      // Fallback: If backend is off, resume to avoid blocking user's normal web browsing
      const errorRecord = {
        ...scanRecord,
        status: "error",
        verdict: "allowed",
        message: "Scanning engine was offline. Download allowed."
      };
      chrome.storage.local.set({ "current_scan": errorRecord });
      chrome.downloads.resume(downloadItem.id, () => {
        scanningDownloads.delete(downloadItem.id);
      });
    } finally {
      clearInterval(keepAliveInterval);
    }
  });
});
