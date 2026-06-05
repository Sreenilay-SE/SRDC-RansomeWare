"""
SRDC Shield — Safe Zero-Day Test Sample
========================================
This is a 100% HARMLESS test executable designed to trigger
the SRDC AI ransomware detection model.

It contains ransomware-like string patterns and API references
that the SRDC GPT-2 model recognizes as malicious behavioral DNA,
but it performs NO harmful actions whatsoever.

Purpose: Demonstrate that SRDC Shield can detect zero-day threats
that traditional signature-based antivirus (VirusTotal) cannot.

Author: SRDC Shield Research Team
"""

import sys
import os
import time

# =====================================================================
# HARMLESS STRING DECLARATIONS
# These strings exist ONLY to populate the PE binary's embedded strings
# section. The SRDC model's static analyzer extracts these and feeds
# them to the GPT-2 classifier, which recognizes the ransomware pattern.
# NONE of these are ever called or executed.
# =====================================================================

# Ransomware-like API function references (NEVER called — just strings)
API_REFERENCES = [
    "API:kernel create file",
    "API:kernel write file",
    "API:crypt encrypt",
    "API:crypt decrypt",
    "API:crypt acquire context",
    "API:crypt gen key",
    "API:crypt destroy key",
    "API:find first file",
    "API:find next file",
    "API:find close",
    "API:create file",
    "API:write file",
    "API:delete file",
    "API:move file",
    "API:get logical drive strings",
    "API:reg set value",
    "API:reg open key",
    "API:virtual alloc",
    "API:virtual protect",
    "CryptEncrypt",
    "CryptDecrypt",
    "CryptAcquireContextW",
    "CryptGenKey",
    "CryptDestroyKey",
    "FindFirstFileW",
    "FindNextFileW",
    "FindClose",
    "CreateFileW",
    "WriteFile",
    "DeleteFileW",
    "MoveFileW",
    "GetLogicalDriveStringsW",
    "RegSetValueExW",
    "RegOpenKeyExW",
    "VirtualAlloc",
    "VirtualProtect",
    "InternetOpenA",
    "InternetConnectA",
    "HttpSendRequestA",
]

# Registry persistence paths (NEVER accessed — just strings)
REGISTRY_PATHS = [
    "HKEY_CURRENT_USER\\Software\\Microsoft\\Windows\\CurrentVersion\\Run",
    "HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Run",
    "HKEY_CURRENT_USER\\Software\\Microsoft\\Windows\\CurrentVersion\\RunOnce",
    "HKEY_LOCAL_MACHINE\\SYSTEM\\CurrentControlSet\\Services",
    "opened registry HKEY_CURRENT_USER\\Software\\Microsoft\\Windows\\CurrentVersion\\Run",
    "opened registry HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Cryptography",
]

# File extension targets typical of ransomware (NEVER used — just strings)
TARGET_EXTENSIONS = [
    "operations involved opening file with extension exe",
    "operations involved opening file with extension dll",
    "operations involved opening file with extension tmp",
    "operations involved opening file with extension bat",
    "operations involved opening file with extension scr",
    ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".pdf", ".jpg", ".png", ".mp3", ".mp4",
    ".txt", ".csv", ".db", ".sql", ".zip",
    ".crypt", ".locked", ".encrypted",
]

# Dropped file indicators (NEVER created — just strings)
DROPPED_FILES = [
    "dropped file's extension involved tmp",
    "dropped file's extension involved exe",
    "opened file in C:\\Users\\AppData\\Local\\Temp\\payload.exe",
    "opened file in C:\\Users\\Desktop\\README_DECRYPT.txt",
    "opened file in C:\\Windows\\System32\\config\\systemprofile",
]

# Directory enumeration indicators (NEVER performed — just strings)
DIRECTORY_ENUM = [
    "enumerated directory C\\Documents and Settings\\MyUser\\Desktop\\test-personal-files\\img\\",
    "enumerated directory C:\\Users\\Public\\Documents",
    "enumerated directory C:\\Users\\Default\\Downloads",
]

# Embedded strings that match ransomware behavioral patterns
EMBEDDED_STRINGS = [
    "embeded string CryptEncrypt",
    "embeded string FindFirstFile",
    "embeded string RegSetValue",
    "embeded string VirtualAlloc",
    "embeded string payload",
    "embeded string decrypt",
    "embeded string ransom",
    "embeded string bitcoin",
    "embeded string encrypt",
    "embeded string lockfile",
]

# Ransom note content (NEVER written to disk — just a string)
RANSOM_NOTE = """
YOUR FILES HAVE BEEN ENCRYPTED!
All your documents, photos, databases and other important files
have been encrypted with strongest encryption and unique key.
The only method of recovering files is to purchase decrypt tool.
To decrypt your files, send 0.5 BTC to: 1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa
Contact: decrypt_support@protonmail.com
WARNING: Do not rename encrypted files. Do not try to decrypt using third party software.
"""

# Network callback indicators (NEVER connected — just strings)
NETWORK_INDICATORS = [
    "http://malicious-c2-server.onion/callback",
    "tcp://command-and-control:4444/beacon",
    "InternetOpenA",
    "HttpSendRequestA",
    "URLDownloadToFileA",
]


def main():
    """This function is the ONLY code that actually executes."""
    print("=" * 60)
    print("  SRDC SHIELD - SAFE ZERO-DAY TEST SAMPLE")
    print("=" * 60)
    print()
    print("  This file is 100% HARMLESS.")
    print("  It exists ONLY to test the SRDC Shield detection engine.")
    print()
    print("  It contains ransomware-like string patterns that the")
    print("  SRDC GPT-2 AI model recognizes as malicious, but it")
    print("  performs NO harmful actions whatsoever.")
    print()
    print("  If you see this message, the file was NOT blocked.")
    print("  The SRDC Shield extension should have intercepted this")
    print("  download before it reached your disk.")
    print("=" * 60)

    # Keep console open if double-clicked
    input("\nPress Enter to exit...")


if __name__ == "__main__":
    main()
