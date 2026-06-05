"""
SRDC Shield — PE Import Patcher (LIEF 0.16+ API)
==================================================
Patches test_zero_day_sample.exe to inject ransomware-like
API imports into its PE Import Address Table (IAT).
"""

import lief
import os
import shutil

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ORIGINAL_EXE = os.path.join(SCRIPT_DIR, "test_zero_day_sample_ORIGINAL.exe")
OUTPUT_EXE = os.path.join(SCRIPT_DIR, "test_zero_day_sample.exe")


def patch_pe():
    # Use the original backup if it exists, otherwise use the current exe
    input_exe = ORIGINAL_EXE if os.path.exists(ORIGINAL_EXE) else OUTPUT_EXE

    if not os.path.exists(input_exe):
        print(f"ERROR: {input_exe} not found!")
        return False

    print(f"[+] Parsing PE file: {os.path.basename(input_exe)}")
    binary = lief.parse(input_exe)

    if binary is None:
        print("ERROR: Failed to parse PE file!")
        return False

    # ================================================================
    # Inject ransomware-like imports using LIEF 0.16+ API
    # ================================================================

    # --- advapi32.dll: Registry persistence + crypto context ---
    print("[+] Adding advapi32.dll imports (registry + crypto)...")
    advapi = binary.add_import("advapi32.dll")
    for func in ["RegSetValueExW", "RegOpenKeyExW", "RegCloseKey",
                 "RegCreateKeyExW", "CryptAcquireContextW", "CryptGenKey",
                 "CryptEncrypt", "CryptDecrypt", "CryptDestroyKey",
                 "CryptReleaseContext"]:
        advapi.add_entry(func)

    # --- crypt32.dll: Certificate and message encryption ---
    print("[+] Adding crypt32.dll imports (encryption)...")
    crypt32 = binary.add_import("crypt32.dll")
    for func in ["CryptEncryptMessage", "CryptDecryptMessage",
                 "CertOpenStore", "CertFindCertificateInStore"]:
        crypt32.add_entry(func)

    # --- kernel32.dll: File enumeration + drive scanning ---
    print("[+] Adding kernel32.dll imports (file enumeration)...")
    k32 = binary.add_import("kernel32.dll")
    for func in ["FindFirstFileW", "FindNextFileW", "FindClose",
                 "GetLogicalDriveStringsW", "DeleteFileW", "MoveFileW",
                 "CreateFileW", "WriteFile", "SetFileAttributesW"]:
        k32.add_entry(func)

    # --- wininet.dll: C2 communication ---
    print("[+] Adding wininet.dll imports (network C2)...")
    wininet = binary.add_import("wininet.dll")
    for func in ["InternetOpenA", "InternetConnectA",
                 "HttpSendRequestA", "HttpOpenRequestA"]:
        wininet.add_entry(func)

    # ================================================================
    # Write the patched PE
    # ================================================================
    print(f"[+] Rebuilding and writing patched PE to: {os.path.basename(OUTPUT_EXE)}")
    config = lief.PE.Builder.config_t()
    config.imports = True  # Crucial: tell LIEF to rebuild the Import Table
    
    builder = lief.PE.Builder(binary, config)
    builder.build()
    builder.write(OUTPUT_EXE)

    # Verify
    patched = lief.parse(OUTPUT_EXE)
    import_count = 0
    print("\n[+] Verifying patched imports:")
    for lib in patched.imports:
        entries = [e.name for e in lib.entries if e.name]
        if entries:
            print(f"    {lib.name}: {len(entries)} functions")
            import_count += len(entries)

    size_mb = os.path.getsize(OUTPUT_EXE) / (1024 * 1024)
    print("\n[+] Patch complete!")
    print(f"    Total imported functions: {import_count}")
    print(f"    File size: {size_mb:.1f} MB")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("  SRDC Shield -- Zero-Day Test PE Patcher")
    print("=" * 60)
    print()
    success = patch_pe()
    if success:
        print("\n" + "=" * 60)
        print("  TEST FILE READY!")
        print("  1. Restart file server:  python host_test_file.py")
        print("  2. Restart backend:      python ../srdc_shield_backend.py")
        print("  3. Open Chrome:          http://localhost:8080")
        print("=" * 60)
    else:
        print("\n[ERROR] Patching failed!")

