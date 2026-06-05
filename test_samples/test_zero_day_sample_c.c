#include <windows.h>
#include <wininet.h>
#include <stdio.h>

// Harmless strings that match ransomware behavioral patterns to populate .rdata section.
// Since these are read-only strings in a compiled C binary, they will be written
// in plain ASCII to the PE data sections, allowing the backend's static strings parser
// to extract them perfectly.
const char* dummy_strings[] = {
    "HKEY_CURRENT_USER\\Software\\Microsoft\\Windows\\CurrentVersion\\Run",
    "HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Run",
    "HKEY_CURRENT_USER\\Software\\Microsoft\\Windows\\CurrentVersion\\RunOnce",
    "HKEY_LOCAL_MACHINE\\SYSTEM\\CurrentControlSet\\Services",
    "opened registry HKEY_CURRENT_USER\\Software\\Microsoft\\Windows\\CurrentVersion\\Run",
    "opened registry HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Cryptography",
    "operations involved opening file with extension crypt",
    "operations involved opening file with extension locked",
    "operations involved opening file with extension encrypted",
    "operations involved opening file with extension tmp",
    "operations involved opening file with extension bat",
    "operations involved opening file with extension scr",
    "opened file in C:\\Users\\AppData\\Local\\Temp\\payload.exe",
    "opened file in C:\\Users\\Desktop\\README_DECRYPT.txt",
    "opened file in C:\\Windows\\System32\\config\\systemprofile",
    "enumerated directory C\\Documents and Settings\\MyUser\\Desktop\\test-personal-files\\img\\",
    "enumerated directory C:\\Users\\Public\\Documents",
    "enumerated directory C:\\Users\\Default\\Downloads",
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
    "YOUR FILES HAVE BEEN ENCRYPTED!",
    "All your documents, photos, databases and other important files have been encrypted",
    "The only method of recovering files is to purchase decrypt tool",
    "To decrypt your files, send BTC to support",
    "decrypt_support@protonmail.com",
    "http://malicious-c2-server.onion/callback"
};

int main(int argc, char* argv[]) {
    printf("============================================================\n");
    printf("  SRDC SHIELD - HARMLESS C TEST FILE (STRINGS + IAT)\n");
    printf("============================================================\n");
    printf("This is a 100%% harmless binary compiled from C.\n");
    printf("It contains mock ransomware behavior patterns in its read-only data section.\n");
    printf("If you see this, the file was NOT blocked by the scanner.\n");
    printf("============================================================\n");

    // Force link suspicious APIs and references to keep them in the binary.
    if (argc < 0) {
        // Force strings to be linked
        for (int i = 0; i < sizeof(dummy_strings) / sizeof(dummy_strings[0]); i++) {
            printf("%s", dummy_strings[i]);
        }

        // kernel32 functions
        FindFirstFileW(NULL, NULL);
        FindNextFileW(NULL, NULL);
        FindClose(NULL);
        GetLogicalDriveStringsW(0, NULL);
        DeleteFileW(NULL);
        MoveFileW(NULL, NULL);
        CreateFileW(NULL, 0, 0, NULL, 0, 0, NULL);
        WriteFile(NULL, NULL, 0, NULL, NULL);
        SetFileAttributesW(NULL, 0);

        // advapi32 functions
        RegSetValueExW(NULL, NULL, 0, 0, NULL, 0);
        RegOpenKeyExW(NULL, NULL, 0, 0, NULL);
        RegCreateKeyExW(NULL, NULL, 0, NULL, 0, 0, NULL, NULL, NULL);
        CryptAcquireContextW(NULL, NULL, NULL, 0, 0);
        CryptGenKey(0, 0, 0, NULL);
        CryptEncrypt(0, 0, 0, 0, NULL, NULL, 0);
        CryptDecrypt(0, 0, 0, 0, NULL, NULL);
        CryptDestroyKey(0);
        CryptReleaseContext(0, 0);

        // wininet functions
        InternetOpenA(NULL, 0, NULL, NULL, 0);
        InternetConnectA(NULL, NULL, 0, NULL, NULL, 0, 0, 0);
        HttpSendRequestA(NULL, NULL, 0, NULL, 0);
        HttpOpenRequestA(NULL, NULL, NULL, NULL, NULL, NULL, 0, 0);
    }

    return 0;
}
