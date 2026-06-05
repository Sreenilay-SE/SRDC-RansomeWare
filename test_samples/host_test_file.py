"""
SRDC Shield — Local HTTP File Server for Testing
=================================================
Hosts the compiled test_zero_day_sample.exe on localhost:8080
so it can be downloaded through Chrome to trigger the extension.

Usage:
  1. Place test_zero_day_sample.exe in this same folder
  2. Run: python host_test_file.py
  3. Open Chrome: http://localhost:8080
  4. Click the download link to trigger SRDC Shield
"""

import http.server
import socketserver
import os

PORT = 8080
DIRECTORY = os.path.dirname(os.path.abspath(__file__))

# HTML page with a download button for a realistic demo experience
HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Software Download Portal (Test)</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #fff;
        }
        .container {
            background: rgba(255,255,255,0.05);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 20px;
            padding: 50px;
            max-width: 600px;
            text-align: center;
        }
        h1 { font-size: 28px; margin-bottom: 10px; }
        .subtitle { color: #aaa; margin-bottom: 30px; font-size: 14px; }
        .file-info {
            background: rgba(255,255,255,0.08);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 25px;
            text-align: left;
        }
        .file-info p { margin: 5px 0; font-size: 14px; color: #ccc; }
        .file-info span { color: #7dd3fc; }
        .download-btn {
            display: inline-block;
            padding: 15px 40px;
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            text-decoration: none;
            border-radius: 12px;
            font-size: 16px;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        }
        .download-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
        }
        .warning {
            margin-top: 25px;
            padding: 15px;
            background: rgba(34, 197, 94, 0.1);
            border: 1px solid rgba(34, 197, 94, 0.3);
            border-radius: 10px;
            font-size: 12px;
            color: #86efac;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Software Download Portal</h1>
        <p class="subtitle">Free System Optimization Tool v2.1.0</p>

        <div class="file-info">
            <p>File: <span>test_zero_day_sample.exe</span></p>
            <p>Version: <span>2.1.0</span></p>
            <p>Publisher: <span>Unknown Publisher</span></p>
            <p>Category: <span>System Utility</span></p>
        </div>

        <a href="/test_zero_day_sample.exe" class="download-btn">
            Download Now
        </a>

        <div class="warning">
            This is a safe SRDC Shield test page. The .exe file is 100% harmless
            and is designed to test zero-day ransomware detection capabilities.
        </div>
    </div>
</body>
</html>
"""


class TestHandler(http.server.SimpleHTTPRequestHandler):
    """Custom handler that serves the HTML page at root and files from directory."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            # Serve the custom download page
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode())
        else:
            # Serve actual files (like the .exe)
            super().do_GET()

    def log_message(self, format, *args):
        print(f"[FILE SERVER] {args[0]}")


if __name__ == "__main__":
    # Check if the .exe exists
    exe_path = os.path.join(DIRECTORY, "test_zero_day_sample.exe")
    if not os.path.exists(exe_path):
        print("=" * 60)
        print("  WARNING: test_zero_day_sample.exe not found!")
        print("  Please compile it first with:")
        print("    pyinstaller --onefile test_zero_day_sample.py")
        print("  Then copy dist/test_zero_day_sample.exe to this folder.")
        print("=" * 60)
    else:
        size_mb = os.path.getsize(exe_path) / (1024 * 1024)
        print(f"  Found: test_zero_day_sample.exe ({size_mb:.1f} MB)")

    print()
    print("=" * 60)
    print("  SRDC Shield — Test File Server")
    print("=" * 60)
    print(f"  Serving from : {DIRECTORY}")
    print(f"  Server URL   : http://localhost:{PORT}")
    print(f"  Download URL : http://localhost:{PORT}/test_zero_day_sample.exe")
    print()
    print("  Open http://localhost:8080 in Chrome to test!")
    print("  Press Ctrl+C to stop the server.")
    print("=" * 60)

    with socketserver.ThreadingTCPServer(("", PORT), TestHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[FILE SERVER] Stopped.")
