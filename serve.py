"""
외부 접속용 서버 런처

사용법:
    python serve.py              # localhost + ngrok 터널
    python serve.py --local      # localhost만 (터널 없이)
    python serve.py --port 8502  # 포트 지정

모바일에서 접속:
    실행 후 출력되는 ngrok URL을 브라우저에 입력
"""

import argparse
import subprocess
import sys
import time
import threading


def start_streamlit(port: int):
    """Streamlit 서버를 시작한다."""
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        "web/app.py",
        "--server.port", str(port),
        "--server.address", "0.0.0.0",
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false",
    ]
    return subprocess.Popen(cmd)


def start_ngrok(port: int) -> str:
    """ngrok 터널을 열고 public URL을 반환한다."""
    from pyngrok import ngrok, conf

    # ngrok 설정
    conf.get_default().monitor_thread = False

    tunnel = ngrok.connect(port, "http")
    return tunnel.public_url


def main():
    parser = argparse.ArgumentParser(description="BlackEvent Predictor Server")
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--local", action="store_true", help="localhost only, no tunnel")
    args = parser.parse_args()

    print(f"\n=== BlackEvent Predictor Server ===\n")

    # Start Streamlit
    print(f"Starting Streamlit on port {args.port}...")
    proc = start_streamlit(args.port)

    # Wait for Streamlit to start
    time.sleep(3)

    print(f"  Local:  http://localhost:{args.port}")

    if not args.local:
        try:
            print(f"  Starting ngrok tunnel...")
            url = start_ngrok(args.port)
            print(f"\n  *** External URL: {url} ***")
            print(f"  *** Mobile: open this URL in your phone browser ***\n")
        except Exception as e:
            print(f"\n  ngrok failed: {e}")
            print(f"  Tip: 'ngrok authtoken <TOKEN>' if you have an ngrok account")
            print(f"  Running local-only mode.\n")

    print("Press Ctrl+C to stop.\n")

    try:
        proc.wait()
    except KeyboardInterrupt:
        print("\nShutting down...")
        proc.terminate()
        try:
            from pyngrok import ngrok
            ngrok.kill()
        except Exception:
            pass


if __name__ == "__main__":
    main()
