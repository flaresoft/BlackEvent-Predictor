"""
Collect — 뉴스 수집 + 태깅 + 스코어링
"""

import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

import streamlit as st
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

COOKIE_FILE = ROOT / "config" / "bigkinds_cookie.txt"
DOWNLOAD_DIR = ROOT / "data" / "raw" / "bigkinds"
BIGKINDS_BASE = "https://www.bigkinds.or.kr"


def _extract_chrome_cookies(domain: str = "bigkinds.or.kr") -> tuple[str, list[str]]:
    """Chrome 쿠키를 추출한다 (Windows 전용).

    Chrome의 프로파일(쿠키 DB + Local State)을 임시 디렉토리에 복사한 뒤
    headless Chrome을 띄워 CDP로 복호화된 쿠키를 읽어온다.
    Chrome 127+ App-Bound Encryption(v20)에서도 작동한다.

    Returns:
        (cookie_string, log_lines)
    """
    import os
    import shutil
    import stat
    import tempfile
    import time
    import traceback

    log: list[str] = []

    def _log(msg: str):
        log.append(msg)

    local_app = os.environ.get("LOCALAPPDATA", "")
    src_profile = Path(local_app) / "Google" / "Chrome" / "User Data"
    _log(f"Chrome profile: {src_profile}")
    _log(f"  exists: {src_profile.exists()}")
    if not src_profile.exists():
        raise FileNotFoundError("Chrome User Data 디렉토리를 찾을 수 없습니다")

    # ── 복사할 파일 목록과 상태 점검 ──
    files_to_copy = [
        ("Local State", src_profile / "Local State"),
        ("Cookies", src_profile / "Default" / "Network" / "Cookies"),
        ("Cookies-journal", src_profile / "Default" / "Network" / "Cookies-journal"),
        ("Preferences", src_profile / "Default" / "Preferences"),
    ]
    for label, p in files_to_copy:
        if p.exists():
            try:
                st_info = p.stat()
                mode_str = stat.filemode(st_info.st_mode)
                _log(f"  {label}: {st_info.st_size:,} bytes, {mode_str}")
            except Exception as e:
                _log(f"  {label}: stat failed — {e}")
        else:
            _log(f"  {label}: NOT FOUND (skip)")

    def _safe_copy(src: Path, dst: Path):
        """Chrome이 잠근 파일도 복사한다."""
        try:
            with open(src, "rb") as f:
                data = f.read()
            with open(dst, "wb") as f:
                f.write(data)
            _log(f"  copy OK: {src.name} → {len(data):,} bytes")
        except PermissionError:
            _log(f"  copy LOCKED: {src.name}")
            raise

    def _kill_chrome() -> list[int]:
        """Chrome 프로세스를 종료하고 PID 목록을 반환한다."""
        import subprocess
        # 실행 중인 chrome.exe PID 수집
        result = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq chrome.exe", "/FO", "CSV", "/NH"],
            capture_output=True, text=True,
        )
        pids = []
        for line in result.stdout.strip().splitlines():
            parts = line.replace('"', '').split(',')
            if len(parts) >= 2 and parts[0].lower() == "chrome.exe":
                pids.append(int(parts[1]))

        if pids:
            subprocess.run(
                ["taskkill", "/F", "/IM", "chrome.exe"],
                capture_output=True, timeout=10,
            )
            time.sleep(1)  # 파일 잠금 해제 대기
        return pids

    def _relaunch_chrome():
        """Chrome을 다시 시작한다 (이전 탭 자동 복원)."""
        import subprocess
        chrome_paths = [
            Path(os.environ.get("PROGRAMFILES", "")) / "Google" / "Chrome" / "Application" / "chrome.exe",
            Path(os.environ.get("PROGRAMFILES(X86)", "")) / "Google" / "Chrome" / "Application" / "chrome.exe",
        ]
        for p in chrome_paths:
            if p.exists():
                subprocess.Popen([str(p)], close_fds=True)
                return
        # PATH에서 찾기
        subprocess.Popen(["chrome.exe"], close_fds=True, shell=True)

    # ── 필수 파일 목록 ──
    copy_plan = [
        ("Local State", src_profile / "Local State", None),
        ("Cookies", src_profile / "Default" / "Network" / "Cookies", "Default/Network"),
        ("Cookies-journal", src_profile / "Default" / "Network" / "Cookies-journal", "Default/Network"),
        ("Preferences", src_profile / "Default" / "Preferences", "Default"),
    ]

    # ── 1차 시도: Chrome 켠 상태에서 복사 ──
    tmp_dir = Path(tempfile.mkdtemp(prefix="chrome_cookie_"))
    _log(f"Temp dir: {tmp_dir}")
    chrome_was_killed = False

    try:
        locked_files: list[tuple] = []
        for label, src_path, subdir in copy_plan:
            if not src_path.exists():
                _log(f"  {label}: not found (skip)")
                continue
            dst_dir = tmp_dir / subdir if subdir else tmp_dir
            dst_dir.mkdir(parents=True, exist_ok=True)
            try:
                _safe_copy(src_path, dst_dir / src_path.name)
            except PermissionError:
                locked_files.append((label, src_path, subdir))

        # ── 잠긴 파일이 있으면 Chrome 재시작 후 재시도 ──
        if locked_files:
            _log(f"Locked files: {[f[0] for f in locked_files]}")
            _log("Killing Chrome to unlock files...")
            killed = _kill_chrome()
            _log(f"  killed {len(killed)} Chrome processes")
            chrome_was_killed = True

            for label, src_path, subdir in locked_files:
                dst_dir = tmp_dir / subdir if subdir else tmp_dir
                dst_dir.mkdir(parents=True, exist_ok=True)
                _safe_copy(src_path, dst_dir / src_path.name)  # 이제 잠금 해제됨

        # ── headless Chrome 기동 → CDP로 쿠키 추출 (사이트 접속 없이) ──
        _log("Launching headless Chrome...")
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options

        options = Options()
        options.add_argument(f"--user-data-dir={tmp_dir}")
        options.add_argument("--headless=new")
        options.add_argument("--no-first-run")
        options.add_argument("--disable-gpu")
        options.add_argument("--disable-extensions")

        driver = webdriver.Chrome(options=options)
        try:
            # 사이트 접속하면 서버가 새 세션을 발급하므로,
            # CDP로 저장된 쿠키를 직접 꺼낸다 (네트워크 요청 없음)
            all_cookies = driver.execute_cdp_cmd("Network.getAllCookies", {})
            cdp_cookies = all_cookies.get("cookies", [])
            _log(f"CDP: total {len(cdp_cookies)} cookies in profile")

            matched = [
                c for c in cdp_cookies
                if domain in c.get("domain", "")
            ]
            _log(f"Matched {len(matched)} cookies for {domain}:")
            for c in matched:
                _log(f"  {c['name']} (domain={c['domain']}, httpOnly={c.get('httpOnly')})")
        finally:
            driver.quit()
            _log("Chrome closed")

        if not matched:
            raise ValueError(
                f"{domain} 쿠키가 없습니다. "
                "Chrome에서 BigKinds에 로그인되어 있는지 확인하세요."
            )

        cookie_str = "; ".join(f"{c['name']}={c['value']}" for c in matched)
        return cookie_str, log
    except Exception as exc:
        _log(f"EXCEPTION:\n{traceback.format_exc()}")
        exc.extract_log = log
        raise
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        if chrome_was_killed:
            _log("Relaunching Chrome...")
            try:
                _relaunch_chrome()
            except Exception:
                pass


def _scan_collected_dates(download_dir: Path) -> set[str]:
    """실제 데이터가 존재하는 날짜를 모든 소스에서 추출한다.

    확인 소스:
    1. bigkinds_*.xlsx (raw 수집 파일)
    2. corpus_daily/*.parquet (일별 태깅 파일)
    3. corpus_tagged.parquet (메인 코퍼스)
    4. daily_scores.csv (스코어링 완료)
    """
    collected = set()

    # 1) raw xlsx
    if download_dir.exists():
        for f in download_dir.glob("bigkinds_*.xlsx"):
            match = re.search(r"bigkinds_(\d{4}-\d{2}-\d{2})_", f.name)
            if match and f.stat().st_size > 1024:
                collected.add(match.group(1))

    # 2) corpus_daily parquet
    daily_dir = ROOT / "data" / "processed" / "corpus_daily"
    if daily_dir.exists():
        for f in daily_dir.glob("*.parquet"):
            match = re.match(r"(\d{4}-\d{2}-\d{2})\.parquet", f.name)
            if match:
                collected.add(match.group(1))

    # 3) corpus_tagged.parquet — row group statistics로 날짜 범위만 추출 (즉시)
    if "_corpus_dates" not in st.session_state:
        corpus_path = ROOT / "data" / "processed" / "corpus_tagged.parquet"
        if corpus_path.exists():
            try:
                import pyarrow.parquet as pq
                from datetime import date as _date
                pf = pq.ParquetFile(corpus_path)
                date_col_idx = next(
                    i for i, n in enumerate(pf.schema_arrow.names) if n == "date"
                )
                first_stats = pf.metadata.row_group(0).column(date_col_idx).statistics
                last_stats = pf.metadata.row_group(
                    pf.metadata.num_row_groups - 1
                ).column(date_col_idx).statistics
                corpus_min = first_stats.min
                corpus_max = last_stats.max
                corpus_dates = set()
                d = corpus_min
                while d <= corpus_max:
                    corpus_dates.add(d.isoformat())
                    d += timedelta(days=1)
                st.session_state["_corpus_dates"] = corpus_dates
            except Exception:
                st.session_state["_corpus_dates"] = set()
        else:
            st.session_state["_corpus_dates"] = set()
    collected |= st.session_state["_corpus_dates"]

    # 4) daily_scores.csv
    scores_csv = ROOT / "data" / "outputs" / "daily_scores.csv"
    if scores_csv.exists():
        try:
            scores_df = pd.read_csv(scores_csv, usecols=["date"])
            collected |= set(scores_df["date"].tolist())
        except Exception:
            pass

    return collected


st.set_page_config(page_title="Collect", page_icon="📰", layout="wide")
st.header("뉴스 수집 & 스코어링")

# ── 1. 쿠키 입력 ──
st.subheader("BigKinds 쿠키 설정")

current_cookie = ""
if COOKIE_FILE.exists():
    current_cookie = COOKIE_FILE.read_text(encoding="utf-8").strip()

# Chrome에서 쿠키 자동 추출
if st.button("Chrome에서 쿠키 가져오기"):
    with st.spinner("Chrome 쿠키 읽는 중 (headless Chrome 기동, 5~10초)..."):
        extract_log: list[str] = []
        try:
            cookie_str, extract_log = _extract_chrome_cookies()
            st.session_state["auto_cookie"] = cookie_str
            n = cookie_str.count(";") + 1
            st.success(f"쿠키 {n}개 추출 완료! 아래 '쿠키 저장' 을 눌러주세요.")
            with st.expander("추출 로그"):
                st.code("\n".join(extract_log))
        except Exception as e:
            st.error(f"자동 추출 실패: {e}")
            extract_log = getattr(e, "extract_log", [])
            with st.expander("디버그 로그", expanded=True):
                st.code("\n".join(extract_log) if extract_log else str(e))
            with st.expander("수동으로 쿠키 복사하는 방법"):
                st.markdown("""
1. Chrome에서 **bigkinds.or.kr** 로그인
2. **F12** → **Network** 탭 열기
3. 페이지 새로고침 또는 아무 검색 실행
4. 요청 하나 클릭 → **Headers** → **Request Headers**
5. **Cookie** 값 전체 복사 → 아래 텍스트 영역에 붙여넣기
""")

# 자동 추출된 쿠키가 있으면 우선 사용
display_cookie = st.session_state.get("auto_cookie", current_cookie)

cookie_input = st.text_area(
    "BigKinds 쿠키",
    value=display_cookie,
    height=100,
    placeholder="JSESSIONID=...; 또는 전체 Cookie 헤더 값",
)

col_save, col_verify = st.columns(2)

with col_save:
    if st.button("쿠키 저장", type="primary"):
        cookie_text = cookie_input.strip()
        if not cookie_text:
            st.error("쿠키 내용이 비어 있습니다.")
        else:
            COOKIE_FILE.parent.mkdir(parents=True, exist_ok=True)
            COOKIE_FILE.write_text(cookie_text, encoding="utf-8")
            st.success("쿠키가 저장되었습니다.")

with col_verify:
    if st.button("쿠키 검증"):
        cookie_text = cookie_input.strip()
        if not cookie_text:
            st.error("쿠키를 먼저 입력하세요.")
        else:
            with st.spinner("BigKinds 다운로드 엔드포인트로 인증 테스트 중..."):
                try:
                    import json as _json
                    import urllib.parse

                    import requests
                    from src.daily_pipeline.run import _build_bigkinds_param

                    session = requests.Session()

                    # 다운로드 엔드포인트로 테스트 (검색 API는 비로그인도 응답함)
                    headers_dl = {
                        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                        "Content-Type": "application/x-www-form-urlencoded",
                        "Origin": BIGKINDS_BASE,
                        "Referer": f"{BIGKINDS_BASE}/v2/news/search.do",
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36",
                        "Cookie": cookie_text,
                    }
                    param = _build_bigkinds_param("2025-01-02", "2025-01-02")
                    param["sectionDiv"] = "1000"
                    param["realURI"] = "/api/news/previewData.do"
                    param["interval"] = 2

                    form_data = "jsonSearchParam=" + urllib.parse.quote(
                        _json.dumps(param, ensure_ascii=False)
                    )
                    resp = session.post(
                        f"{BIGKINDS_BASE}/api/news/download.do",
                        headers=headers_dl,
                        data=form_data.encode("utf-8"),
                        timeout=30,
                        stream=True,
                    )
                    resp.raise_for_status()

                    content_type = resp.headers.get("Content-Type", "")
                    content_disp = resp.headers.get("Content-Disposition", "")
                    # 본문은 읽지 않고 헤더만 확인 후 닫기
                    resp.close()

                    if "html" in content_type.lower() or (
                        not content_disp and "octet" not in content_type.lower()
                        and "excel" not in content_type.lower()
                        and "spreadsheet" not in content_type.lower()
                    ):
                        st.error(
                            "쿠키 만료 또는 무효 — BigKinds에 다시 로그인하세요.\n\n"
                            f"응답 Content-Type: `{content_type}`"
                        )
                    else:
                        st.success("쿠키 유효! (다운로드 인증 확인)")

                except requests.exceptions.ConnectionError:
                    st.error("BigKinds 서버에 연결할 수 없습니다.")
                except Exception as e:
                    st.error(f"검증 실패: {e}")

st.divider()

# ── 2. 뉴스 수집 ──
st.subheader("뉴스 수집")

col_start, col_end = st.columns(2)
with col_start:
    start_date = st.date_input(
        "시작일",
        value=datetime.now().date() - timedelta(days=7),
        max_value=datetime.now().date(),
    )
with col_end:
    end_date = st.date_input(
        "종료일",
        value=datetime.now().date() - timedelta(days=1),
        max_value=datetime.now().date(),
    )

if start_date > end_date:
    st.error("시작일이 종료일보다 늦습니다.")
    st.stop()

# 범위 내 모든 날짜
all_dates: list[str] = []
d = start_date
while d <= end_date:
    all_dates.append(d.strftime("%Y-%m-%d"))
    d += timedelta(days=1)

collected_dates = _scan_collected_dates(DOWNLOAD_DIR)
missing_dates = [dt for dt in all_dates if dt not in collected_dates]
existing_dates = [dt for dt in all_dates if dt in collected_dates]

# 스코어링 현황 확인
_scores_csv = ROOT / "data" / "outputs" / "daily_scores.csv"
_scored_dates = set()
if _scores_csv.exists():
    try:
        _scored_dates = set(pd.read_csv(_scores_csv, usecols=["date"])["date"].tolist())
    except Exception:
        pass
unscored_dates = [dt for dt in existing_dates if dt not in _scored_dates]

# xlsx 없지만 코퍼스에 있는 날짜 구분
_xlsx_dates = set()
if DOWNLOAD_DIR.exists():
    for f in DOWNLOAD_DIR.glob("bigkinds_*.xlsx"):
        match = re.search(r"bigkinds_(\d{4}-\d{2}-\d{2})_", f.name)
        if match and f.stat().st_size > 1024:
            _xlsx_dates.add(match.group(1))
corpus_only_dates = [dt for dt in existing_dates if dt not in _xlsx_dates]

col_m1, col_m2, col_m3, col_m4 = st.columns(4)
with col_m1:
    st.metric("전체", f"{len(all_dates)}일")
with col_m2:
    st.metric("데이터 있음", f"{len(existing_dates)}일")
with col_m3:
    st.metric("미수집", f"{len(missing_dates)}일")
with col_m4:
    st.metric("미스코어링", f"{len(unscored_dates)}일")

if missing_dates:
    with st.expander(f"미수집 날짜 ({len(missing_dates)}일)"):
        st.write(", ".join(missing_dates))
elif corpus_only_dates:
    st.info(f"선택 범위의 모든 날짜에 데이터가 있습니다. (코퍼스 전용: {len(corpus_only_dates)}일)")
else:
    st.info("선택한 범위의 모든 날짜가 이미 수집되었습니다.")

if unscored_dates:
    with st.expander(f"미스코어링 날짜 ({len(unscored_dates)}일)"):
        st.write(", ".join(unscored_dates))

# ── 빠진 날짜 자동 채우기 ──
if st.button(
    f"빠진 날짜 자동 채우기 ({len(missing_dates)}일)",
    type="primary",
    disabled=len(missing_dates) == 0,
    key="btn_batch_collect",
):
    cookie_text = ""
    if COOKIE_FILE.exists():
        cookie_text = COOKIE_FILE.read_text(encoding="utf-8").strip()
    if not cookie_text:
        st.error("쿠키가 설정되지 않았습니다. 먼저 쿠키를 저장하세요.")
        st.stop()

    from src.daily_pipeline.run import collect_news

    progress_bar = st.progress(0)
    status_text = st.empty()

    report_rows: list[dict] = []
    n_collected = 0
    n_zero = 0
    n_failed = 0

    # 기존 날짜 리포트
    for date_str in existing_dates:
        file_path = DOWNLOAD_DIR / f"bigkinds_{date_str}_{date_str}.xlsx"
        size_kb = file_path.stat().st_size / 1024 if file_path.exists() else None
        report_rows.append({
            "날짜": date_str,
            "상태": "⏭️ 기존",
            "기사 수": "-",
            "파일 크기": f"{size_kb:,.0f} KB" if size_kb else "-",
        })

    # 빠진 날짜 수집
    for i, date_str in enumerate(missing_dates):
        status_text.info(f"수집 중: {date_str} ({i + 1}/{len(missing_dates)})")
        progress_bar.progress((i + 1) / len(missing_dates))

        try:
            xlsx_path = collect_news(date_str, DOWNLOAD_DIR)
            if xlsx_path is not None:
                size_kb = xlsx_path.stat().st_size / 1024
                try:
                    from openpyxl import load_workbook
                    wb = load_workbook(xlsx_path, read_only=True)
                    ws = wb.active
                    article_count = sum(1 for _ in ws.rows) - 1  # 헤더 제외
                    wb.close()
                except Exception:
                    article_count = None
                report_rows.append({
                    "날짜": date_str,
                    "상태": "✅ 수집완료",
                    "기사 수": f"{article_count:,}" if isinstance(article_count, int) else "-",
                    "파일 크기": f"{size_kb:,.0f} KB",
                })
                n_collected += 1
            else:
                report_rows.append({
                    "날짜": date_str,
                    "상태": "⬜ 0건 (휴일)",
                    "기사 수": "0",
                    "파일 크기": "-",
                })
                n_zero += 1
        except Exception:
            report_rows.append({
                "날짜": date_str,
                "상태": "❌ 실패",
                "기사 수": "-",
                "파일 크기": "-",
            })
            n_failed += 1

    progress_bar.progress(1.0)
    status_text.success(
        f"수집 완료! 신규 {n_collected}일 · 0건 {n_zero}일 · 실패 {n_failed}일"
    )

    # 날짜순 정렬 후 세션에 저장
    report_rows.sort(key=lambda r: r["날짜"])
    st.session_state["collect_report"] = report_rows

# 리포트 표시 (세션에 저장된 경우)
if "collect_report" in st.session_state:
    st.dataframe(
        pd.DataFrame(st.session_state["collect_report"]),
        hide_index=True,
        use_container_width=True,
    )

st.divider()

# ── 3. 태깅 + 스코어링 ──
st.subheader("태깅 + 스코어링")

col_tag_date, col_tag_mode = st.columns(2)
with col_tag_date:
    tag_date = st.date_input(
        "처리 날짜",
        value=datetime.now().date() - timedelta(days=1),
        max_value=datetime.now().date(),
        key="tag_date",
    )
with col_tag_mode:
    score_only = st.checkbox("스코어만 재계산 (태깅 스킵)")

if st.button("태깅 + 스코어링 실행", type="primary", key="btn_tag_score"):
    date_str = tag_date.strftime("%Y-%m-%d")
    xlsx_path = DOWNLOAD_DIR / f"bigkinds_{date_str}_{date_str}.xlsx"

    if not score_only and not xlsx_path.exists():
        st.error(f"{date_str}의 수집 파일이 없습니다. 먼저 뉴스를 수집하세요.")
        st.stop()

    log_container = st.empty()
    progress_bar = st.progress(0)

    try:
        from src.utils import load_config, get_path
        from src.daily_pipeline.run import process_and_tag, score

        config = load_config()
        processed_path = get_path(config, "processed_data")
        outputs_path = get_path(config, "outputs")
        daily_dir = processed_path / "corpus_daily"
        ref_date = pd.Timestamp(date_str)

        if not score_only:
            with st.spinner(f"Step 1/2: {date_str} 임베딩 + 태깅 중 (GPU, 1-2분)..."):
                log_container.info("임베딩 모델 로드 + 태깅 중...")
                process_and_tag(xlsx_path, daily_dir, config)
                progress_bar.progress(50)
            log_container.info("태깅 완료!")
        else:
            progress_bar.progress(50)

        with st.spinner("Step 2/2: 리스크 스코어 산출 중..."):
            import pyarrow.parquet as pq

            if daily_dir.exists() and list(daily_dir.glob("*.parquet")):
                latest_daily = sorted(daily_dir.glob("*.parquet"))[-1]
                scoring_date = pd.Timestamp(latest_daily.stem)
            else:
                pf = pq.ParquetFile(processed_path / "corpus_tagged.parquet")
                last_rg = pf.read_row_group(
                    pf.metadata.num_row_groups - 1, columns=["date"]
                )
                last_df = last_rg.to_pandas()
                last_df["date"] = pd.to_datetime(last_df["date"])
                scoring_date = last_df["date"].max()

            log_container.info(f"스코어링 기준일: {scoring_date.strftime('%Y-%m-%d')}")
            result = score(processed_path, outputs_path, daily_dir, scoring_date, config)
            progress_bar.progress(100)

        st.success("태깅 + 스코어링 완료!")
        risk_score = result["risk_score"]
        status = result["status"]

        col_r1, col_r2 = st.columns(2)
        with col_r1:
            st.metric("Risk Score", f"{risk_score:.2f}/100")
        with col_r2:
            st.metric("Status", status)

        if status == "WARNING":
            st.warning(f"리스크 스코어 {risk_score:.2f} — WARNING!")

    except Exception as e:
        progress_bar.progress(100)
        st.error(f"오류: {e}")
        import traceback
        st.code(traceback.format_exc())

st.divider()

# ── 4. 수집 이력 ──
st.subheader("수집 이력")

if DOWNLOAD_DIR.exists():
    files = sorted(DOWNLOAD_DIR.glob("*.xlsx"), reverse=True)
    if files:
        file_data = []
        for f in files[:30]:
            size_kb = f.stat().st_size / 1024
            file_data.append({
                "파일명": f.name,
                "크기": f"{size_kb:.0f} KB",
                "수정일": datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
            })
        st.dataframe(pd.DataFrame(file_data), hide_index=True, width="stretch")
    else:
        st.info("수집된 파일이 없습니다.")
else:
    st.info("수집 디렉토리가 아직 없습니다.")
