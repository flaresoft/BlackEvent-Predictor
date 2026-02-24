"""
빅카인즈(BIGKinds) 뉴스 데이터 자동 다운로드 스크립트 v2

변경점: 1개월 단위로 쪼개고, 건수 확인 즉시 다운로드 (파일이 바로바로 생김)

사용법:
1. 브라우저에서 bigkinds.or.kr 로그인
2. F12 → Application → Cookies → Bigkinds 값 복사
3. 아래 SESSION_COOKIE에 붙여넣기
4. python bigkinds_downloader.py 실행
"""

import requests
import json
import time
import os
import urllib.parse
from datetime import datetime, timedelta
from pathlib import Path

# ============================================================
# 🔧 설정 - 이것만 바꾸면 됨
# ============================================================

# F12 → Network → 아무 요청 → Headers → Cookie 값 전체 복사해서 붙여넣기
COOKIE_STRING = "NCPVPCLBTG=7e3126e4020ff3044fad9532a9c561180183efeb2cf78a256ad85bdee9bd72b6; _gid=GA1.3.332216020.1771718329; todayPopup_135=true; Bigkinds=A4A318D3206829ECC61A8A5137552990.tomcat1; _gat=1; _ga_QWY27BS8JM=GS2.1.s1771718330$o1$g1$t1771720572$j54$l0$h0; _ga=GA1.1.1361116665.1771718329"
DOWNLOAD_DIR = Path("data/raw/bigkinds")

START_YEAR = 2011
END_YEAR = 2014

# ============================================================
# 카테고리 코드 (정치 + 경제 + 사회 전체)
# ============================================================

ALL_CATEGORY_CODES = [
    # 정치
    "001000000", "001005000", "001001000", "001004000",
    "001003000", "001007000", "001002000", "001006000",
    # 경제
    "002000000", "002004000", "002010000", "002008000",
    "002014000", "002011000", "002009000", "002005000",
    "002001000", "002012000", "002006000", "002002000",
    "002007000", "002003000", "002013000",
    # 사회
    "003000000", "003007000", "003003000", "003006000",
    "003002000", "003009000", "003001000", "003005000",
    "003010000", "003008000", "003004000",
]

# ============================================================
# API
# ============================================================

BASE_URL = "https://www.bigkinds.or.kr"

HEADERS_JSON = {
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "Content-Type": "application/json;charset=UTF-8",
    "Origin": BASE_URL,
    "Referer": f"{BASE_URL}/v2/news/search.do",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36",
    "X-Requested-With": "XMLHttpRequest",
}

HEADERS_DOWNLOAD = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Content-Type": "application/x-www-form-urlencoded",
    "Origin": BASE_URL,
    "Referer": f"{BASE_URL}/v2/news/search.do",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36",
}


def get_session():
    session = requests.Session()
    session.headers.update({
        "Cookie": COOKIE_STRING,
    })
    return session


def build_param(start_date, end_date):
    return {
        "indexName": "news",
        "searchKey": "  ",
        "searchKeys": [{}],
        "byLine": "",
        "searchFilterType": "1",
        "searchScopeType": "1",
        "searchSortType": "date",
        "sortMethod": "date",
        "mainTodayPersonYn": "",
        "startDate": start_date,
        "endDate": end_date,
        "newsIds": [],
        "categoryCodes": ALL_CATEGORY_CODES,
        "providerCodes": [],
        "incidentCodes": [],
        "networkNodeType": "",
        "topicOrigin": "",
        "dateCodes": [],
        "editorialIs": False,
        "startNo": 1,
        "resultNumber": 10,
        "isTmUsable": False,
        "isNotTmUsable": False,
    }


def get_count(session, start_date, end_date):
    """기사 건수 조회"""
    param = build_param(start_date, end_date)
    param["resultNumber"] = 1
    resp = session.post(f"{BASE_URL}/api/news/search.do", headers=HEADERS_JSON, json=param, timeout=30)
    resp.raise_for_status()
    return resp.json().get("totalCount", 0)


def download(session, start_date, end_date, save_path):
    """엑셀 다운로드"""
    param = build_param(start_date, end_date)
    param["sectionDiv"] = "1000"
    param["realURI"] = "/api/news/previewData.do"
    param["interval"] = 2

    form_data = "jsonSearchParam=" + urllib.parse.quote(json.dumps(param, ensure_ascii=False))

    resp = session.post(
        f"{BASE_URL}/api/news/download.do",
        headers=HEADERS_DOWNLOAD,
        data=form_data.encode("utf-8"),
        timeout=180,
        stream=True,
    )
    resp.raise_for_status()

    content_type = resp.headers.get("Content-Type", "")
    if "html" in content_type.lower() and len(resp.content) < 5000:
        print(f"    ❌ 세션 만료! 브라우저에서 다시 로그인하고 쿠키를 갱신하세요.")
        return False

    with open(save_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)

    size_kb = os.path.getsize(save_path) / 1024
    print(f"    ✅ 저장: {save_path.name} ({size_kb:.0f} KB)")
    return True


def split_and_download(session, start_date, end_date, depth=0):
    """
    건수 확인 → 20,000 이하면 바로 다운로드
    초과면 반으로 쪼개서 재귀 (최대 depth 5)
    """
    indent = "  " * depth

    # 이미 다운로드된 파일 확인
    filename = f"bigkinds_{start_date}_{end_date}.xlsx"
    save_path = DOWNLOAD_DIR / filename
    if save_path.exists() and os.path.getsize(save_path) > 1024:
        print(f"{indent}⏭️  {start_date} ~ {end_date}: 이미 존재, 스킵")
        return 1, 0

    # 건수 확인
    count = get_count(session, start_date, end_date)
    time.sleep(0.5)

    if count == 0:
        print(f"{indent}⬜ {start_date} ~ {end_date}: 0건, 스킵")
        return 0, 0

    if count <= 20000:
        print(f"{indent}📥 {start_date} ~ {end_date}: {count:,}건 → 다운로드...")
        ok = download(session, start_date, end_date, save_path)
        time.sleep(3)
        return (1, 0) if ok else (0, 1)

    # 20,000건 초과 → 분할
    if depth >= 5:
        print(f"{indent}⚠️  {start_date} ~ {end_date}: {count:,}건, 분할 한계 도달! 스킵")
        return 0, 1

    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    mid_dt = start_dt + (end_dt - start_dt) // 2

    mid_str = mid_dt.strftime("%Y-%m-%d")
    next_str = (mid_dt + timedelta(days=1)).strftime("%Y-%m-%d")

    print(f"{indent}🔀 {start_date} ~ {end_date}: {count:,}건 → 분할")

    s1, f1 = split_and_download(session, start_date, mid_str, depth + 1)
    s2, f2 = split_and_download(session, next_str, end_date, depth + 1)

    return s1 + s2, f1 + f2


def generate_monthly_periods(start_year, end_year):
    """월 단위 기간 리스트"""
    periods = []
    today = datetime.now()

    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            start = datetime(year, month, 1)
            if start > today:
                break
            if month == 12:
                end = datetime(year, 12, 31)
            else:
                end = datetime(year, month + 1, 1) - timedelta(days=1)
            if end > today:
                end = today
            periods.append((start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")))

    return periods


def main():
    print("=" * 60)
    print("빅카인즈 뉴스 자동 다운로드 v2")
    print("(월 단위 → 즉시 다운로드)")
    print("=" * 60)

    if COOKIE_STRING == "여기에_전체_쿠키_붙여넣기":
        print("\n❌ COOKIE_STRING을 설정하세요!")
        print("   F12 → Network → 아무 요청 → Headers → Cookie 값 전체 복사")
        return

    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    session = get_session()

    # 세션 체크
    print("\n🔍 세션 확인...")
    try:
        test = get_count(session, "2025-01-01", "2025-01-31")
        if test == 0:
            print("❌ 세션 만료. 다시 로그인 후 쿠키 갱신.")
            return
        print(f"✅ 세션 유효 (2025-01: {test:,}건)")
    except Exception as e:
        print(f"❌ 실패: {e}")
        return

    # 월별 기간
    periods = generate_monthly_periods(START_YEAR, END_YEAR)
    print(f"\n📅 {len(periods)}개월 처리 예정")

    # 다운로드
    total_s = 0
    total_f = 0

    for i, (sd, ed) in enumerate(periods, 1):
        print(f"\n[{i}/{len(periods)}] ──── {sd[:7]} ────")
        s, f = split_and_download(session, sd, ed)
        total_s += s
        total_f += f

    # 결과
    print("\n" + "=" * 60)
    files = sorted(DOWNLOAD_DIR.glob("*.xlsx"))
    total_size = sum(os.path.getsize(f) for f in files)
    print(f"📁 파일: {len(files)}개 | 💾 {total_size/1024/1024:.1f} MB")
    print(f"✅ 성공: {total_s} | ❌ 실패: {total_f}")
    print(f"📂 {DOWNLOAD_DIR.resolve()}")

    if total_f > 0:
        print(f"\n⚠️ 실패분은 쿠키 갱신 후 재실행하면 이어받기됩니다.")


if __name__ == "__main__":
    main()
