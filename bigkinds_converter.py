"""
BIGKinds xlsx → Parquet 변환기
빅카인즈에서 다운로드한 xlsx 파일들을 전처리하여 단일 parquet 파일로 변환하는 GUI 도구
"""

import json
import os
import re
import sys
import threading
import tkinter as tk
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import fastexcel
import pandas as pd

# ---------------------------------------------------------------------------
# 상수
# ---------------------------------------------------------------------------
APP_TITLE = "BIGKinds xlsx → Parquet 변환기"
STATE_FILE = Path(__file__).with_name(".bigkinds_converter_state.json")

# BIGKinds 원본 컬럼 → 변환 후 컬럼
COLUMN_MAP = {
    "뉴스 식별자": "article_id",
    "일자": "date",
    "제목": "title",
    "본문": "lead",
}
EXCLUDE_COL = "분석제외 여부"

# GUI 색상
COLOR_BG = "#1e1e2e"
COLOR_SURFACE = "#2a2a3d"
COLOR_BORDER = "#3a3a5c"
COLOR_TEXT = "#e0e0e0"
COLOR_TEXT_DIM = "#8888aa"
COLOR_ACCENT = "#7c5cfc"
COLOR_ACCENT_HOVER = "#9b7fff"
COLOR_SUCCESS = "#50c878"
COLOR_PENDING = "#f0ad4e"
COLOR_DROP_ACTIVE = "#3d3d6b"
COLOR_DELETE = "#e05555"
COLOR_DELETE_HOVER = "#ff6b6b"


# ---------------------------------------------------------------------------
# 유틸리티
# ---------------------------------------------------------------------------
def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"files": [], "completed": [], "output_dir": ""}


def save_state(state: dict):
    STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def format_date(val) -> str:
    """YYYYMMDD (int/str) → YYYY-MM-DD"""
    s = str(int(val))
    if len(s) == 8:
        return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
    return s


# ---------------------------------------------------------------------------
# 전처리 로직
# ---------------------------------------------------------------------------
NEED_COLS = list(COLUMN_MAP.keys()) + [EXCLUDE_COL]
MAX_WORKERS = min(6, (os.cpu_count() or 2))


def _read_one_xlsx(fp: str) -> pd.DataFrame | None:
    """fastexcel로 단일 xlsx 파일을 읽어 전처리된 DataFrame을 반환한다."""
    try:
        reader = fastexcel.read_excel(fp)
        try:
            sheet = reader.load_sheet(0, use_columns=NEED_COLS)
        except fastexcel.ColumnNotFoundError:
            # 분석제외 여부 컬럼이 없는 경우 필수 컬럼만으로 재시도
            sheet = reader.load_sheet(0, use_columns=list(COLUMN_MAP.keys()))
        df = sheet.to_pandas()
    except Exception:
        return None

    # 필수 컬럼 존재 확인
    missing = [c for c in COLUMN_MAP if c not in df.columns]
    if missing:
        return None

    # 분석제외 행 제거
    if EXCLUDE_COL in df.columns:
        df = df[df[EXCLUDE_COL].isna()]
        df = df.drop(columns=[EXCLUDE_COL])

    # 필요한 컬럼만 추출 및 이름 변환
    df = df[list(COLUMN_MAP.keys())].rename(columns=COLUMN_MAP)

    # date 형식 변환 (벡터화)
    df["date"] = df["date"].astype(str).str.replace(r"\.0$", "", regex=True)
    df["date"] = (
        df["date"].str[:4] + "-" + df["date"].str[4:6] + "-" + df["date"].str[6:8]
    )

    # article_id를 문자열로 통일
    df["article_id"] = df["article_id"].astype(str)

    return df


_FILE_DATE_RE = re.compile(r"bigkinds_(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})\.xlsx$")


def detect_coverage_gaps(file_paths: list[str]) -> dict:
    """
    파일명에서 날짜 범위를 파싱하여 누락 구간을 탐지한다.

    Returns:
        {
            "total_files": int,
            "parsed_files": int,
            "coverage_start": str | None,
            "coverage_end": str | None,
            "gaps": [{"from": str, "to": str, "missing_days": int}, ...],
            "overlaps": [{"file_a": str, "file_b": str, "overlap_days": int}, ...],
        }
    """
    from datetime import date, timedelta

    ranges = []
    for fp in file_paths:
        name = os.path.basename(fp)
        m = _FILE_DATE_RE.search(name)
        if m:
            s = date.fromisoformat(m.group(1))
            e = date.fromisoformat(m.group(2))
            ranges.append((s, e, name))

    result = {
        "total_files": len(file_paths),
        "parsed_files": len(ranges),
        "coverage_start": None,
        "coverage_end": None,
        "gaps": [],
        "overlaps": [],
    }

    if not ranges:
        return result

    ranges.sort()
    result["coverage_start"] = str(ranges[0][0])
    result["coverage_end"] = str(ranges[-1][1])

    for i in range(1, len(ranges)):
        prev_end = ranges[i - 1][1]
        curr_start = ranges[i][0]
        gap_days = (curr_start - prev_end).days

        if gap_days > 1:
            # 빠진 구간: 이전 파일 종료일 다음날 ~ 현재 파일 시작일 전날
            result["gaps"].append({
                "from": str(prev_end + timedelta(days=1)),
                "to": str(curr_start - timedelta(days=1)),
                "missing_days": gap_days - 1,
                "after_file": ranges[i - 1][2],
                "before_file": ranges[i][2],
            })
        elif gap_days < 1:
            result["overlaps"].append({
                "file_a": ranges[i - 1][2],
                "file_b": ranges[i][2],
                "overlap_days": 1 - gap_days,
            })

    return result


def process_files(file_paths: list[str], output_path: str, progress_cb=None, log_cb=None):
    """
    xlsx 파일 목록을 전처리하여 하나의 parquet로 저장한다.
    fastexcel + ThreadPoolExecutor로 병렬 처리.
    Returns: (total_articles, duplicates_removed, final_count, processed_files)
    """
    # ── 변환 전 날짜 커버리지 갭 검사 ──
    gap_report = detect_coverage_gaps(file_paths)
    if log_cb:
        parsed = gap_report["parsed_files"]
        log_cb(f"날짜 파싱: {parsed}/{len(file_paths)}개 파일 "
               f"({gap_report['coverage_start']} ~ {gap_report['coverage_end']})")
        if gap_report["gaps"]:
            log_cb(f"⚠ 누락 구간 {len(gap_report['gaps'])}건 발견:")
            for g in gap_report["gaps"]:
                log_cb(f"  ▸ {g['from']} ~ {g['to']} ({g['missing_days']}일)")
        else:
            log_cb("날짜 커버리지 검사 통과 — 누락 구간 없음")
        if gap_report["overlaps"]:
            log_cb(f"⚠ 날짜 겹침 {len(gap_report['overlaps'])}건 발견")

    frames = []
    processed = []
    total = len(file_paths)
    done_count = 0

    if log_cb:
        log_cb(f"처리 시작: {total}개 파일 (워커 {MAX_WORKERS}개)")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_fp = {executor.submit(_read_one_xlsx, fp): fp for fp in file_paths}

        for future in as_completed(future_to_fp):
            fp = future_to_fp[future]
            done_count += 1

            try:
                df = future.result()
            except Exception as e:
                if log_cb:
                    log_cb(f"[{done_count}/{total}] ⚠ {Path(fp).name}: {e}")
                continue

            if df is None:
                if log_cb:
                    log_cb(f"[{done_count}/{total}] ⚠ 건너뜀: {Path(fp).name}")
                continue

            frames.append(df)
            processed.append(fp)

            if progress_cb:
                progress_cb(done_count / total)
            if log_cb and done_count % 50 == 0:
                log_cb(f"[{done_count}/{total}] 처리 중...")

    if not frames:
        raise ValueError("처리된 파일이 없습니다. xlsx 파일과 컬럼 형식을 확인하세요.")

    if log_cb:
        log_cb("파일 병합 및 중복 제거 중...")

    merged = pd.concat(frames, ignore_index=True)
    total_articles = len(merged)

    merged.drop_duplicates(subset="article_id", inplace=True)
    final_count = len(merged)
    duplicates_removed = total_articles - final_count

    # 날짜순 정렬
    merged.sort_values("date", inplace=True)
    merged.reset_index(drop=True, inplace=True)

    merged.to_parquet(output_path, index=False, engine="pyarrow")

    return total_articles, duplicates_removed, final_count, processed


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("820x700")
        self.minsize(700, 550)
        self.configure(bg=COLOR_BG)
        self.resizable(True, True)

        # 상태 복원
        state = load_state()
        self.file_list: list[str] = []
        self.completed_set: set[str] = set(state.get("completed", []))
        self.output_dir = state.get("output_dir", "")
        self._processing = False

        # 저장된 파일 중 아직 존재하는 것만 복원
        for f in state.get("files", []):
            if os.path.isfile(f) and f not in self.file_list:
                self.file_list.append(f)

        self._build_ui()
        self._refresh_file_list()
        self._setup_dnd()

    # ── UI 구성 ──────────────────────────────────────────────────────────
    def _build_ui(self):
        style = ttk.Style(self)
        style.theme_use("clam")

        # Configure ttk styles
        style.configure("TFrame", background=COLOR_BG)
        style.configure("Surface.TFrame", background=COLOR_SURFACE)
        style.configure("TLabel", background=COLOR_BG, foreground=COLOR_TEXT, font=("Segoe UI", 10))
        style.configure("Title.TLabel", font=("Segoe UI", 13, "bold"), foreground=COLOR_TEXT)
        style.configure("Dim.TLabel", foreground=COLOR_TEXT_DIM, font=("Segoe UI", 9))
        style.configure("Status.TLabel", foreground=COLOR_SUCCESS, font=("Segoe UI", 10, "bold"))
        style.configure("Accent.TButton", background=COLOR_ACCENT, foreground="white",
                        font=("Segoe UI", 10, "bold"), padding=(16, 8))
        style.map("Accent.TButton",
                  background=[("active", COLOR_ACCENT_HOVER), ("disabled", COLOR_BORDER)])
        style.configure("Secondary.TButton", background=COLOR_SURFACE, foreground=COLOR_TEXT,
                        font=("Segoe UI", 9), padding=(10, 5))
        style.map("Secondary.TButton",
                  background=[("active", COLOR_BORDER)])
        style.configure("Delete.TButton", background=COLOR_DELETE, foreground="white",
                        font=("Segoe UI", 9), padding=(8, 4))
        style.map("Delete.TButton",
                  background=[("active", COLOR_DELETE_HOVER)])

        # Progress bar style
        style.configure("Accent.Horizontal.TProgressbar",
                        troughcolor=COLOR_SURFACE, background=COLOR_ACCENT,
                        thickness=12)

        pad = {"padx": 12, "pady": 4}

        # ── 상단: 타이틀 ─────────────────────────────────────────────────
        header = ttk.Frame(self)
        header.pack(fill="x", padx=16, pady=(14, 4))
        ttk.Label(header, text=APP_TITLE, style="Title.TLabel").pack(side="left")

        # ── 드래그 앤 드롭 영역 ──────────────────────────────────────────
        self.drop_frame = tk.Frame(self, bg=COLOR_SURFACE, highlightbackground=COLOR_BORDER,
                                   highlightthickness=2, cursor="hand2")
        self.drop_frame.pack(fill="x", padx=16, pady=8, ipady=18)

        self.drop_label = tk.Label(
            self.drop_frame, text="여기에 xlsx 파일 또는 폴더를 드래그 & 드롭 하세요",
            bg=COLOR_SURFACE, fg=COLOR_TEXT_DIM, font=("Segoe UI", 11))
        self.drop_label.pack(expand=True)

        sub = tk.Label(self.drop_frame, text="또는 클릭하여 파일 선택",
                       bg=COLOR_SURFACE, fg=COLOR_TEXT_DIM, font=("Segoe UI", 9))
        sub.pack()

        # 클릭하면 파일 선택 대화상자
        for w in (self.drop_frame, self.drop_label, sub):
            w.bind("<Button-1>", lambda e: self._browse_files())

        # ── 파일 목록 ────────────────────────────────────────────────────
        list_header = ttk.Frame(self)
        list_header.pack(fill="x", padx=16, pady=(6, 0))
        self.file_count_label = ttk.Label(list_header, text="파일 목록 (0개)", style="TLabel")
        self.file_count_label.pack(side="left")
        ttk.Button(list_header, text="전체 삭제", style="Delete.TButton",
                   command=self._clear_files).pack(side="right")
        ttk.Button(list_header, text="선택 삭제", style="Secondary.TButton",
                   command=self._remove_selected).pack(side="right", padx=(0, 6))

        list_frame = tk.Frame(self, bg=COLOR_SURFACE, highlightbackground=COLOR_BORDER,
                              highlightthickness=1)
        list_frame.pack(fill="both", expand=True, padx=16, pady=6)

        # Treeview for file list
        cols = ("status", "filename", "path")
        self.tree = ttk.Treeview(list_frame, columns=cols, show="headings",
                                 selectmode="extended", height=10)
        self.tree.heading("status", text="상태")
        self.tree.heading("filename", text="파일명")
        self.tree.heading("path", text="경로")
        self.tree.column("status", width=70, minwidth=60, anchor="center")
        self.tree.column("filename", width=300, minwidth=150)
        self.tree.column("path", width=400, minwidth=200)

        # Treeview style
        style.configure("Treeview",
                        background=COLOR_SURFACE, foreground=COLOR_TEXT,
                        fieldbackground=COLOR_SURFACE, font=("Segoe UI", 9),
                        rowheight=26)
        style.configure("Treeview.Heading",
                        background=COLOR_BG, foreground=COLOR_TEXT,
                        font=("Segoe UI", 9, "bold"))
        style.map("Treeview", background=[("selected", COLOR_ACCENT)])

        # Tag colors
        self.tree.tag_configure("done", foreground=COLOR_SUCCESS)
        self.tree.tag_configure("pending", foreground=COLOR_PENDING)

        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        self.tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # ── 출력 설정 / 실행 ─────────────────────────────────────────────
        ctrl_frame = ttk.Frame(self)
        ctrl_frame.pack(fill="x", padx=16, pady=6)

        ttk.Label(ctrl_frame, text="저장 경로:").pack(side="left")
        self.output_var = tk.StringVar(value=self.output_dir)
        self.output_entry = tk.Entry(ctrl_frame, textvariable=self.output_var,
                                     bg=COLOR_SURFACE, fg=COLOR_TEXT,
                                     insertbackground=COLOR_TEXT,
                                     font=("Segoe UI", 9), relief="flat",
                                     highlightbackground=COLOR_BORDER,
                                     highlightthickness=1)
        self.output_entry.pack(side="left", fill="x", expand=True, padx=6)
        ttk.Button(ctrl_frame, text="찾아보기", style="Secondary.TButton",
                   command=self._browse_output).pack(side="left")

        # ── 프로그레스 바 ────────────────────────────────────────────────
        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_bar = ttk.Progressbar(
            self, variable=self.progress_var, maximum=1.0,
            style="Accent.Horizontal.TProgressbar")
        self.progress_bar.pack(fill="x", padx=16, pady=(4, 2))

        self.progress_label = ttk.Label(self, text="", style="Dim.TLabel")
        self.progress_label.pack(padx=16, anchor="w")

        # ── 실행 버튼 & 결과 ─────────────────────────────────────────────
        bottom = ttk.Frame(self)
        bottom.pack(fill="x", padx=16, pady=(2, 12))

        self.run_btn = ttk.Button(bottom, text="변환 시작", style="Accent.TButton",
                                  command=self._start_conversion)
        self.run_btn.pack(side="left")

        self.result_label = ttk.Label(bottom, text="", style="Status.TLabel")
        self.result_label.pack(side="left", padx=16)

    # ── Drag & Drop ─────────────────────────────────────────────────────
    def _setup_dnd(self):
        try:
            from tkinterdnd2 import DND_FILES, TkinterDnD  # noqa: F811

            # tkinterdnd2 requires the root window to be TkinterDnD.
            # Since we already created a tk.Tk, we monkey-patch DnD support.
            # A cleaner approach: recreate with TkinterDnD, but here we
            # register the existing Tk instance.
            if not hasattr(self, "drop_target_register"):
                # Try registering manually
                try:
                    self.tk.call("package", "require", "tkdnd")
                except tk.TclError:
                    self._dnd_fallback()
                    return

                # If tkdnd loaded, register drop targets
                self._register_dnd_target(self.drop_frame, DND_FILES)
                self._register_dnd_target(self.drop_label, DND_FILES)
            else:
                self.drop_target_register(DND_FILES)
                self.dnd_bind("<<Drop>>", self._on_drop)
        except ImportError:
            self._dnd_fallback()

    def _register_dnd_target(self, widget, dnd_type):
        """Register a widget as a DnD drop target using raw Tcl calls."""
        try:
            widget_path = str(widget)
            self.tk.call("tkdnd::drop_target", "register", widget_path, dnd_type)

            # Bind drop event
            def _on_drop_tcl(*args):
                # args contains the data as a Tcl list
                data = args[0] if args else ""
                self._on_drop_data(data)
                return "copy"

            cb_name = self.register(_on_drop_tcl)
            self.tk.call("bind", widget_path, "<<Drop:DND_Files>>",
                         cb_name + " %D")

            # Visual feedback on drag enter/leave
            def _on_enter(*args):
                self.drop_frame.configure(bg=COLOR_DROP_ACTIVE)
                self.drop_label.configure(bg=COLOR_DROP_ACTIVE)
                return "copy"

            def _on_leave(*args):
                self.drop_frame.configure(bg=COLOR_SURFACE)
                self.drop_label.configure(bg=COLOR_SURFACE)

            enter_cb = self.register(_on_enter)
            leave_cb = self.register(_on_leave)
            self.tk.call("bind", widget_path, "<<DragEnter>>", enter_cb)
            self.tk.call("bind", widget_path, "<<DragLeave>>", leave_cb)
        except tk.TclError:
            self._dnd_fallback()

    def _on_drop(self, event):
        self._on_drop_data(event.data)

    def _on_drop_data(self, data: str):
        """Process dropped file paths."""
        # Tcl may wrap paths with braces for paths containing spaces
        paths = []
        raw = data.strip()
        if not raw:
            return

        # Parse Tcl list format: {path with spaces} normalpath ...
        i = 0
        while i < len(raw):
            if raw[i] == "{":
                end = raw.index("}", i)
                paths.append(raw[i + 1 : end])
                i = end + 2
            elif raw[i] == " ":
                i += 1
            else:
                end = raw.find(" ", i)
                if end == -1:
                    end = len(raw)
                paths.append(raw[i:end])
                i = end + 1

        self._add_paths(paths)
        self.drop_frame.configure(bg=COLOR_SURFACE)
        self.drop_label.configure(bg=COLOR_SURFACE)

    def _dnd_fallback(self):
        """DnD 사용 불가 시 안내 메시지"""
        self.drop_label.configure(
            text="클릭하여 xlsx 파일을 선택하세요 (Drag & Drop 미지원 환경)")

    # ── 파일 관리 ────────────────────────────────────────────────────────
    def _add_paths(self, paths: list[str]):
        added = 0
        for p in paths:
            p = os.path.normpath(p)
            if os.path.isdir(p):
                for f in sorted(Path(p).rglob("*.xlsx")):
                    fp = str(f)
                    if fp not in self.file_list:
                        self.file_list.append(fp)
                        added += 1
            elif p.lower().endswith(".xlsx") and os.path.isfile(p):
                if p not in self.file_list:
                    self.file_list.append(p)
                    added += 1
        if added:
            self._refresh_file_list()
            self._save_state()

    def _browse_files(self):
        files = filedialog.askopenfilenames(
            title="xlsx 파일 선택",
            filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")])
        if files:
            self._add_paths(list(files))

    def _browse_output(self):
        path = filedialog.askdirectory(title="저장 폴더 선택")
        if path:
            self.output_var.set(path)
            self.output_dir = path
            self._save_state()

    def _remove_selected(self):
        selected = self.tree.selection()
        if not selected:
            return
        for item in selected:
            vals = self.tree.item(item, "values")
            fp = vals[2]  # path column
            if fp in self.file_list:
                self.file_list.remove(fp)
                self.completed_set.discard(fp)
        self._refresh_file_list()
        self._save_state()

    def _clear_files(self):
        if not self.file_list:
            return
        if messagebox.askyesno("확인", "파일 목록을 모두 삭제하시겠습니까?"):
            self.file_list.clear()
            self.completed_set.clear()
            self._refresh_file_list()
            self._save_state()

    def _refresh_file_list(self):
        self.tree.delete(*self.tree.get_children())
        for fp in self.file_list:
            name = Path(fp).name
            done = fp in self.completed_set
            tag = "done" if done else "pending"
            status = "✓ 완료" if done else "● 대기"
            self.tree.insert("", "end", values=(status, name, fp), tags=(tag,))
        self.file_count_label.configure(text=f"파일 목록 ({len(self.file_list)}개)")

    def _save_state(self):
        save_state({
            "files": self.file_list,
            "completed": list(self.completed_set),
            "output_dir": self.output_var.get(),
        })

    # ── 변환 실행 ────────────────────────────────────────────────────────
    def _start_conversion(self):
        if self._processing:
            return

        pending = [f for f in self.file_list if f not in self.completed_set]
        if not pending:
            messagebox.showinfo("알림", "처리할 새 파일이 없습니다.")
            return

        output_dir = self.output_var.get().strip()
        if not output_dir:
            messagebox.showwarning("경고", "저장 경로를 선택하세요.")
            return

        # ── 변환 전 날짜 커버리지 갭 검사 ──
        gap_report = detect_coverage_gaps(pending)
        if gap_report["gaps"]:
            gap_lines = [f"  • {g['from']} ~ {g['to']} ({g['missing_days']}일)"
                         for g in gap_report["gaps"]]
            gap_msg = (
                f"날짜 커버리지에 누락 구간 {len(gap_report['gaps'])}건이 발견되었습니다:\n\n"
                + "\n".join(gap_lines[:20])
                + ("\n  ..." if len(gap_lines) > 20 else "")
                + "\n\n그래도 변환을 계속하시겠습니까?"
            )
            if not messagebox.askyesno("누락 구간 경고", gap_msg, icon="warning"):
                return

        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f"bigkinds_{timestamp}.parquet")

        self._processing = True
        self.run_btn.configure(state="disabled")
        self.result_label.configure(text="")
        self.progress_var.set(0.0)
        self.progress_label.configure(text="변환 중...")

        def _run():
            try:
                total, dups, final, processed = process_files(
                    pending, output_path,
                    progress_cb=lambda v: self.after(0, self._update_progress, v),
                    log_cb=lambda msg: self.after(0, self._update_log, msg),
                )
                # 완료된 파일 기록
                for fp in processed:
                    self.completed_set.add(fp)
                self.after(0, self._on_done, total, dups, final, output_path)
            except Exception as e:
                self.after(0, self._on_error, str(e))

        threading.Thread(target=_run, daemon=True).start()

    def _update_progress(self, value: float):
        self.progress_var.set(value)
        pct = int(value * 100)
        self.progress_label.configure(text=f"진행률: {pct}%")

    def _update_log(self, msg: str):
        self.progress_label.configure(text=msg)

    def _on_done(self, total, dups, final, output_path):
        self._processing = False
        self.run_btn.configure(state="normal")
        self.progress_var.set(1.0)

        self.result_label.configure(
            text=f"총 {total:,}건 → 중복 제거 {dups:,}건 → 최종 {final:,}건")
        self.progress_label.configure(text=f"저장 완료: {output_path}")

        self._refresh_file_list()
        self._save_state()

        messagebox.showinfo("완료",
                            f"변환이 완료되었습니다.\n\n"
                            f"총 기사:  {total:,}건\n"
                            f"중복 제거: {dups:,}건\n"
                            f"최종 건수: {final:,}건\n\n"
                            f"저장: {output_path}")

    def _on_error(self, msg: str):
        self._processing = False
        self.run_btn.configure(state="normal")
        self.progress_var.set(0.0)
        self.progress_label.configure(text="오류 발생")
        messagebox.showerror("오류", msg)


# ---------------------------------------------------------------------------
# tkinterdnd2 호환 루트 윈도우
# ---------------------------------------------------------------------------
def main():
    """DnD 지원 여부에 따라 적절한 윈도우를 생성"""
    try:
        # tkinterdnd2의 TkinterDnD를 루트로 사용하면 네이티브 DnD가 동작
        from tkinterdnd2 import TkinterDnD

        class DnDApp(TkinterDnD.Tk, App.__mro__[0]):
            """TkinterDnD.Tk 기반 App"""

            def __init__(self):
                TkinterDnD.Tk.__init__(self)
                # App.__init__에서 tk.Tk.__init__을 다시 호출하지 않도록
                # 직접 초기화 코드를 실행
                self.title(APP_TITLE)
                self.geometry("820x700")
                self.minsize(700, 550)
                self.configure(bg=COLOR_BG)
                self.resizable(True, True)

                state = load_state()
                self.file_list: list[str] = []
                self.completed_set: set[str] = set(state.get("completed", []))
                self.output_dir = state.get("output_dir", "")
                self._processing = False

                for f in state.get("files", []):
                    if os.path.isfile(f) and f not in self.file_list:
                        self.file_list.append(f)

                self._build_ui()
                self._refresh_file_list()
                self._setup_dnd_native()

            def _setup_dnd_native(self):
                """TkinterDnD 네이티브 DnD 등록"""
                from tkinterdnd2 import DND_FILES

                for widget in (self.drop_frame, self.drop_label):
                    widget.drop_target_register(DND_FILES)
                    widget.dnd_bind("<<DropEnter>>", self._dnd_enter)
                    widget.dnd_bind("<<DropLeave>>", self._dnd_leave)
                    widget.dnd_bind("<<Drop>>", self._dnd_drop)

                # 전체 윈도우에도 등록
                self.drop_target_register(DND_FILES)
                self.dnd_bind("<<Drop>>", self._dnd_drop)

            def _dnd_enter(self, event):
                self.drop_frame.configure(bg=COLOR_DROP_ACTIVE)
                self.drop_label.configure(bg=COLOR_DROP_ACTIVE)
                return event.action

            def _dnd_leave(self, event):
                self.drop_frame.configure(bg=COLOR_SURFACE)
                self.drop_label.configure(bg=COLOR_SURFACE)

            def _dnd_drop(self, event):
                self.drop_frame.configure(bg=COLOR_SURFACE)
                self.drop_label.configure(bg=COLOR_SURFACE)
                self._on_drop_data(event.data)
                return event.action

        # App의 메서드를 DnDApp에 복사
        for attr in dir(App):
            if attr.startswith("_") and not attr.startswith("__"):
                if attr not in ("_setup_dnd_native", "_dnd_enter", "_dnd_leave", "_dnd_drop",
                                "__init__"):
                    method = getattr(App, attr)
                    if callable(method):
                        setattr(DnDApp, attr, method)

        app = DnDApp()
        app.mainloop()

    except ImportError:
        # tkinterdnd2 없으면 기본 tk로
        app = App()
        app.mainloop()


if __name__ == "__main__":
    main()
