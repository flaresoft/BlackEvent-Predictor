@echo off
echo BlackEvent Daily Scorer - Task Scheduler 등록
echo.

schtasks /create ^
    /tn "BlackEvent_DailyScorer" ^
    /tr "C:\work\BlackEvent-Predictor\scripts\daily_score.bat" ^
    /sc daily ^
    /st 09:00 ^
    /f

if %errorlevel% equ 0 (
    echo.
    echo [OK] 등록 완료: 매일 09:00 실행
    echo     작업 이름: BlackEvent_DailyScorer
    echo     실행 파일: C:\work\BlackEvent-Predictor\scripts\daily_score.bat
    echo     로그 파일: C:\work\BlackEvent-Predictor\data\outputs\daily_score.log
    echo.
    echo 확인: schtasks /query /tn "BlackEvent_DailyScorer"
    echo 삭제: schtasks /delete /tn "BlackEvent_DailyScorer" /f
) else (
    echo.
    echo [ERROR] 등록 실패. 관리자 권한으로 실행해주세요.
)

pause
