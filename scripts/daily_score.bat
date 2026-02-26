@echo off
cd /d C:\work\BlackEvent-Predictor
python -X utf8 -m src.daily_pipeline.run >> data\outputs\daily_pipeline.log 2>&1
