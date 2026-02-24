"""공통 유틸리티 — 설정 로드, 로깅, 경로 관리"""

import os
import logging
from pathlib import Path

import yaml
from dotenv import load_dotenv


# 프로젝트 루트
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config() -> dict:
    """config/settings.yaml 로드"""
    config_path = PROJECT_ROOT / "config" / "settings.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_env():
    """프로젝트 루트의 .env 파일 로드"""
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        logging.warning(".env 파일이 없습니다. .env.example을 참고하여 생성하세요.")


def get_path(config: dict, key: str) -> Path:
    """설정에서 경로를 가져와 절대 경로로 반환"""
    rel = config["paths"][key]
    return PROJECT_ROOT / rel


def setup_logging(name: str, level: int = logging.INFO) -> logging.Logger:
    """모듈별 로거 생성"""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger
