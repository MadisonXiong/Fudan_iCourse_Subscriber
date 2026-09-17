import json
import os
import urllib.request

STUDENT_ID = os.environ.get("StuId", "")
PASSWORD = os.environ.get("UISPsw", "")

WEBVPN_BASE = "https://webvpn.fudan.edu.cn"
IDP_BASE = "https://id.fudan.edu.cn"
ICOURSE_BASE = "https://icourse.fudan.edu.cn"

WEBVPN_AES_KEY = b"wrdvpnisthebest!"
WEBVPN_AES_IV = b"wrdvpnisthebest!"

TENANT_CODE = "222"
GROUP_CODE = "2095000001"

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

# ModelScope 的模型仓库与 API-Inference 在线模型不是同一张表。下面只放
# ``GET /v1/models`` 当前公布的文本模型候选；运行时还会再次读取该接口，
# 自动剔除已撤下 Provider 的模型。
# Keep this list deliberately short.  Proofreading touches every three-minute
# window, so cycling through many large models can turn one provider outage into
# hours of retries.  Both IDs are also checked against GET /v1/models at startup.
MODELSCOPE_TEXT_MODELS = [
    "Qwen/Qwen3.8-Flash-Next",
    "Qwen/Qwen3.5-35B-A3B",
]

MODELSCOPE_VISION_MODELS = [
    "OpenGVLab/InternVL3_5-241B-A28B",
]

_MODELSCOPE_DISCOVERY_CACHE: dict[str, set[str] | None] = {}


def modelscope_available_models(base_url: str) -> set[str] | None:
    """Return the live API-Inference model IDs, or ``None`` on lookup failure.

    ModelScope exposes an OpenAI-compatible ``GET /v1/models`` endpoint.  A
    successful response is authoritative for Provider availability; a failed
    discovery must not make an otherwise healthy run unusable, so callers fall
    back to the configured candidates in that case.
    """
    normalized = str(base_url or "").rstrip("/")
    if normalized in _MODELSCOPE_DISCOVERY_CACHE:
        return _MODELSCOPE_DISCOVERY_CACHE[normalized]
    if os.environ.get("MODELSCOPE_MODEL_DISCOVERY", "1").strip().lower() in {
        "0", "false", "no", "off",
    }:
        _MODELSCOPE_DISCOVERY_CACHE[normalized] = None
        return None

    url = normalized + "/models"
    timeout = max(1, int(os.environ.get("MODELSCOPE_DISCOVERY_TIMEOUT", "15")))
    try:
        request = urllib.request.Request(
            url,
            headers={"Accept": "application/json", "User-Agent": "FiCS/1.0"},
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        ids = {
            str(item.get("id") or "").strip()
            for item in payload.get("data", [])
            if isinstance(item, dict) and str(item.get("id") or "").strip()
        }
        if not ids:
            raise ValueError("ModelScope /models returned no model IDs")
        _MODELSCOPE_DISCOVERY_CACHE[normalized] = ids
        print(
            f"[ModelScope] discovered {len(ids)} live API-Inference model(s).",
            flush=True,
        )
        return ids
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(
            f"[ModelScope] live model discovery unavailable: "
            f"{type(exc).__name__}: {exc}; using configured candidates.",
            flush=True,
        )
        _MODELSCOPE_DISCOVERY_CACHE[normalized] = None
        return None


def filter_modelscope_models(models: list[str], base_url: str) -> list[str]:
    """Drop models that currently have no ModelScope inference Provider."""
    candidates = list(dict.fromkeys(str(m).strip() for m in models if str(m).strip()))
    available = modelscope_available_models(base_url)
    if available is None:
        return candidates
    selected = [model for model in candidates if model in available]
    unavailable = [model for model in candidates if model not in available]
    if unavailable:
        print(
            "[ModelScope] skipping model(s) without a live Provider: "
            + ", ".join(unavailable),
            flush=True,
        )
    return selected


# 模型服务商配置（按列表顺序作为优先级，从前往后尝试）。
# 用户可以在这里随意添加/删除/重排服务商和模型。
# 兼容性：只设置 DASHSCOPE_API_KEY 也能跑（modelscope 项的 api_key 直接读取它）。
# 同名 provider 多次出现 → resolve_model_providers() 把它们的 models 合并到首次出
# 现的那条；这避免 Summarizer 内部按 name 索引 client 字典时被后写覆盖。
MODEL_PROVIDERS: list[dict] = [
    {
        "name": "modelscope",
        "api_key_env": "DASHSCOPE_API_KEY",
        "base_url_env": "DASHSCOPE_BASE_URL",
        "default_base_url": "https://api-inference.modelscope.cn/v1/",
        "models": MODELSCOPE_TEXT_MODELS,
    },
    {
        "name": "deepseek",
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url_env": "DEEPSEEK_BASE_URL",
        "default_base_url": "https://api.deepseek.com",
        "models": [
            "deepseek-v4-flash"
        ],
    },
    {
        "name": "gemini",
        "api_key_env": "GEMINI_API_KEY",
        "base_url_env": "GEMINI_BASE_URL",
        "default_base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "models": [
            "gemini-3.6-flash",
            "gemini-3.5-flash",
        ],
    }
]


def resolve_model_providers() -> list[dict]:
    """Resolve MODEL_PROVIDERS into runtime configs.

    Drops providers whose api_key env var is unset. Same-name entries get
    their model lists merged into the first occurrence (Summarizer's client
    dict keys on name and would otherwise collide).

    Returns:
        list of {name, api_key, base_url, models}.
    """
    resolved: list[dict] = []
    by_name: dict[str, dict] = {}
    for p in MODEL_PROVIDERS:
        api_key = os.environ.get(p["api_key_env"], "").strip()
        if not api_key:
            continue
        base_url = (
            os.environ.get(p.get("base_url_env", ""), "").strip()
            or p.get("default_base_url", "")
        )
        if not base_url:
            continue
        if p["name"] in by_name:
            existing = by_name[p["name"]]
            for m in p["models"]:
                if m not in existing["models"]:
                    existing["models"].append(m)
            continue
        entry = {
            "name": p["name"],
            "api_key": api_key,
            "base_url": base_url,
            "models": list(p["models"]),
        }
        if p["name"] == "modelscope":
            entry["models"] = filter_modelscope_models(
                entry["models"], entry["base_url"]
            )
            if not entry["models"]:
                print(
                    "[ModelScope] none of the configured text models currently "
                    "has an API-Inference Provider; disabling this provider.",
                    flush=True,
                )
                continue
        resolved.append(entry)
        by_name[p["name"]] = entry
    return resolved


# Legacy compatibility shims (kept so other modules importing these don't break)
DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# QQ SMTP
SMTP_EMAIL = os.environ.get("SMTP_EMAIL", "")
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD", "")
RECEIVER_EMAIL = os.environ.get("RECEIVER_EMAIL", "")
SMTP_HOST = "smtp.qq.com"
SMTP_PORT = 465

# Database & Storage
DATA_DIR = os.environ.get("DATA_DIR", "data")
VIDEO_DIR = os.path.join(DATA_DIR, "videos")
AUDIO_DIR = os.path.join(DATA_DIR, "audio")
DB_PATH = os.environ.get("DB_PATH", os.path.join(DATA_DIR, "icourse.db"))

# ASR
ASR_MODEL_DIR = os.environ.get(
    "ASR_MODEL_DIR",
    os.environ.get(
        "SENSEVOICE_MODEL_DIR",
        "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17",
    ),
)
SENSEVOICE_MODEL_DIR = ASR_MODEL_DIR
SILERO_VAD_PATH = os.environ.get("SILERO_VAD_PATH", "silero_vad.onnx")
ASR_BACKEND = os.environ.get("ASR_BACKEND", "sensevoice").strip().lower()
ASR_NUM_THREADS = int(os.environ.get("ASR_NUM_THREADS", "4"))

# Scheduler concurrency
IMAGE_WORKERS = int(os.environ.get("IMAGE_WORKERS", "20"))
OCR_MAX_WORKERS = int(os.environ.get("OCR_MAX_WORKERS", "8"))
OCR_MAX_TARGET = int(os.environ.get("OCR_MAX_TARGET", "2"))
VIDEO_DOWNLOAD_CONCURRENCY = int(
    os.environ.get("VIDEO_DOWNLOAD_CONCURRENCY", "2")
)

# Blackboard / handwritten-math extraction. Invoked only for whitelisted
# course titles (default: 泛函分析).
#
# Dense extraction remains every 15s so short-lived board states are not lost.
# A local temporal selector then reduces expensive multimodal requests:
# - one stable full-coverage anchor about every 60s;
# - extra persistent board-change events;
# - a soft default cap of 240 selected frames for very long lectures.
BLACKBOARD_SAMPLE_SEC = int(os.environ.get("BLACKBOARD_SAMPLE_SEC", "15"))
BLACKBOARD_COVERAGE_SEC = int(os.environ.get("BLACKBOARD_COVERAGE_SEC", "60"))
BLACKBOARD_ANALYSIS_WIDTH = int(os.environ.get("BLACKBOARD_ANALYSIS_WIDTH", "192"))
BLACKBOARD_CHANGE_THRESHOLD = int(os.environ.get("BLACKBOARD_CHANGE_THRESHOLD", "24"))
BLACKBOARD_STABLE_THRESHOLD = int(os.environ.get("BLACKBOARD_STABLE_THRESHOLD", "14"))
BLACKBOARD_CHANGE_RATIO = float(os.environ.get("BLACKBOARD_CHANGE_RATIO", "0.008"))
BLACKBOARD_EVENT_GAP_SEC = int(os.environ.get("BLACKBOARD_EVENT_GAP_SEC", "30"))
BLACKBOARD_VISION_BATCH_SIZE = int(os.environ.get("BLACKBOARD_VISION_BATCH_SIZE", "4"))
BLACKBOARD_MAX_FRAMES = int(os.environ.get("BLACKBOARD_MAX_FRAMES", "240"))
BLACKBOARD_FFMPEG_TIMEOUT = int(os.environ.get("BLACKBOARD_FFMPEG_TIMEOUT", "1800"))

# Monitored courses
COURSE_IDS = [
    c.strip()
    for c in os.environ.get("COURSE_IDS", "").split(",")
    if c.strip()
]

# Deprecated legacy semester-crawl setting.
CRAWL_TERM = os.environ.get("CRAWL_TERM", "").strip()
