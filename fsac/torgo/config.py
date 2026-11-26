import json
import os
from pathlib import Path



PROJECT_ROOT = Path(__file__).resolve().parents[2]
TORGO_ROOT = Path(os.environ.get("FSAC_DATA_DIR", PROJECT_ROOT / "data" / "torgo" / "raw"))

FSAC_CACHE_DIR = Path(os.environ.get("FSAC_CACHE_DIR", PROJECT_ROOT / "cache"))
EMBEDDING_DIR = FSAC_CACHE_DIR / "torgo"

_BASE_RESULT_DIR = Path(
    os.environ.get("FSAC_RESULTS_DIR", PROJECT_ROOT / "results" / "torgo" / "runs" / "fds_vgate_main")
)
RESULT_DIR = Path(os.environ["TORGO_RESULT_DIR"]) if "TORGO_RESULT_DIR" in os.environ else _BASE_RESULT_DIR
WHISPER_FDS_RESULT_DIR = RESULT_DIR / "whisper_fds"


SAMPLE_RATE = 16_000
MIC = "wav_headMic"




DYSARTHRIC_SPEAKERS = {
    "F01": {"severity": "severe",            "diagnosis": "CP"},
    "F03": {"severity": "moderate",          "diagnosis": "CP"},
    "F04": {"severity": "mild",              "diagnosis": "CP/Athetoid"},
    "M01": {"severity": "severe",            "diagnosis": "CP"},
    "M02": {"severity": "severe",            "diagnosis": "CP"},
    "M03": {"severity": "mild",              "diagnosis": "CP"},
    "M04": {"severity": "severe",            "diagnosis": "ALS"},
    "M05": {"severity": "moderately-severe", "diagnosis": "CP"},
}

CONTROL_SPEAKERS = ["FC01", "FC02", "FC03", "MC01", "MC02"]



SEVERITY_BINS = {
    "Severe":            ["F01", "M01", "M02", "M04"],
    "Moderately-severe": ["M05"],
    "Moderate":          ["F03"],
    "Mild":              ["F04", "M03"],
}

SEVERITY_ORDER = ["Severe", "Moderately-severe", "Moderate", "Mild"]

SPEAKER_TO_SEVERITY = {}
for _sev, _spk_list in SEVERITY_BINS.items():
    for _spk in _spk_list:
        SPEAKER_TO_SEVERITY[_spk] = _sev


FDS_K = 5
FDS_T = 1000
FDS_SEED = 42



WHISPER_MODEL_NAME = os.environ.get("WHISPER_MODEL", "openai/whisper-base.en")

_WHISPER_DIM_MAP = {
    "tiny": 384, "base": 512, "small": 768, "medium": 1024,
    "large": 1280, "large-v2": 1280, "large-v3": 1280,
}
def _infer_whisper_dim(model_name: str) -> int:
    for key in sorted(_WHISPER_DIM_MAP, key=len, reverse=True):
        if key in model_name:
            return _WHISPER_DIM_MAP[key]
    return 512

WHISPER_ENCODER_DIM = _infer_whisper_dim(WHISPER_MODEL_NAME)

def _whisper_tag(model_name: str) -> str:
    for key in ["large-v3", "large-v2", "large", "medium", "small", "base", "tiny"]:
        if key in model_name:
            return key.replace("-", "").replace(".", "")
    return "base"

WHISPER_EMBEDDING_DIR = (
    FSAC_CACHE_DIR / "whisper_embeddings" / f"torgo_whisper_{_whisper_tag(WHISPER_MODEL_NAME)}"
)


NBEST_N = 10
NBEST_BEAM_SIZE = 10












TRAIN_RATIO = 0.8
SPLIT_SEED = 42
SPLIT_POLICY = os.environ.get("TORGO_SPLIT_POLICY", "speaker_disjoint_v1")







_DEFAULT_SPEAKER_DISJOINT_EVAL_SPEAKERS = ["F01", "M04", "F04", "FC01", "MC02"]
_eval_override = os.environ.get("TORGO_EVAL_SPEAKERS")
if _eval_override:
    SPEAKER_DISJOINT_EVAL_SPEAKERS = [s.strip() for s in _eval_override.split(",") if s.strip()]
else:
    SPEAKER_DISJOINT_EVAL_SPEAKERS = list(_DEFAULT_SPEAKER_DISJOINT_EVAL_SPEAKERS)
SPEAKER_DISJOINT_EVAL_SET = set(SPEAKER_DISJOINT_EVAL_SPEAKERS)
ALL_TORGO_SPEAKERS = CONTROL_SPEAKERS + list(DYSARTHRIC_SPEAKERS.keys())
SPEAKER_DISJOINT_TRAIN_SPEAKERS = [
    spk for spk in ALL_TORGO_SPEAKERS if spk not in SPEAKER_DISJOINT_EVAL_SET
]


MIN_SENTENCE_WORDS = 3


def _filename_to_speaker(filename):
    filename = Path(str(filename)).name
    return filename.split("_", 1)[0]


def get_random_utterance_split(filenames, train_ratio=TRAIN_RATIO, seed=SPLIT_SEED):
    import random
    fnames = sorted(set(str(f) for f in filenames))
    rng = random.Random(seed)
    rng.shuffle(fnames)
    n_train = int(len(fnames) * train_ratio)
    train_files = set(fnames[:n_train])
    test_files = set(fnames[n_train:])
    return train_files, test_files


def get_speaker_disjoint_split(filenames, eval_speakers=None):





    eval_set = set(eval_speakers or SPEAKER_DISJOINT_EVAL_SPEAKERS)
    train_files, test_files = set(), set()
    for fname in sorted(set(str(f) for f in filenames)):
        speaker = _filename_to_speaker(fname)
        if speaker in eval_set:
            test_files.add(fname)
        else:
            train_files.add(fname)
    return train_files, test_files


def get_canonical_split(filenames, train_ratio=TRAIN_RATIO, seed=SPLIT_SEED):









    if os.environ.get("TORGO_SPLIT_FILE"):
        train, dev, test = _load_split_file(filenames, os.environ["TORGO_SPLIT_FILE"])
        return train, dev | test
    policy = SPLIT_POLICY.lower()
    if policy in {"speaker_disjoint", "speaker_disjoint_v1", "speaker_disjoint_severity"}:
        return get_speaker_disjoint_split(filenames)
    if policy in {"utterance_random", "random_80_20"}:
        return get_random_utterance_split(filenames, train_ratio=train_ratio, seed=seed)
    raise ValueError(f"Unknown TORGO_SPLIT_POLICY: {SPLIT_POLICY}")


def _load_split_file(filenames, path):














    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"TORGO_SPLIT_FILE not found: {p}")
    payload = json.loads(p.read_text(encoding="utf-8"))
    available = {str(f) for f in filenames}
    out = []
    for key in ("train", "dev", "test"):
        wanted = {str(f) for f in payload.get(key, [])}
        out.append(wanted & available)
    train, dev, test = out
    overlap_td = train & dev
    overlap_tt = train & test
    overlap_dt = dev & test
    if overlap_td or overlap_tt or overlap_dt:
        raise ValueError(
            f"TORGO_SPLIT_FILE has overlapping splits: "
            f"train∩dev={len(overlap_td)} train∩test={len(overlap_tt)} "
            f"dev∩test={len(overlap_dt)} (file={p})"
        )
    return train, dev, test


def get_canonical_three_way_split(filenames, train_ratio=TRAIN_RATIO, seed=SPLIT_SEED):









    split_file = os.environ.get("TORGO_SPLIT_FILE")
    if not split_file:
        raise ValueError("TORGO_SPLIT_FILE is required for train/dev/test splitting.")
    return _load_split_file(filenames, split_file)


SPLIT_FILE = os.environ.get("TORGO_SPLIT_FILE")
SPLIT_TAG = os.environ.get("TORGO_SPLIT_TAG", "")
