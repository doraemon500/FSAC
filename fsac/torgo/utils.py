import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from fsac.torgo.config import (
    CONTROL_SPEAKERS,
    DYSARTHRIC_SPEAKERS,
    MIC,
    MIN_SENTENCE_WORDS,
    TORGO_ROOT,
    TRAIN_RATIO,
)


def get_speaker_dir(speaker_id: str) -> Path:
    return TORGO_ROOT / speaker_id


def get_sessions(speaker_id: str) -> List[Path]:

    speaker_dir = get_speaker_dir(speaker_id)
    if not speaker_dir.exists():
        return []
    sessions = sorted(
        [d for d in speaker_dir.iterdir()
         if d.is_dir() and d.name.startswith("Session")],
        key=lambda d: d.name,
    )
    return sessions


def load_prompts(session_dir: Path) -> Dict[str, str]:

    prompts_dir = session_dir / "prompts"
    if not prompts_dir.exists():
        return {}
    result = {}
    for f in sorted(prompts_dir.glob("*.txt")):
        utt_id = f.stem
        text = f.read_text().strip()
        result[utt_id] = text
    return result


def is_valid_sentence(text: str) -> bool:

    if not text or text == "xxx":
        return False
    if text.startswith("input/") or text.startswith("input\\"):
        return False
    if text.startswith("[") and text.endswith("]"):
        return False
    words = text.split()
    return len(words) >= MIN_SENTENCE_WORDS


def get_speaker_wavs(
    speaker_id: str,
    sessions: Optional[List[Path]] = None,
    sentences_only: bool = True,
) -> List[Dict]:





    if sessions is None:
        sessions = get_sessions(speaker_id)

    entries = []
    for sess_dir in sessions:
        mic_dir = sess_dir / MIC
        if not mic_dir.exists():
            continue
        prompts = load_prompts(sess_dir)
        for wav_path in sorted(mic_dir.glob("*.wav")):
            utt_id = wav_path.stem
            text = prompts.get(utt_id, "")
            if sentences_only and not is_valid_sentence(text):
                continue
            entries.append({
                "path": wav_path,
                "session": sess_dir.name,
                "utt_id": utt_id,
                "prompt": text,
                "speaker": speaker_id,
            })
    return entries


def get_all_speakers() -> List[str]:
    return CONTROL_SPEAKERS + list(DYSARTHRIC_SPEAKERS.keys())


def make_filename(speaker_id: str, session: str, utt_id: str) -> str:

    return f"{speaker_id}_{session}_{utt_id}.wav"


def train_test_split_speaker(
    entries: List[Dict],
    train_ratio: float = TRAIN_RATIO,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict]]:

    import random
    rng = random.Random(seed)
    shuffled = list(entries)
    rng.shuffle(shuffled)
    n_train = int(len(shuffled) * train_ratio)
    return shuffled[:n_train], shuffled[n_train:]


def train_test_split_all(
    sentences_only: bool = True,
    train_ratio: float = TRAIN_RATIO,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict]]:

    all_train, all_test = [], []
    for spk in get_all_speakers():
        entries = get_speaker_wavs(spk, sentences_only=sentences_only)
        tr, te = train_test_split_speaker(entries, train_ratio, seed)
        all_train.extend(tr)
        all_test.extend(te)
    return all_train, all_test
