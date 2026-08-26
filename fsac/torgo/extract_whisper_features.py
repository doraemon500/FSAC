import argparse
from pathlib import Path

import numpy as np
import torch
import torchaudio
from transformers import WhisperFeatureExtractor, WhisperModel

from fsac.torgo.config import (
    CONTROL_SPEAKERS,
    DYSARTHRIC_SPEAKERS,
    SAMPLE_RATE,
    WHISPER_EMBEDDING_DIR,
    WHISPER_MODEL_NAME,
)
from fsac.torgo.utils import get_speaker_wavs, make_filename


def load_wav(path: Path, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    wav, sr = torchaudio.load(str(path))
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav.squeeze(0).numpy()


def compute_valid_frames(audio_len_samples: int) -> int:

    mel_frames = audio_len_samples // 160
    return max(mel_frames // 2, 1)


def extract_speaker(
    speaker_id: str,
    feature_extractor: WhisperFeatureExtractor,
    model: WhisperModel,
    device: torch.device,
    sentences_only: bool = True,
    dtype: torch.dtype = torch.float32,
) -> None:
    out_path = WHISPER_EMBEDDING_DIR / f"{speaker_id}.npz"
    if out_path.exists():
        print(f"  [skip] {out_path} already exists")
        return

    entries = get_speaker_wavs(speaker_id, sentences_only=sentences_only)
    if not entries:
        print(f"  [warn] no usable WAV files for {speaker_id}")
        return

    all_utt_embs = []
    all_frame_embs = []
    all_frame_lengths = []
    all_fnames = []
    all_prompts = []

    for entry in entries:
        wav_path = entry["path"]
        audio = load_wav(wav_path)
        n_valid = compute_valid_frames(len(audio))

        inputs = feature_extractor(
            audio, sampling_rate=SAMPLE_RATE, return_tensors="pt"
        )
        input_features = inputs.input_features.to(device=device, dtype=dtype)

        with torch.no_grad():
            enc_out = model.encoder(input_features)
            hidden = enc_out.last_hidden_state

        frames = hidden[0, :n_valid, :].float().cpu().numpy()
        utt_emb = frames.mean(axis=0)

        canonical_name = make_filename(
            entry["speaker"], entry["session"], entry["utt_id"]
        )
        all_frame_embs.append(frames)
        all_frame_lengths.append(n_valid)
        all_utt_embs.append(utt_emb)
        all_fnames.append(canonical_name)
        all_prompts.append(entry["prompt"])

    X_utt = np.stack(all_utt_embs, axis=0)
    np.savez(
        out_path,
        X_utt=X_utt,
        filenames=np.array(all_fnames),
        prompts=np.array(all_prompts),
        frame_lengths=np.array(all_frame_lengths),
        **{f"frames_{i}": f for i, f in enumerate(all_frame_embs)},
    )
    print(f"  [done] {speaker_id}: {X_utt.shape[0]} utterances, dim={X_utt.shape[1]}")


def main():
    parser = argparse.ArgumentParser(description="Extract Whisper features for TORGO")
    parser.add_argument("--model", default=WHISPER_MODEL_NAME)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--include-words", action="store_true",
                        help="Include single-word utterances too")
    args = parser.parse_args()

    WHISPER_EMBEDDING_DIR.mkdir(parents=True, exist_ok=True)

    use_fp16 = args.device == "cuda" and "large" in args.model
    dtype = torch.float16 if use_fp16 else torch.float32
    print(f"Loading Whisper encoder from {args.model} on {args.device} (dtype={dtype}) ...")
    feature_extractor = WhisperFeatureExtractor.from_pretrained(args.model)
    model = WhisperModel.from_pretrained(args.model, torch_dtype=dtype)
    device = torch.device(args.device)
    model = model.to(device).eval()
    print(f"  Encoder dim: {model.config.d_model}")

    all_speakers = CONTROL_SPEAKERS + list(DYSARTHRIC_SPEAKERS.keys())
    sentences_only = not args.include_words

    for spk in all_speakers:
        print(f"Processing {spk} ...")
        extract_speaker(spk, feature_extractor, model, device,
                        sentences_only=sentences_only, dtype=dtype)

    print("Whisper feature extraction complete.")


if __name__ == "__main__":
    main()
