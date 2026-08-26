import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchaudio
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

from fsac.torgo.config import (
    CONTROL_SPEAKERS,
    DYSARTHRIC_SPEAKERS,
    NBEST_BEAM_SIZE,
    NBEST_N,
    RESULT_DIR,
    SAMPLE_RATE,
    SPEAKER_TO_SEVERITY,
    WHISPER_MODEL_NAME,
)
from fsac.torgo.utils import get_all_speakers, get_speaker_wavs, make_filename


def load_wav(path: Path, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    wav, sr = torchaudio.load(str(path))
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav.squeeze(0).numpy()


def _compute_log_prob(model, processor, input_features, sequences):

    decoder_input_ids = sequences[:, :-1]
    target_ids = sequences[:, 1:]

    with torch.no_grad():
        outputs = model(
            input_features=input_features.expand(sequences.shape[0], -1, -1),
            decoder_input_ids=decoder_input_ids,
        )

    log_probs = torch.log_softmax(outputs.logits, dim=-1)

    token_log_probs = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)

    special_ids = set(processor.tokenizer.all_special_ids)
    results = []
    for i in range(sequences.shape[0]):
        mask = torch.tensor(
            [t.item() not in special_ids for t in target_ids[i]],
            device=sequences.device,
        )
        seq_log_prob = float((token_log_probs[i] * mask).sum())
        n_tok = int(mask.sum())
        results.append((seq_log_prob, n_tok))

    return results


def decode_nbest(
    model, processor, audio, device,
    n=10, beam_size=10,
    temperatures=(0.2, 0.4, 0.6, 0.8, 1.0),
    samples_per_temp=5,
):

    inputs = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    input_features = inputs.input_features.to(device=device, dtype=model.dtype)

    is_multilingual = getattr(model.config, "is_multilingual", False)
    lang_kwargs = {}
    if is_multilingual:
        lang_kwargs = {"language": "en", "task": "transcribe"}

    candidates = {}

    with torch.no_grad():
        beam_output = model.generate(
            input_features=input_features,
            num_beams=beam_size,
            num_return_sequences=min(n, beam_size),
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=256,
            **lang_kwargs,
        )

    for i in range(beam_output.sequences.shape[0]):
        text = processor.decode(beam_output.sequences[i], skip_special_tokens=True).strip()
        if text and text not in candidates:
            score = float(beam_output.sequences_scores[i])
            n_tok = sum(1 for t in beam_output.sequences[i].tolist()
                        if t not in processor.tokenizer.all_special_ids)
            candidates[text] = {"log_prob": score, "n_tokens": n_tok,
                                "seq": beam_output.sequences[i]}

    if len(candidates) < n:
        for temp in temperatures:
            if len(candidates) >= n * 2:
                break

            with torch.no_grad():
                sample_output = model.generate(
                    input_features=input_features,
                    do_sample=True,
                    temperature=temp,
                    top_p=0.95,
                    top_k=50,
                    num_return_sequences=samples_per_temp,
                    num_beams=1,
                    return_dict_in_generate=True,
                    output_scores=True,
                    max_new_tokens=256,
                    **lang_kwargs,
                )

            for i in range(sample_output.sequences.shape[0]):
                text = processor.decode(
                    sample_output.sequences[i], skip_special_tokens=True
                ).strip()
                if text and text not in candidates:
                    candidates[text] = {"seq": sample_output.sequences[i],
                                        "log_prob": None, "n_tokens": 0}

    needs_scoring = [t for t, v in candidates.items() if v["log_prob"] is None]
    if needs_scoring:
        raw_seqs = [candidates[t]["seq"] for t in needs_scoring]
        max_len = max(s.shape[0] for s in raw_seqs)
        pad_id = processor.tokenizer.pad_token_id if processor.tokenizer.pad_token_id is not None else processor.tokenizer.eos_token_id
        padded = torch.full((len(raw_seqs), max_len), pad_id,
                            device=device, dtype=torch.long)
        for i, seq in enumerate(raw_seqs):
            padded[i, :seq.shape[0]] = seq.to(device)

        batch_size = 8
        scored = []
        for start in range(0, len(needs_scoring), batch_size):
            end = min(start + batch_size, len(needs_scoring))
            batch_seqs = padded[start:end]
            batch_scores = _compute_log_prob(model, processor, input_features, batch_seqs)
            scored.extend(batch_scores)

        for text, (lp, nt) in zip(needs_scoring, scored):
            candidates[text]["log_prob"] = lp
            candidates[text]["n_tokens"] = nt

    sorted_cands = sorted(candidates.items(),
                          key=lambda x: x[1]["log_prob"], reverse=True)

    result = sorted_cands[:n]
    while len(result) < n:
        top = result[0]
        result.append(top)

    hypotheses = []
    for rank, (text, info) in enumerate(result):
        lp = info["log_prob"]
        nt = info["n_tokens"]
        hypotheses.append({
            "rank": rank + 1,
            "hypothesis": text,
            "log_prob": lp,
            "n_tokens": nt,
            "avg_log_prob": lp / max(nt, 1),
        })

    return hypotheses


def decode_with_timestamps(model, processor, audio, device):

    inputs = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    input_features = inputs.input_features.to(device=device, dtype=model.dtype)

    is_multilingual = getattr(model.config, "is_multilingual", False)

    ts_kwargs = {
        "input_features": input_features,
        "return_timestamps": True,
        "max_new_tokens": 256,
    }
    if is_multilingual:
        ts_kwargs["language"] = "en"
        ts_kwargs["task"] = "transcribe"

    with torch.no_grad():
        ts_output = model.generate(**ts_kwargs)

    if isinstance(ts_output, dict):
        sequences_ts = ts_output.get("sequences", ts_output.get("output_ids"))
        if sequences_ts is None:
            sequences_ts = list(ts_output.values())[0]
            if not isinstance(sequences_ts, torch.Tensor):
                sequences_ts = torch.tensor([[]])
    else:
        sequences_ts = ts_output

    text = processor.decode(sequences_ts[0], skip_special_tokens=True).strip()

    word_timestamps = []
    try:
        decoded_with_ts = processor.decode(
            sequences_ts[0], skip_special_tokens=False,
            output_offsets=True,
        )
        if isinstance(decoded_with_ts, dict) and "offsets" in decoded_with_ts:
            for offset in decoded_with_ts["offsets"]:
                ts_pair = offset.get("timestamp", (0.0, 0.0))
                if ts_pair is None:
                    ts_pair = (0.0, 0.0)
                word_timestamps.append({
                    "word": offset.get("text", "").strip(),
                    "start": ts_pair[0] if ts_pair[0] is not None else 0.0,
                    "end": ts_pair[1] if ts_pair[1] is not None else 0.0,
                })
    except Exception:
        pass

    conf_kwargs = {
        "input_features": input_features,
        "return_dict_in_generate": True,
        "output_scores": True,
        "max_new_tokens": 256,
    }
    if is_multilingual:
        conf_kwargs["language"] = "en"
        conf_kwargs["task"] = "transcribe"

    with torch.no_grad():
        conf_output = model.generate(**conf_kwargs)

    sequences_conf = conf_output.sequences
    scores_tuple = conf_output.scores

    token_confs = []
    if scores_tuple is not None:
        for step_idx, logits in enumerate(scores_tuple):
            probs = torch.softmax(logits[0], dim=-1)
            tok_idx = step_idx + 1
            if tok_idx < sequences_conf.shape[1]:
                token_id = sequences_conf[0, tok_idx].item()
                conf = float(probs[token_id])
                token_confs.append(conf)

    avg_conf = float(np.mean(token_confs)) if token_confs else 0.0

    return {
        "text": text,
        "avg_confidence": avg_conf,
        "token_confidences": token_confs,
        "word_timestamps": word_timestamps,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate n-best for TORGO")
    parser.add_argument("--model", default=WHISPER_MODEL_NAME)
    parser.add_argument("--n", type=int, default=NBEST_N)
    parser.add_argument("--beam", type=int, default=NBEST_BEAM_SIZE)
    parser.add_argument("--samples-per-temp", type=int, default=5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    use_fp16 = args.device == "cuda" and "large" in args.model
    dtype = torch.float16 if use_fp16 else torch.float32
    print(f"Loading {args.model} on {args.device} (dtype={dtype}) ...")
    processor = WhisperProcessor.from_pretrained(args.model)
    model = WhisperForConditionalGeneration.from_pretrained(args.model, torch_dtype=dtype)
    device = torch.device(args.device)
    model = model.to(device).eval()

    nbest_rows = []
    conf_rows = []

    for spk in get_all_speakers():
        print(f"Processing {spk} ...")
        entries = get_speaker_wavs(spk, sentences_only=True)
        if not entries:
            continue

        is_ctrl = spk in CONTROL_SPEAKERS
        severity = None if is_ctrl else SPEAKER_TO_SEVERITY.get(spk)

        for entry in entries:
            wav_path = entry["path"]
            ref_text = entry["prompt"]
            canonical = make_filename(spk, entry["session"], entry["utt_id"])

            audio = load_wav(wav_path)

            hypotheses = decode_nbest(
                model, processor, audio, device,
                n=args.n, beam_size=args.beam,
                samples_per_temp=args.samples_per_temp,
            )
            for hyp in hypotheses:
                nbest_rows.append({
                    "speaker": spk,
                    "filename": canonical,
                    "session": entry["session"],
                    "is_control": is_ctrl,
                    "severity": severity,
                    "ref_text": ref_text,
                    "rank": hyp["rank"],
                    "hypothesis": hyp["hypothesis"],
                    "log_prob": hyp["log_prob"],
                    "avg_log_prob": hyp["avg_log_prob"],
                    "n_tokens": hyp["n_tokens"],
                })

            ts_result = decode_with_timestamps(
                model, processor, audio, device,
            )
            conf_rows.append({
                "speaker": spk,
                "filename": canonical,
                "avg_confidence": ts_result["avg_confidence"],
                "n_conf_tokens": len(ts_result["token_confidences"]),
                "word_timestamps": json.dumps(ts_result["word_timestamps"]),
            })

        n_done = sum(1 for r in nbest_rows if r["speaker"] == spk)
        print(f"  {spk}: {n_done} hypothesis entries")

    if not nbest_rows:
        raise SystemExit(
            "No TORGO utterances found. Set FSAC_DATA_DIR to the TORGO root "
            "containing speaker directories such as F01/Session*/wav_headMic."
        )

    df_nbest = pd.DataFrame(nbest_rows)
    nbest_csv = RESULT_DIR / "nbest_hypotheses.csv"
    df_nbest.to_csv(nbest_csv, index=False)
    print(f"\nN-best saved to {nbest_csv}")
    print(f"  Total entries: {len(df_nbest)}")
    print(f"  Unique utterances: {df_nbest['filename'].nunique()}")

    n_unique_per_utt = df_nbest.groupby("filename")["hypothesis"].nunique()
    print(f"  Avg unique hyps per utterance: {n_unique_per_utt.mean():.2f}")
    print(f"  Utterances with >1 unique hyp: "
          f"{(n_unique_per_utt > 1).sum()}/{len(n_unique_per_utt)}")

    df_conf = pd.DataFrame(conf_rows)
    conf_csv = RESULT_DIR / "token_confidences.csv"
    df_conf.to_csv(conf_csv, index=False)
    print(f"Confidence + timestamps saved to {conf_csv}")


if __name__ == "__main__":
    main()
