import json
import re

import numpy as np
import pandas as pd

from fsac.torgo.config import FDS_T, RESULT_DIR, WHISPER_FDS_RESULT_DIR


def risk_level(score: float, thresholds=(55, 72, 85)) -> str:






    if score < thresholds[0]:
        return "low"
    elif score < thresholds[1]:
        return "moderate"
    elif score < thresholds[2]:
        return "high"
    return "very_high"


def estimate_time_from_frame(frame_idx: int, hop_samples: int = 320,
                              sr: int = 16000) -> float:
    return frame_idx * hop_samples / sr


def build_risk_prompt(
    utt_risk: float,
    word_timestamps_json: str = None,
    frame_risks: np.ndarray = None,
    top_k_segments: int = 5,
) -> str:

    level = risk_level(utt_risk)
    lines = [f"- utterance_risk: {utt_risk:.3f} ({level})"]


    if word_timestamps_json:
        try:
            word_ts = json.loads(word_timestamps_json)
            if word_ts and isinstance(word_ts, list):
                high_risk_words = []
                for w in word_ts:
                    word_text = w.get("word", "").strip()
                    if not word_text:
                        continue

                    high_risk_words.append(word_text)
                if high_risk_words:
                    lines.append(f"- n_words: {len(high_risk_words)}")
        except (json.JSONDecodeError, TypeError):
            pass

    if frame_risks is not None and len(frame_risks) > 0:
        high_threshold = np.percentile(frame_risks, 90)
        high_mask = frame_risks > high_threshold
        high_indices = np.where(high_mask)[0]

        if len(high_indices) > 0:
            segments = []
            start = high_indices[0]
            prev = start
            for idx in high_indices[1:]:
                if idx == prev + 1:
                    prev = idx
                else:
                    t_start = estimate_time_from_frame(start)
                    t_end = estimate_time_from_frame(prev + 1)
                    segments.append((t_start, t_end, frame_risks[start:prev + 1].mean()))
                    start = idx
                    prev = idx
            t_start = estimate_time_from_frame(start)
            t_end = estimate_time_from_frame(prev + 1)
            segments.append((t_start, t_end, frame_risks[start:prev + 1].mean()))

            segments.sort(key=lambda x: -x[2])
            top_segs = segments[:top_k_segments]
            seg_strs = [f"{s:.2f}-{e:.2f}s" for s, e, _ in top_segs]
            lines.append(f"- high_risk_segments: [{', '.join(seg_strs)}]")
            lines.append(f"- n_high_risk_frames: {len(high_indices)}/{len(frame_risks)}")

        lines.append(f"- high_risk_fraction: {high_mask.mean():.3f}")

    severity_est = ("normal" if level == "low" else
                    "mild" if level == "moderate" else "low_intelligibility")
    lines.append(f"- severity_estimate: {severity_est}")

    return "\n".join(lines)


def format_ger_prompt(risk_summary: str, hypotheses: list, max_hyps: int = 5) -> str:
    hyp_lines = []
    for h in hypotheses[:max_hyps]:
        hyp_lines.append(f"{h['rank']}. (score={h['log_prob']:.2f}) {h['hypothesis']}")

    prompt = (
        "[INST] You are correcting ASR output for dysarthric speech.\n\n"
        f"Pronunciation risk:\n{risk_summary}\n\n"
        f"N-best hypotheses:\n" + "\n".join(hyp_lines) + "\n\n"
        "Output the corrected transcript. [/INST]"
    )
    return prompt


def generate_prompts():
    frame_fds_csv = WHISPER_FDS_RESULT_DIR / "frame_fds_scores.csv"
    nbest_csv = RESULT_DIR / "nbest_hypotheses.csv"
    conf_csv = RESULT_DIR / "token_confidences.csv"

    if not frame_fds_csv.exists():
        raise FileNotFoundError(f"{frame_fds_csv} not found. Run frame_fds first.")
    if not nbest_csv.exists():
        raise FileNotFoundError(f"{nbest_csv} not found. Run generate_nbest first.")

    frame_df = pd.read_csv(frame_fds_csv)
    nbest_df = pd.read_csv(nbest_csv)

    mean_col = "frame_fds_mean" if "frame_fds_mean" in frame_df.columns else "frame_fds_mean" if "frame_fds_mean" in frame_df.columns else "frame_risk_mean"


    ts_map = {}
    if conf_csv.exists():
        conf_df = pd.read_csv(conf_csv)
        if "word_timestamps" in conf_df.columns:
            ts_map = dict(zip(conf_df["filename"], conf_df["word_timestamps"]))

    prompts = []
    for fname, grp in nbest_df.groupby("filename"):
        frame_row = frame_df[frame_df["filename"] == fname]
        utt_risk = float(frame_row.iloc[0][mean_col]) if not frame_row.empty else 0.0

        word_ts_json = ts_map.get(fname)
        risk_summary = build_risk_prompt(utt_risk, word_ts_json)

        hyps = grp.sort_values("rank").to_dict("records")
        prompt = format_ger_prompt(risk_summary, hyps)

        ref_text = grp.iloc[0].get("ref_text", "")
        prompts.append({
            "filename": fname,
            "speaker": grp.iloc[0]["speaker"],
            "session": grp.iloc[0].get("session", ""),
            "is_control": grp.iloc[0]["is_control"],
            "severity": grp.iloc[0].get("severity"),
            "ref_text": ref_text,
            "prompt": prompt,
            "target": str(ref_text),
            "utt_risk": utt_risk,
        })

    df_prompts = pd.DataFrame(prompts)
    out_csv = RESULT_DIR / "ger_prompts.csv"
    df_prompts.to_csv(out_csv, index=False)
    print(f"GER prompts saved to {out_csv}")
    print(f"  Total: {len(df_prompts)} utterances")
    print(f"\nSample prompt:\n{df_prompts.iloc[0]['prompt'][:500]}")

    return df_prompts


def main():
    generate_prompts()


if __name__ == "__main__":
    main()
