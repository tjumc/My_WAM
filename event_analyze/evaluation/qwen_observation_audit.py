#!/usr/bin/env python3
"""Prepare blind human reviews of the exact images shown to Qwen, then score them."""

import argparse
import csv
import hashlib
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path


FIELDS = (
    "robot_location", "door", "dish_rack", "cutlery_basket",
    "right_hand", "left_hand", "knife_location", "fork_location", "plate_location",
)
NONCONCRETE = {"uncertain", "not_visible"}
VALID = {
    "robot_location": {"away", "approaching_station", "at_station", *NONCONCRETE},
    "door": {"closed", "opening", "open", "closing", *NONCONCRETE},
    "dish_rack": {"in", "moving_out", "out", "moving_in", *NONCONCRETE},
    "cutlery_basket": {"in", "moving_out", "out", "moving_in", *NONCONCRETE},
    "right_hand": {"free", "approaching", "contact_door", "contact_dish_rack", "contact_cutlery_basket", "holding_knife", "holding_fork", "holding_plate", *NONCONCRETE},
    "left_hand": {"free", "approaching", "contact_door", "contact_dish_rack", "contact_cutlery_basket", "holding_knife", "holding_fork", "holding_plate", *NONCONCRETE},
    "knife_location": {"tabletop", "right_hand", "left_hand", "cutlery_basket", "other", *NONCONCRETE},
    "fork_location": {"tabletop", "right_hand", "left_hand", "cutlery_basket", "other", *NONCONCRETE},
    "plate_location": {"tabletop", "right_hand", "left_hand", "dish_rack", "other", *NONCONCRETE},
}


def read_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sample_timeline(rows, count, rng):
    """One random proposal per time bin, avoiding a sample concentrated at one end."""
    if count <= 0 or len(rows) <= count:
        return rows
    return [rows[rng.randrange(i * len(rows) // count, (i + 1) * len(rows) // count)] for i in range(count)]


def locate_image(cache, source, event_id):
    if source == "base":
        matches = sorted((cache / "proposal/contact_sheets").glob(f"event_{event_id:02d}_f*.jpg"))
    else:
        matches = sorted((cache / "targeted_temporal_strips").glob(f"targeted_{event_id - 20000:02d}_*.jpg"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one Qwen input image for {source} event {event_id} in {cache}; found {len(matches)}")
    return matches[0]


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def prepare(args):
    out = Path(args.output_dir).resolve()
    if out.exists():
        raise SystemExit(f"Refusing to overwrite audit packet: {out}")
    out.mkdir(parents=True)
    images = out / "images"
    images.mkdir()
    human, predictions = [], []
    seen = set()

    for cache_arg in args.cache_dir:
        cache = Path(cache_arg).resolve()
        episode = cache.parents[2].name.removesuffix("_analysis") if cache.name == "cache" else cache.name
        if episode in seen:
            raise SystemExit(f"Duplicate episode: {episode}")
        seen.add(episode)
        episode_seed = int.from_bytes(hashlib.sha256(f"{args.seed}:{episode}".encode()).digest()[:8], "big")
        base = sample_timeline(read_jsonl(cache / "entity_observations_base.jsonl"), args.base_per_episode, random.Random(episode_seed))
        targeted = read_jsonl(cache / "targeted_observations.jsonl")
        if not base:
            raise SystemExit(f"No base Qwen observations in {cache}")
        for source, rows in (("base", base), ("targeted", targeted)):
            for obs in rows:
                event_id = int(obs["_event_id"])
                sample_id = f"{episode}__{source}_{event_id:05d}"
                image = images / f"{sample_id}.jpg"
                shutil.copyfile(locate_image(cache, source, event_id), image)
                entities = FIELDS if source == "base" else tuple(dict.fromkeys(
                    [*obs.get("_target_entities", []), "right_hand", "left_hand"]
                ))
                for entity in entities:
                    key = f"{entity}_location" if entity in {"knife", "fork", "plate"} else entity
                    if key not in obs.get("state_before", {}):
                        continue
                    common = {
                        "sample_id": sample_id,
                        "episode": episode,
                        "source": source,
                        "event_id": event_id,
                        "start_frame": obs["_window_start_frame"],
                        "end_frame": obs["_window_end_frame"],
                        "entity": key,
                        "image": str(image.relative_to(out)),
                    }
                    human.append({**common, "human_before": "", "human_after": "", "notes": ""})
                    predictions.append({
                        **common,
                        "prediction_before": obs["state_before"].get(key, "uncertain"),
                        "prediction_after": obs["state_after"].get(key, "uncertain"),
                        "confidence_before": obs.get("confidence_before", {}).get(key, 0),
                        "confidence_after": obs.get("confidence_after", {}).get(key, 0),
                    })

    common_fields = ["sample_id", "episode", "source", "event_id", "start_frame", "end_frame", "entity", "image"]
    write_csv(out / "human_labels.csv", human, common_fields + ["human_before", "human_after", "notes"])
    write_csv(out / "predictions.csv", predictions, common_fields + ["prediction_before", "prediction_after", "confidence_before", "confidence_after"])
    print(f"Prepared {len({r['sample_id'] for r in human})} images, {len(human)} entity reviews in {out}")


def score(args):
    packet = Path(args.packet_dir)
    with (packet / "human_labels.csv").open(newline="", encoding="utf-8") as file:
        human = list(csv.DictReader(file))
    with (packet / "predictions.csv").open(newline="", encoding="utf-8") as file:
        predictions = {(r["sample_id"], r["entity"]): r for r in csv.DictReader(file)}
    counts = defaultdict(lambda: defaultdict(int))
    for row in human:
        pred = predictions[(row["sample_id"], row["entity"])]
        group = (row["source"], row["episode"], row["entity"])
        for side in ("before", "after"):
            label = row[f"human_{side}"].strip()
            value = pred[f"prediction_{side}"].strip()
            if not label:
                counts[group]["unreviewed"] += 1
                continue
            if label not in VALID[row["entity"]]:
                raise ValueError(f"Invalid human label {label!r} for {row['entity']} in {row['sample_id']}")
            if label == "uncertain":
                counts[group]["human_uncertain"] += 1
                if value not in NONCONCRETE:
                    counts[group]["model_claim_on_human_uncertain"] += 1
                continue
            counts[group]["evaluable"] += 1
            if value not in NONCONCRETE:
                counts[group]["concrete_claims"] += 1
            if label not in NONCONCRETE:
                counts[group]["concrete_evaluable"] += 1
                if value not in NONCONCRETE:
                    counts[group]["concrete_covered"] += 1
                    if value == label:
                        counts[group]["concrete_correct"] += 1
            elif value not in NONCONCRETE:
                counts[group]["model_claim_when_not_visible"] += 1
            if value != "uncertain":
                counts[group]["emitted"] += 1
                if value == label:
                    counts[group]["correct"] += 1
                elif float(pred[f"confidence_{side}"] or 0) >= 0.85:
                    counts[group]["high_confidence_errors"] += 1
        before, after = row["human_before"].strip(), row["human_after"].strip()
        if before and after and before not in NONCONCRETE and after not in NONCONCRETE and before != after:
            counts[group]["human_transitions"] += 1
            if (pred["prediction_before"], pred["prediction_after"]) == (before, after):
                counts[group]["correct_transitions"] += 1

    pooled = defaultdict(lambda: defaultdict(int))
    for (source, episode, entity), values in counts.items():
        for key, value in values.items():
            pooled[(source, entity)][key] += value

    def summary(values):
        n = values["evaluable"]
        concrete = values["concrete_evaluable"]
        claims = values["concrete_claims"]
        return {
            **values,
            "exact_accuracy": values["correct"] / n if n else None,
            "concrete_recall": values["concrete_correct"] / concrete if concrete else None,
            "concrete_precision": values["concrete_correct"] / claims if claims else None,
            "concrete_coverage": values["concrete_covered"] / concrete if concrete else None,
        }
    report = {"pooled": {}, "by_episode": {}}
    for (source, entity), values in sorted(pooled.items()):
        report["pooled"].setdefault(source, {})[entity] = summary(values)
    for (source, episode, entity), values in sorted(counts.items()):
        report["by_episode"].setdefault(episode, {}).setdefault(source, {})[entity] = summary(values)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--cache-dir", action="append", required=True, help="Explicit episode version cache; held-out is never scanned")
    prep.add_argument("--output-dir", required=True)
    prep.add_argument("--base-per-episode", type=int, default=10)
    prep.add_argument("--seed", type=int, default=20260928)
    prep.set_defaults(func=prepare)
    scoring = sub.add_parser("score")
    scoring.add_argument("packet_dir")
    scoring.set_defaults(func=score)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
