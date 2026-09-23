#!/usr/bin/env python3
"""Utilities for loading declarative task-family schemas.

The schema is human-configurable once per task family. Per-trajectory inference
must consume it automatically without manual decisions.
"""
import json
from pathlib import Path


def load_schema(path):
    p = Path(path)
    with open(p, encoding="utf-8") as f:
        s = json.load(f)
    if "entities" not in s:
        raise ValueError(f"schema missing entities: {p}")
    return s


def entity_items(schema, entity_type=None):
    for name, cfg in schema.get("entities", {}).items():
        if entity_type is None or cfg.get("type") == entity_type:
            yield name, cfg


def observation_key(name, cfg):
    return cfg.get("observation_key", name)


def build_tracker_specs(schema):
    specs = {}
    for name, cfg in entity_items(schema):
        key = observation_key(name, cfg)
        typ = cfg.get("type")
        states = list(cfg.get("states", []))
        if not states:
            continue
        spec = {
            "states": states,
            "schema_entity": name,
            "type": typ,
        }
        if "adjacency" in cfg:
            spec["adj"] = {k: list(v) for k, v in cfg["adjacency"].items()}
        elif typ == "portable_object":
            src = list(cfg.get("source_states", ["tabletop"]))
            hands = list(cfg.get("hand_states", ["right_hand", "left_hand"]))
            target = cfg.get("target")
            other = [x for x in states if x not in set(src + hands + [target])]
            adj = {}
            for s in src:
                adj[s] = list(hands)
            for h in hands:
                adj[h] = list(dict.fromkeys(src + ([target] if target else []) + other))
            if target:
                adj[target] = list(hands)
            for x in other:
                adj[x] = list(hands)
            spec["adj"] = adj
            spec["target"] = target
        else:
            raise ValueError(f"schema entity {name} needs adjacency")

        stable = cfg.get("stable_states")
        if stable:
            spec["stable"] = set(stable)
        if cfg.get("target"):
            spec["target"] = cfg["target"]
        spec["motion_source"] = dict(cfg.get("motion_source", {}))
        spec["motion_endpoint"] = dict(cfg.get("motion_endpoint", {}))
        specs[key] = spec
    return specs


def container_rules(schema):
    out = {}
    for name, cfg in entity_items(schema):
        transitions = cfg.get("transitions", [])
        if not transitions:
            continue
        rules = {}
        for t in transitions:
            rules[(t["from"], t["to"])] = t["skill_type"]
        out[name] = {
            "observation_key": observation_key(name, cfg),
            "type": cfg.get("type"),
            "rules": rules,
            "motion_source": dict(cfg.get("motion_source", {})),
            "motion_endpoint": dict(cfg.get("motion_endpoint", {})),
            "contact_tokens": list(cfg.get("contact_tokens", [])),
            "requires": list(cfg.get("requires", [])),
            "robot_interaction": dict(cfg.get("robot_interaction", {})),
            "expected_initial_state": cfg.get("expected_initial_state"),
            "expected_final_state": cfg.get("expected_final_state"),
            "usage_state": cfg.get("usage_state"),
        }
    return out


def portable_rules(schema):
    out = {}
    for name, cfg in entity_items(schema, "portable_object"):
        out[name] = {
            "observation_key": observation_key(name, cfg),
            "source_states": list(cfg.get("source_states", ["tabletop"])),
            "hand_states": list(cfg.get("hand_states", ["right_hand", "left_hand"])),
            "target": cfg["target"],
            "parent_class": cfg.get("parent_class"),
            "skill_type": cfg["skill_type"],
            "episode_completion": cfg.get("episode_completion", "target_without_regrasp_before_object_context_switch"),
            "holding_tokens": list(cfg.get("holding_tokens", [f"holding_{name}"])),
        }
    return out


def skill_labels(schema):
    return dict(schema.get("skill_labels", {}))


def receptacle_placements(schema):
    """Map receptacle entity -> object placement skill types."""
    out = {}
    portable = portable_rules(schema)
    for obj, cfg in portable.items():
        out.setdefault(cfg["target"], set()).add(cfg["skill_type"])
    return out


def accessibility_requirements(schema):
    return {
        name: list(cfg.get("requires", []))
        for name, cfg in entity_items(schema)
        if cfg.get("requires")
    }


def stable_motion_rules(schema):
    """Rules for trajectory initialization from direct observations."""
    out = {}
    for name, cfg in entity_items(schema):
        stable = set(cfg.get("stable_states", []))
        motion_source = dict(cfg.get("motion_source", {}))
        if stable:
            out[name] = {
                "observation_key": observation_key(name, cfg),
                "stable": stable,
                "motion_source": motion_source,
            }
    return out


def semantic_parent(schema, object_name):
    cfg = schema.get("entities", {}).get(object_name, {})
    return cfg.get("parent_class")


def parent_members(schema, parent):
    return list((schema.get("semantic_classes", {}).get(parent) or {}).get("members", []))


def training_semantic(schema, object_name):
    """Return the preferred training label/entity for a portable object.

    Fine identity remains available for diagnostics. A task schema may declare
    a policy-equivalent parent class as the preferred supervision granularity.
    """
    entities = schema.get("entities", {})
    classes = schema.get("semantic_classes", {})
    cfg = entities.get(object_name, {})
    fine_skill = cfg.get("skill_type")
    parent = cfg.get("parent_class")
    if parent:
        pcfg = classes.get(parent, {})
        if pcfg.get("training_label_policy") == "parent":
            label = pcfg.get("preferred_training_label")
            if label:
                return {
                    "skill_type": label,
                    "entity": parent,
                    "backoff": True,
                    "fine_skill_type": fine_skill,
                    "fine_entity": object_name,
                    "allow_parent_held_evidence": bool(pcfg.get("allow_parent_held_evidence", False)),
                }
    return {
        "skill_type": fine_skill,
        "entity": object_name,
        "backoff": False,
        "fine_skill_type": fine_skill,
        "fine_entity": object_name,
        "allow_parent_held_evidence": False,
    }


def policy_skill_aliases(schema):
    """Map fine skill labels to task-preferred training labels."""
    out = {}
    for name, cfg in portable_rules(schema).items():
        sem = training_semantic(schema, name)
        if cfg.get("skill_type") and sem.get("skill_type"):
            out[cfg["skill_type"]] = sem["skill_type"]
    return out


def task_completion_policy(schema):
    """Return schema-level task completion/frontier policy."""
    return dict(schema.get("task_completion", {}))
