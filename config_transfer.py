"""
Exporting models from config.json, and merging an exported file into it.

Exports have the same shape as config.json ({"loras": [{"name": ..., "url": ..., "sha256": ...}], ...}), with
only the sections asked for. Imports merge into the existing configuration: new names are added, nothing is
removed, and an entry whose name exists with a different URL is a conflict, resolved for all conflicts at once
by keeping the existing entry or using the imported one. Imported files may come from anyone, so every entry is
checked, and entries that fail are skipped and reported.
"""
import re

from .model_paths import check_model_name

EXPORTED_FIELDS = ("name", "url", "sha256")

_SHA256 = re.compile(r"[0-9a-fA-F]{64}")


def export_sections(config, keys):
    """
    Returns the given sections of config (all model sections it has if keys is None), with only the exported fields.
    """
    keys = [key for key in config if isinstance(config[key], list)] if keys is None else keys
    return {key: [{field: entry[field] for field in EXPORTED_FIELDS if field in entry}
                  for entry in config.get(key, []) if isinstance(entry, dict)]
            for key in keys}


def plan_import(config, imported, known_keys, local_prefix):
    """
    Works out how imported would merge into config, without changing either.

    Returns a dict:
        sections: {key: {"added": [...], "unchanged": [...], "hash_added": [...],
                         "conflicts": [{"name", "url", "imported_url"}], "skipped": [{"name", "reason"}]}}
            ("added", "unchanged" and "hash_added" list model names; "offline" entries among the added are in "offline")
        skipped_sections: [{"key", "reason"}]
        entries: {key: [validated imported entries]}, used by apply_import
    """
    plan = {"sections": {}, "skipped_sections": [], "entries": {}}
    if not isinstance(imported, dict):
        plan["skipped_sections"].append({"key": "", "reason": "the file is not an exported configuration (a JSON object)"})
        return plan

    for key, entries in imported.items():
        if key not in known_keys:
            plan["skipped_sections"].append({"key": key, "reason": "not a model list of these loaders"})
            continue
        if not isinstance(entries, list):
            plan["skipped_sections"].append({"key": key, "reason": "not a list of models"})
            continue

        existing = {entry.get("name"): entry for entry in config.get(key, []) if isinstance(entry, dict)}
        section = {"added": [], "offline": [], "unchanged": [], "hash_added": [], "conflicts": [], "skipped": []}
        valid = []
        seen = set()
        for entry in entries:
            entry, reason = _validate_entry(entry, local_prefix)
            if reason:
                section["skipped"].append({"name": entry.get("name") if isinstance(entry, dict) else None, "reason": reason})
                continue
            name = entry["name"]
            if name in seen:
                section["skipped"].append({"name": name, "reason": "appears more than once in the file"})
                continue
            seen.add(name)
            valid.append(entry)

            mine = existing.get(name)
            if mine is None:
                section["added"].append(name)
                if entry["url"] == "offline":
                    section["offline"].append(name)
            elif mine.get("url") == entry["url"]:
                if entry.get("sha256") and not mine.get("sha256"):
                    section["hash_added"].append(name)
                else:
                    section["unchanged"].append(name)
            else:
                section["conflicts"].append({"name": name, "url": mine.get("url"), "imported_url": entry["url"]})

        plan["sections"][key] = section
        plan["entries"][key] = valid
    return plan


def apply_import(config, plan, use_imported):
    """
    Returns a copy of config with plan applied. Conflicting entries take the imported URL (and hash) if use_imported,
    else they are left as they are.
    """
    merged = {key: ([dict(entry) if isinstance(entry, dict) else entry for entry in value] if isinstance(value, list) else value)
              for key, value in config.items()}
    for key, entries in plan["entries"].items():
        section = plan["sections"][key]
        conflicts = {conflict["name"] for conflict in section["conflicts"]}
        models = merged.setdefault(key, [])
        by_name = {entry.get("name"): entry for entry in models if isinstance(entry, dict)}
        for entry in entries:
            mine = by_name.get(entry["name"])
            if mine is None:
                models.append(dict(entry))
            elif entry["name"] in section["hash_added"]:
                mine["sha256"] = entry["sha256"]
            elif entry["name"] in conflicts and use_imported:
                mine["url"] = entry["url"]
                if entry.get("sha256"):
                    mine["sha256"] = entry["sha256"]
                else:
                    mine.pop("sha256", None)
    return merged


def count_changes(plan, use_imported):
    """
    The number of entries apply_import would add or change.
    """
    return sum(len(section["added"]) + len(section["hash_added"]) + (len(section["conflicts"]) if use_imported else 0)
               for section in plan["sections"].values())


def _validate_entry(entry, local_prefix):
    """
    Returns (the entry with only the exported fields, None), or (entry, reason it is skipped).
    """
    if not isinstance(entry, dict):
        return entry, "not a model entry"
    name, url = entry.get("name"), entry.get("url")
    if not isinstance(name, str) or not name:
        return entry, "has no name"
    if name.startswith(local_prefix):
        return entry, f"names cannot start with '{local_prefix.strip()}'"
    problem = check_model_name(name)
    if problem:
        return entry, f"the name cannot be used: {problem}"
    if not isinstance(url, str) or not (url.startswith(("https://", "http://")) or url == "offline"):
        return entry, "the URL must start with http:// or https://, or be 'offline'"

    validated = {"name": name, "url": url}
    sha256 = entry.get("sha256")
    if isinstance(sha256, str) and _SHA256.fullmatch(sha256):
        validated["sha256"] = sha256.lower()
    return validated, None
