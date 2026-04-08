#!/usr/bin/env python3
"""
Hermes Agent file reader — reads memory, skills, and config from disk.

Pure stdlib implementation (pathlib, os, re). No external dependencies.
Supports HERMES_HOME env var for profile isolation.
"""

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


def _hermes_home() -> Path:
    """Return the Hermes home directory, respecting HERMES_HOME env var."""
    return Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))


def _count_entries(text: str) -> int:
    """Count memory entries (lines starting with '- ')."""
    return sum(1 for line in text.splitlines() if line.lstrip().startswith("- "))


def _read_file(path: Path) -> Optional[str]:
    """Read a file, returning None if it doesn't exist or can't be read."""
    try:
        return path.read_text(encoding="utf-8")
    except (FileNotFoundError, PermissionError, OSError):
        return None


def _parse_yaml_frontmatter(text: str) -> Dict[str, Any]:
    """
    Extract YAML frontmatter from a SKILL.md file.
    Minimal parser — no pyyaml dependency. Handles the fields we care about:
    name, description, version, platforms, and nested metadata.
    """
    match = re.match(r"^---\s*\n(.*?)\n---", text, re.DOTALL)
    if not match:
        return {}

    frontmatter = match.group(1)
    result: Dict[str, Any] = {}

    for line in frontmatter.splitlines():
        # Skip blank lines, comments, and deeply nested lines (2+ indent levels)
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        # Top-level key: value
        kv = re.match(r"^(\w[\w-]*)\s*:\s*(.+)$", line)
        if kv:
            key, value = kv.group(1), kv.group(2).strip()

            # Inline list: [item1, item2]
            list_match = re.match(r"^\[(.+)\]$", value)
            if list_match:
                items = [item.strip().strip("'\"") for item in list_match.group(1).split(",")]
                result[key] = items
            else:
                # Strip quotes
                result[key] = value.strip("'\"")

    return result


def _find_memory_file(hermes: Path, filename: str) -> Optional[Path]:
    """
    Find a memory file, checking both new and legacy locations.
    New: ~/.hermes/memories/<filename>
    Legacy: ~/.hermes/<filename>
    """
    candidates = [
        hermes / "memories" / filename,
        hermes / filename,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def read_memory() -> Dict[str, Any]:
    """
    Read Hermes memory files (MEMORY.md and USER.md).

    Returns dict with:
        memory: {content, path, entries} or None
        user_profile: {content, path, entries} or None
    """
    hermes = _hermes_home()
    result: Dict[str, Any] = {"memory": None, "user_profile": None}

    # MEMORY.md
    mem_path = _find_memory_file(hermes, "MEMORY.md")
    if mem_path:
        content = _read_file(mem_path)
        if content is not None:
            result["memory"] = {
                "content": content,
                "path": str(mem_path),
                "entries": _count_entries(content),
            }

    # USER.md
    user_path = _find_memory_file(hermes, "USER.md")
    if user_path:
        content = _read_file(user_path)
        if content is not None:
            result["user_profile"] = {
                "content": content,
                "path": str(user_path),
                "entries": _count_entries(content),
            }

    return result


def read_skills() -> List[Dict[str, Any]]:
    """
    Read all skills from ~/.hermes/skills/<category>/<name>/SKILL.md.

    Returns list of skill dicts with:
        name, category, description, version, platforms, path, content
    """
    hermes = _hermes_home()
    skills_dir = hermes / "skills"
    skills: List[Dict[str, Any]] = []

    if not skills_dir.is_dir():
        return skills

    for category_dir in sorted(skills_dir.iterdir()):
        if not category_dir.is_dir():
            continue
        category = category_dir.name

        for skill_dir in sorted(category_dir.iterdir()):
            if not skill_dir.is_dir():
                continue

            skill_file = skill_dir / "SKILL.md"
            content = _read_file(skill_file)
            if content is None:
                continue

            fm = _parse_yaml_frontmatter(content)
            skills.append({
                "name": fm.get("name", skill_dir.name),
                "category": category,
                "description": fm.get("description", ""),
                "version": fm.get("version", ""),
                "platforms": fm.get("platforms", []),
                "path": str(skill_file),
                "content": content,
            })

    return skills


def read_skill(name: str) -> Optional[Dict[str, Any]]:
    """
    Find a single skill by name (case-insensitive).

    Matches against both the frontmatter 'name' field and the directory name.
    Returns the skill dict or None if not found.
    """
    name_lower = name.lower()
    for skill in read_skills():
        if skill["name"].lower() == name_lower:
            return skill
    # Fallback: match against directory name portion of path
    for skill in read_skills():
        dir_name = Path(skill["path"]).parent.name.lower()
        if dir_name == name_lower:
            return skill
    return None


def read_status() -> Dict[str, Any]:
    """
    Read Hermes installation status.

    Returns dict with:
        hermes_home, config_exists, memory_exists, skills_count, provider, model
    """
    hermes = _hermes_home()
    config_path = hermes / "config.yaml"
    config_content = _read_file(config_path)

    # Parse provider and model from config.yaml without pyyaml.
    # We need the values nested under "model:", so we track when we're
    # inside that section (indented lines after "model:").
    provider = ""
    model = ""
    if config_content:
        in_model_section = False
        for line in config_content.splitlines():
            stripped = line.strip()
            # Detect top-level "model:" section
            if re.match(r"^model\s*:", line):
                in_model_section = True
                continue
            # If line is not indented, we've left the model section
            if in_model_section and line and not line[0].isspace():
                in_model_section = False
            if in_model_section:
                if re.match(r"provider\s*:", stripped):
                    provider = stripped.split(":", 1)[1].strip().strip("'\"")
                elif re.match(r"(default|model)\s*:", stripped):
                    model = stripped.split(":", 1)[1].strip().strip("'\"")

    # Count skills
    skills_dir = hermes / "skills"
    skills_count = 0
    if skills_dir.is_dir():
        for category_dir in skills_dir.iterdir():
            if not category_dir.is_dir():
                continue
            for skill_dir in category_dir.iterdir():
                if skill_dir.is_dir() and (skill_dir / "SKILL.md").is_file():
                    skills_count += 1

    # Check memory existence
    memory_exists = (
        _find_memory_file(hermes, "MEMORY.md") is not None
        or _find_memory_file(hermes, "USER.md") is not None
    )

    return {
        "hermes_home": str(hermes),
        "config_exists": config_path.is_file(),
        "memory_exists": memory_exists,
        "skills_count": skills_count,
        "provider": provider,
        "model": model,
    }
