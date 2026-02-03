#!/usr/bin/env python3
"""
Sync version from each package's VERSION file into its pyproject.toml [tool.poetry] section.
VERSION is the source of truth; pyproject.toml is updated to match.
"""
from pathlib import Path
import re
import sys

SRC_DIR = Path(__file__).resolve().parent.parent / "src"


def get_version_from_file(version_path: Path) -> str:
    """First line of VERSION, stripped."""
    text = version_path.read_text().strip()
    return text.splitlines()[0].strip() if text else ""


def sync_pyproject(module_dir: Path) -> bool:
    """Update pyproject.toml version to match VERSION. Returns True if file was changed."""
    version_file = module_dir / "VERSION"
    pyproject = module_dir / "pyproject.toml"
    if not version_file.is_file() or not pyproject.is_file():
        return False

    version = get_version_from_file(version_file)
    if not version:
        return False

    content = pyproject.read_text()
    # Replace version line only within [tool.poetry] section
    in_poetry = False
    new_lines = []
    changed = False
    version_pattern = re.compile(r'^version\s*=\s*["\']?[^"\']*["\']?\s*$')

    for line in content.splitlines(keepends=True):
        if line.strip() == "[tool.poetry]":
            in_poetry = True
        elif in_poetry and line.strip().startswith("["):
            in_poetry = False

        if in_poetry and version_pattern.match(line.strip()):
            new_line = f'version = "{version}"\n'
            if line.rstrip() != new_line.rstrip():
                changed = True
                line = new_line
        new_lines.append(line)

    if changed:
        pyproject.write_text("".join(new_lines))
    return changed


def main() -> int:
    updated = []
    for path in sorted(SRC_DIR.iterdir()):
        if path.is_dir() and sync_pyproject(path):
            updated.append(path.name)

    if updated:
        for name in updated:
            print(f"Updated {name}: pyproject.toml version synced from VERSION")
        return 0
    print("All packages already in sync.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
