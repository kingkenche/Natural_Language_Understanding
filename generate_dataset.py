"""
generate_dataset.py
-------------------
TASK-0  –  Generates 1000 diverse Indian names using the Claude API
           and saves them to TrainingNames.txt.

Requirements
------------
    pip install anthropic

Usage
-----
    python generate_dataset.py                  # produces TrainingNames.txt
    python generate_dataset.py --out MyNames.txt --target 1000
"""

import argparse
import os
import re
import sys

try:
    import anthropic
except ImportError:
    sys.exit("Please install the Anthropic SDK:  pip install anthropic")


# ──────────────────────────────────────────────────────────────────────
# Prompts – one batch per regional / demographic group
# ──────────────────────────────────────────────────────────────────────

PROMPTS = [
    "List 120 common Indian male first names from the Hindi-speaking belt "
    "(UP, MP, Bihar, Rajasthan). Output only names, one per line, no numbering.",

    "List 120 common Indian female first names from North India. "
    "Output only names, one per line, no numbering.",

    "List 120 South Indian first names (Tamil, Telugu, Kannada, Malayalam) "
    "for both males and females. Output only names, one per line, no numbering.",

    "List 100 Bengali first names (both male and female) from West Bengal "
    "and Bangladesh. Output only names, one per line, no numbering.",

    "List 100 Punjabi first names (both male and female). "
    "Output only names, one per line, no numbering.",

    "List 100 Gujarati and Marathi first names (both male and female). "
    "Output only names, one per line, no numbering.",

    "List 100 Indian Muslim first names commonly used in India "
    "(both male and female). Output only names, one per line, no numbering.",

    "List 80 traditional Sanskrit-origin Indian first names not yet covered, "
    "both genders. Output only names, one per line, no numbering.",

    "List 80 Indian names from North-East India (Assam, Manipur, Meghalaya) "
    "and Odia/Bihari regions. Output only names, one per line, no numbering.",

    "List 80 modern and trendy Indian baby names (post-2000 era), "
    "both genders. Output only names, one per line, no numbering.",
]


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def clean_name(raw: str) -> str:
    """Strip bullets, numbers, punctuation; keep only alphabetical name."""
    name = re.sub(r"^[\d\.\-\*\•\–\—\s]+", "", raw).strip()
    name = re.sub(r"\s+", " ", name)          # normalise internal spaces
    # Accept only names made of letters (and optionally a space for compound names)
    if re.fullmatch(r"[A-Za-z]+([ \-][A-Za-z]+)*", name):
        return name
    return ""


def call_claude(client, prompt: str) -> list:
    """Call the Claude API and return a cleaned list of names."""
    msg = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    raw_text = msg.content[0].text
    names = []
    for line in raw_text.splitlines():
        name = clean_name(line.strip())
        if name:
            names.append(name)
    return names


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def generate_dataset(target: int = 1000, out_path: str = "TrainingNames.txt") -> list:
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    client  = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()

    seen       = set()
    all_names  = []

    for i, prompt in enumerate(PROMPTS, 1):
        print(f"[{i}/{len(PROMPTS)}] Fetching batch …", end=" ", flush=True)
        try:
            batch = call_claude(client, prompt)
        except Exception as exc:
            print(f"ERROR – {exc}")
            continue

        added = 0
        for name in batch:
            key = name.lower()
            if key not in seen:
                seen.add(key)
                all_names.append(name)
                added += 1

        print(f"got {len(batch)} → +{added} unique  (total {len(all_names)})")

        if len(all_names) >= target:
            break

    final = all_names[:target]
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(final))

    print(f"\n✓ Saved {len(final)} names to '{out_path}'")
    return final


def main():
    parser = argparse.ArgumentParser(description="Generate Indian name dataset using Claude API")
    parser.add_argument("--out",    default="TrainingNames.txt")
    parser.add_argument("--target", type=int, default=1000, help="Target number of unique names")
    args = parser.parse_args()
    generate_dataset(target=args.target, out_path=args.out)


if __name__ == "__main__":
    main()
