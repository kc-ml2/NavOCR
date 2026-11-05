"""The six prompts. ``{items}`` is replaced with the OCR strings as a JSON array.

p1 bare instruction, p2 explicit reject criterion, p3 role framing, p4 one-shot
example, p5 long keep/drop rules, p6 p1 plus typo tolerance.
"""
from __future__ import annotations

import json

PROMPTS: dict[str, str] = {
    # p1: bare instruction.
    "p1": (
        "Pick place names (shops, buildings, rooms) from this OCR text. "
        "Ignore ads, prices, promos.\n"
        "Return a JSON array.\n"
        "TEXT: {items}"
    ),
    # p2: explicit reject criterion.
    "p2": (
        "From OCR text, return only text that identifies a place "
        "(shop, building, room, floor).\n"
        "Drop ads, prices, slogans.\n"
        "JSON array only.\n"
        "TEXT: {items}"
    ),
    # p3: role framing.
    "p3": (
        "A robot navigates by signboard names. From this OCR text, return the "
        "text naming the place.\n"
        "Nothing else. JSON array.\n"
        "TEXT: {items}"
    ),
    # p4: one-shot example.
    "p4": (
        "Keep only place-identifying text; drop ads and prices.\n"
        'Example: ["SALE", "STARBUCKS", "-50%"] -> ["STARBUCKS"]\n'
        "Input: {items}\n"
        "Output:"
    ),
    # p5: long prompt with detailed conditions and edge-case rules.
    "p5": (
        "You are filtering OCR text extracted from indoor signboards for a "
        "navigation system. The text may be in any language. From the input "
        "list, keep only strings that identify a place, and return them as a "
        "JSON array of strings.\n"
        "\n"
        "KEEP a string if it satisfies any of these conditions:\n"
        "- It is the proper name of a shop, restaurant, cafe, or brand "
        "(e.g. a storefront name on a signboard).\n"
        "- It names a facility or room: restroom, elevator, exit, "
        "information desk, parking, or a numbered room/gate.\n"
        "- It is a floor or zone designator such as B1, 2F, or a wing/section "
        "label.\n"
        "\n"
        "DROP a string if any of these conditions apply:\n"
        "- Advertising or promotion: discounts, percentages, prices, "
        '"SALE", "OPEN", "NEW", seasonal event text.\n'
        "- Slogans, menu items, product names, or descriptive marketing "
        "copy.\n"
        "- Opening hours, phone numbers, URLs, or dates.\n"
        "- OCR noise: single stray characters or fragments that do not form "
        "a name.\n"
        "\n"
        "Edge-case rules:\n"
        "- If a string mixes a place name with promo text, keep it only when "
        "the place name dominates; do not rewrite or trim strings.\n"
        "- Keep each kept string exactly as it appears in the input, "
        "including case and spacing.\n"
        "- Do not deduplicate, translate, or reorder; preserve input order.\n"
        "- If nothing qualifies, return [].\n"
        "\n"
        "Return the JSON array only, with no explanation.\n"
        "TEXT: {items}"
    ),
    # p6: p1 plus typo tolerance, output pinned to verbatim copies.
    "p6": (
        "Pick place names (shops, buildings, rooms) from this OCR text. "
        "Ignore ads, prices, promos.\n"
        "OCR may contain a few typos; a misspelled place name still counts. "
        "Copy strings exactly as given, do not fix spelling.\n"
        "Return a JSON array.\n"
        "TEXT: {items}"
    ),
}


def render(key: str, items: list[str]) -> str:
    if key not in PROMPTS:
        raise KeyError(f"unknown prompt {key!r}; have {sorted(PROMPTS)}")
    payload = json.dumps(items, ensure_ascii=False)
    return PROMPTS[key].replace("{items}", payload)
