#!/usr/bin/env python3
"""Assemble v20 text around accepted v19 viewports and registered v20 figure markup (no asset redraws)."""
from __future__ import annotations
from release_runtime import private_literal as _private_literal, release_path as _release_path

import argparse
import csv
import difflib
import hashlib
import html as escape_html
import json
import re
import unicodedata
from pathlib import Path

from lxml import etree, html
from figure_assets import approved_edits, current_figure_sections, display_asset_mapping, validate_display_mapping, protected_asset
from revision_config import expected_figure_counts
from reference_policy import validate_reference_rows

ROOT = Path(_release_path('path_052a3da416b1c4ab'))
LOCAL_BASE = ROOT / "reproducibility/frozen_v19_baseline"
DEFAULT_BASE = LOCAL_BASE if LOCAL_BASE.is_dir() else ROOT.parent / "revision_v19_visual_reference_revision"
DOCS = {"manuscript": "manuscript.md", "supplement": "supplementary_methods.md"}
CITE = re.compile(r"\[\s*@([A-Za-z0-9_]+)(?:\s*;\s*@([A-Za-z0-9_]+))*\s*\]")
CITE_GROUP = re.compile(r"\[\s*@[A-Za-z0-9_]+(?:\s*;\s*@[A-Za-z0-9_]+)*\s*\]")
SECTION = re.compile(r'<section\b(?=[^>]*\bclass="[^"]*\bfigure-unit\b)[^>]*>.*?</section>', re.S)
FIGURE = re.compile(r"<figure\b[^>]*>.*?</figure>", re.S)
SCIENTIFIC_LITERAL = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)×10\^(-?\d+)(?![\w.])")
# Frozen supplementary captions measured at the accepted 180-mm figure width.
# Heading + image + caption occupy 233.1, 257.1, 215.6 and 257.1 mm, respectively.
SUPPLEMENT_INLINE_CAPTIONS = {1, 4, 7, 14}
INLINE = re.compile(
    r"\\\(.*?\\\)|log10\(fraction\s*\+\s*10\^-8\)|\[\s*@[A-Za-z0-9_]+(?:\s*;\s*@[A-Za-z0-9_]+)*\s*\]"
    r"|(?<![\w.])\d+(?:\.\d+)?×10\^-?\d+(?![\w.])"
    r"|`[^`\n]+`|\[[^\]\n]+\]\([^\s)]+\)|\*\*[^*\n]+\*\*|(?<![\w*])\*[^*\n]+\*(?![\w*])",
    re.S,
)


def digest(data: bytes | str) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def cite_keys(text: str) -> list[str]:
    return [key for group in CITE_GROUP.findall(text) for key in re.findall(r"@([A-Za-z0-9_]+)", group)]


def parse_source(source: str) -> tuple[dict, str, str, list[tuple[str, str]]]:
    """Return front matter, prose, general figure notes, and numbered legends."""
    metadata = {}
    if source.startswith("---\n"):
        front, source = source[4:].split("\n---\n", 1)
        for line in front.splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                metadata[key.strip()] = value.strip().strip('"').strip("'")
    source = re.sub(r"\n# References\s*\Z", "", source).strip()
    parts = re.split(r"(?m)^# (?:Supplementary figure legends|Figure legends)\s*$", source)
    if len(parts) != 2:
        raise ValueError("Expected one figure-legend heading")
    prose, legend_source = parts
    matches = list(re.finditer(r"(?m)^## ((?:Supplementary )?Figure (\d+)\.[^\n]*)$", legend_source))
    if not matches:
        raise ValueError("No numbered figure legends")
    preamble = legend_source[:matches[0].start()].strip()
    legends = []
    for index, match in enumerate(matches):
        if int(match[2]) != index + 1:
            raise ValueError("Legend headings must follow displayed figure order")
        end = matches[index + 1].start() if index + 1 < len(matches) else len(legend_source)
        legends.append((match[1], legend_source[match.end():end].strip()))
    return metadata, prose.strip(), preamble, legends


def frozen_math(base_md: str, base_html: str) -> dict[str, str]:
    tokens = re.findall(r"\\\(.*?\\\)|log10\(fraction\s*\+\s*10\^-8\)", base_md, re.S)
    expressions = [token[2:-2] if token.startswith("\\(") else re.sub(r"\s+", "", token) for token in tokens]
    tree = html.fromstring(base_html)
    spans = tree.xpath('//span[contains(concat(" ", normalize-space(@class), " "), " math ")]')
    if len(expressions) != len(spans):
        raise ValueError(f"Frozen math count mismatch: {len(expressions)} versus {len(spans)}")
    return {formula: etree.tostring(span, encoding="unicode", method="html", with_tail=False)
            for formula, span in zip(expressions, spans)}


def slug(title: str) -> str:
    value = unicodedata.normalize("NFKC", title).lower()
    return re.sub(r"[^\w.-]+", "-", value).strip("-")


class Markdown:
    """A bounded parser for the manuscript's existing Markdown features."""
    def __init__(self, numbers: dict[str, int], math: dict[str, str]):
        self.numbers = numbers
        self.math = math
        self.elementary_math = set()
        self.scientific_notation = []

    def inline(self, source: str) -> str:
        output, cursor = [], 0
        for match in INLINE.finditer(source):
            output.append(escape_html.escape(source[cursor:match.start()]))
            token = match[0]
            if token.startswith("\\(") or token.startswith("log10("):
                formula = token[2:-2] if token.startswith("\\(") else re.sub(r"\s+", "", token)
                if formula in self.math:
                    frozen = re.sub(r'\sdata-latex="[^"]*"', '', self.math[formula], count=1)
                    output.append(re.sub(r"<span\b", lambda _: '<span data-latex="' + escape_html.escape(formula, quote=True) + '"', frozen, count=1))
                elif re.fullmatch(r"[A-Za-z]", formula):
                    # A standalone variable needs only italic type, not a TeX runtime.
                    self.elementary_math.add(formula)
                    output.append('<span class="math inline" data-latex="' + formula + '"><em>' + formula + '</em></span>')
                else:
                    raise ValueError(f"Unrecognized formula; preserve a frozen v19 formula: {formula}")
            elif (scientific := SCIENTIFIC_LITERAL.fullmatch(token)) is not None:
                coefficient, exponent = scientific.groups()
                output.append('<span class="scientific-notation" data-source-literal="' + token + '">' + coefficient + '×10<sup>' + exponent + '</sup></span>')
                self.scientific_notation.append({"source_literal": token, "coefficient": coefficient, "base": "10", "exponent": exponent, "superscript": True})
            elif token.startswith("[@") or re.match(r"\[\s*@", token):
                keys = re.findall(r"@([A-Za-z0-9_]+)", token)
                links = [f'<a href="#ref-{key}">{self.numbers[key]}</a>' for key in keys]
                output.append(f'<span class="citation" data-cite-keys="{";".join(keys)}">[{", ".join(links)}]</span>')
            elif token.startswith("`"):
                output.append("<code>" + escape_html.escape(token[1:-1]) + "</code>")
            elif token.startswith("**"):
                output.append("<strong>" + self.inline(token[2:-2]) + "</strong>")
            elif token.startswith("*"):
                output.append("<em>" + self.inline(token[1:-1]) + "</em>")
            else:
                link = re.fullmatch(r"\[([^\]]+)\]\(([^)]+)\)", token)
                assert link is not None
                if not re.match(r"^(?:https?://|#)", link[2]):
                    raise ValueError(f"Unexpected link scheme: {link[2]}")
                output.append(f'<a href="{escape_html.escape(link[2], quote=True)}">{self.inline(link[1])}</a>')
            cursor = match.end()
        output.append(escape_html.escape(source[cursor:]))
        return "".join(output)

    def blocks(self, source: str) -> str:
        result = []
        for block in re.split(r"\n\s*\n", source.strip()):
            lines = block.strip().splitlines()
            if not lines:
                continue
            heading = re.match(r"^(#{1,6})\s+(.+)$", lines[0])
            if heading:
                if len(lines) != 1:
                    raise ValueError("Heading and paragraph require a blank line")
                level, title = len(heading[1]), heading[2]
                result.append(f'<h{level} id="{slug(title)}">{self.inline(title)}</h{level}>')
            elif len(lines) >= 2 and lines[0].lstrip().startswith("|") and re.fullmatch(r"[| :\-]+", lines[1]):
                rows = [[cell.strip() for cell in line.strip().strip("|").split("|")]
                        for line in [lines[0], *lines[2:]]]
                if any(len(row) != len(rows[0]) for row in rows):
                    raise ValueError("Malformed Markdown table")
                cells = lambda row, tag: "".join(f"<{tag}>{self.inline(cell)}</{tag}>" for cell in row)
                result.append("<table>\n<thead><tr>" + cells(rows[0], "th") + "</tr></thead>\n<tbody>" +
                              "\n".join("<tr>" + cells(row, "td") + "</tr>" for row in rows[1:]) + "</tbody></table>")
            else:
                if any(re.match(r"^(?:```|>|[-*] |\d+\. )", line) for line in lines):
                    raise ValueError("Unsupported Markdown block; use manuscript paragraphs or tables")
                result.append("<p>" + self.inline(" ".join(line.strip() for line in lines)) + "</p>")
        return "\n".join(result)


def assemble_document(source: str, base_md: str, base_html: str, numbers: dict, refs: dict, kind: str, extra_sections: list[str] | None = None, accepted_sections: list[str] | None = None, expected_count: int | None = None) -> tuple[str, dict]:
    metadata, prose, preamble, legends = parse_source(source)
    figure_sections = (accepted_sections if accepted_sections is not None else SECTION.findall(base_html)) + (extra_sections or [])
    expected = len(legends)
    if expected_count is not None and expected != expected_count:
        raise ValueError(f"{kind}: final legend count {expected} differs from registry count {expected_count}")
    if len(figure_sections) != expected or len(legends) != expected:
        raise ValueError(f"{kind}: {len(figure_sections)} frozen figures, {len(legends)} legends, expected {expected}")
    parser = Markdown(numbers, frozen_math(base_md, base_html))
    head = re.search(r"<head\b[^>]*>.*?</head>", base_html, re.S)[0]
    title = metadata.get("title", "Supplementary Methods and Results")
    date = metadata.get("date", "6 October 2026")
    head = re.sub(r"<title>.*?</title>", "<title>" + escape_html.escape(title) + "</title>", head, flags=re.S)
    head = re.sub(r'(<meta name="dcterms.date" content=")[^"]*(")', r'\g<1>2026-10-06\2', head)
    head = head.replace("</head>", '<style>#general-figure-notes{break-inside:avoid}</style>'
        '<style id="journal-table-rules">table{border-collapse:collapse;border-top:1pt solid #000;'
        'border-bottom:1pt solid #000;border-left:0;border-right:0}'
        'table th,table td{border:0;background:#fff}'
        'table thead th{border-bottom:0.75pt solid #000;background:#fff}'
        'table thead{display:table-header-group}</style></head>')
    if extra_sections or any("new-figure" in section for section in figure_sections):
        # Fit the new SVGs as complete viewports on A4, leaving room for headings.
        # Existing figures retain their frozen typography and sizing rules.
        head = head.replace("</head>", '<style>.figure-unit.new-figure figure>svg{max-height:244mm;max-width:180mm;width:auto;height:auto;margin:0 auto}</style></head>')
    if kind == "supplement":
        head = head.replace("</head>", '<style>.figure-unit.tall.caption-inline .figure-legend{break-before:auto}</style></head>')
    header = '<header id="title-block-header"><h1 class="title">' + parser.inline(title) + '</h1><p class="date">' + parser.inline(date) + '</p></header>'
    reference_html = ['<h1 id="references">References</h1>', '<div id="refs">']
    if kind == "supplement":
        reference_html.append('<p>Reference numbers follow the main article; references first cited here continue that sequence.</p>')
    keys = sorted(set(cite_keys(source)), key=numbers.get)
    for key in keys:
        ref = refs[key]
        reference_html.append(f'<p class="reference" id="ref-{key}" data-ref-key="{key}" data-ref-number="{numbers[key]}">'
                              f'[{numbers[key]}] {escape_html.escape(ref["formatted_reference"])} '
                              f'<a href="{escape_html.escape(ref["url"], quote=True)}">Source</a></p>')
    reference_html.append("</div>")
    revised_figures, records = [], []
    for ordinal, (section, (heading, legend)) in enumerate(zip(figure_sections, legends), 1):
        before_figure = FIGURE.search(section)[0]
        revised = re.sub(r"<h2\b[^>]*>.*?</h2>", "<h2>" + parser.inline(heading) + "</h2>", section, count=1, flags=re.S)
        revised = re.sub(r'(\A<section\b[^>]*\bid=")[^"]*(")', lambda match: match[1] + f'{kind}-figure-{ordinal}' + match[2], revised, count=1)
        if kind == "supplement" and ordinal in SUPPLEMENT_INLINE_CAPTIONS and 'caption-inline' not in html.fromstring(revised).get('class','').split():
            revised = re.sub(r'(\A<section\b[^>]*\bclass=")([^"]*)', lambda m: m[1] + m[2] + ' caption-inline', revised, count=1)
        revised, replaced = re.subn(r'(<div\b[^>]*class="figure-legend"[^>]*>).*?(</div>)',
                                   lambda m: m[1] + "\n" + parser.blocks(legend) + "\n" + m[2], revised, count=1, flags=re.S)
        if kind == "supplement" and 'tall' in html.fromstring(revised).get('class','').split() and ordinal not in SUPPLEMENT_INLINE_CAPTIONS:
            continuation = '<h2 class="caption-continuation">' + parser.inline(heading) + ' (continued)</h2>'
            revised = revised.replace('<div class="figure-legend">', '<div class="figure-legend">\n' + continuation, 1)
        if replaced != 1 or FIGURE.search(revised)[0] != before_figure:
            raise ValueError(f"Figure {ordinal}: frozen markup changed or legend not found")
        revised_figures.append(revised)
        tree = html.fromstring(revised)
        records.append({"display_ordinal": ordinal, "heading": heading, "section_id": tree.get("id"),
                        "baseline_ordinal": int(tree.get("data-baseline-ordinal", ordinal)),
                        "classes": tree.get("class"), "svg_nodes": len(tree.xpath('.//*[local-name()="svg"]')),
                        "figure_markup_sha256": digest(before_figure), "frozen_baseline_artwork": tree.get("data-registration-ordinal") is None, "registered_composition": tree.get("data-registration-ordinal") is not None})
    opening = re.search(r"<html\b[^>]*>", base_html)[0]
    prose_html = parser.blocks(prose)
    preamble_html = parser.blocks(preamble) if preamble else ""
    legend_heading = "Figure legends" if kind == "manuscript" else "Supplementary figure legends"
    notes_class = ''
    if kind == "supplement" and not preamble:
        # An empty heading belongs to S1, rather than the preceding references page.
        notes_class = ' class="legend-heading-only"'
        head = head.replace("</head>", '<style id="supplement-legend-heading-layout">'
            '#general-figure-notes.legend-heading-only{break-before:page;break-after:avoid}'
            '#general-figure-notes.legend-heading-only + .figure-unit{break-before:auto}'
            '</style></head>')
    notes = '<div id="general-figure-notes"' + notes_class + '><h1>' + legend_heading + '</h1>\n' + preamble_html + '</div>'
    output = "<!DOCTYPE html>\n" + opening + "\n" + head + "\n<body>\n" + header + '\n<div id="document-prose">\n' + prose_html + "\n</div>\n" + "\n".join(reference_html) + "\n" + notes + "\n" + "\n".join(revised_figures) + "\n</body>\n</html>\n"
    if re.search(r"\[\s*@|\\\(|\\\)|\b(?:TODO|TBD|FIXME)\b", output):
        raise ValueError("Unresolved citation, math or editorial token in assembled HTML")
    tree = html.fromstring(output)
    source_math = [token[2:-2] if token.startswith("\\(") else re.sub(r"\s+", "", token)
                   for token in re.findall(r"\\\(.*?\\\)|log10\(fraction\s*\+\s*10\^-8\)", source, re.S)]
    rendered_math = tree.xpath('//span[contains(concat(" ",normalize-space(@class)," ")," math ")]/@data-latex')
    if rendered_math != source_math:
        raise ValueError("Math expression sequence differs from the final frozen source")
    normalize = lambda text: re.sub(r"\s+", " ", text).strip()
    for label, expected_html, actual_nodes in [
        ("document prose", prose_html, tree.xpath('//*[@id="document-prose"]')),
        ("general figure notes", '<h1>' + legend_heading + '</h1>\n' + preamble_html, tree.xpath('//*[@id="general-figure-notes"]')),
    ]:
        expected_text = html.fromstring('<div>' + expected_html + '</div>').text_content()
        if len(actual_nodes) != 1 or normalize(actual_nodes[0].text_content()) != normalize(expected_text):
            raise ValueError(f"Text-preservation check failed: {label}")
    for section, (_, legend) in zip(tree.xpath('//section[contains(@class,"figure-unit")]'), legends):
        legend_copy = html.fromstring(etree.tostring(section.xpath('./div[@class="figure-legend"]')[0],method='html'))
        for continuation in legend_copy.xpath('./h2[@class="caption-continuation"]'):
            continuation.getparent().remove(continuation)
        actual = legend_copy.text_content()
        expected_text = html.fromstring('<div>' + parser.blocks(legend) + '</div>').text_content()
        if normalize(actual) != normalize(expected_text):
            raise ValueError("Figure legend text-preservation check failed")
    citations = tree.xpath('//span[@class="citation"]')
    rendered_keys = []
    for citation in citations:
        expected_keys = citation.get("data-cite-keys").split(";")
        links = citation.findall("a")
        if [(link.get("href"), link.text) for link in links] != [("#ref-" + key, str(numbers[key])) for key in expected_keys]:
            raise ValueError("Citation key/number/link mapping mismatch")
        rendered_keys.extend(expected_keys)
    if sorted(rendered_keys) != sorted(cite_keys(source)):
        raise ValueError("Rendered citation occurrences differ from source")
    for node in tree.xpath('//p[@data-ref-key]'):
        key = node.get("data-ref-key")
        if node.get("data-ref-number") != str(numbers[key]) or refs[key]["formatted_reference"] not in node.text_content():
            raise ValueError("Reference entry differs from frozen TSV")
    scientific_spans = tree.xpath('//span[@class="scientific-notation"]')
    if [node.get("data-source-literal") for node in scientific_spans] != [item[0] for item in SCIENTIFIC_LITERAL.finditer(source)]:
        raise ValueError("Scientific-notation occurrences differ from frozen source")
    for node in scientific_spans:
        match = SCIENTIFIC_LITERAL.fullmatch(node.get("data-source-literal"))
        if node.text != match[1] + '×10' or len(node) != 1 or node[0].tag != 'sup' or node[0].text != match[2]:
            raise ValueError("Scientific-notation numeric value changed")
    return output, {"figure_blocks": records, "citation_occurrences": len(rendered_keys), "citation_keys": keys,
                    "reference_mapping_exact": True, "all_prose_and_legends_preserved": True,
                    "general_figure_notes_preserved": True, "unresolved_tokens": [],
                    "elementary_math_symbols": sorted(parser.elementary_math),
                    "source_native_math_expressions": source_math,
                    "source_editable_table_count": len(tree.xpath('//*[@id="document-prose"]//table | //div[@class="figure-legend"]//table')),
                    "scientific_notation_typography": parser.scientific_notation,
                    "inline_caption_ordinals": sorted(SUPPLEMENT_INLINE_CAPTIONS) if kind == "supplement" else [],
                    "author_information_placeholders": len(re.findall(r"AUTHOR INFORMATION REQUIRED", prose))}


def validate_frozen_assets(root: Path, base: Path, check_historical_baseline: bool = False) -> dict:
    manifest_path = root / "audit/input_v19_sha256.json"
    if not manifest_path.exists():
        raise ValueError("Missing root-frozen v19 input manifest")
    inputs = json.loads(manifest_path.read_text())
    fixture_path = base / "baseline_fixture_manifest.json"
    fixture = json.loads(fixture_path.read_text()) if fixture_path.exists() else None
    input_failures = []
    if fixture:
        input_failures.extend(str(base / rel) for rel, checksum in fixture["files"].items()
            if not (base / rel).is_file() or digest((base / rel).read_bytes()) != checksum)
    check_live = check_historical_baseline or fixture is None
    if check_live:
        input_failures.extend(name for name, checksum in inputs.items()
            if not Path(name).is_file() or digest(Path(name).read_bytes()) != checksum)
    root_manifest_path = root / "audit/v19_baseline_manifest.json"
    root_manifest = json.loads(root_manifest_path.read_text()) if root_manifest_path.exists() else {}
    root_records = [{"path": str(Path(root_manifest["baseline"]) / relative), "sha256": checksum}
                    for relative, checksum in root_manifest.get("files", {}).items()]
    if check_live:
        input_failures.extend(record["path"] for record in root_records
            if not Path(record["path"]).is_file() or digest(Path(record["path"]).read_bytes()) != record["sha256"])
    allowed = approved_edits(root, base)
    derived_path = root / "audit/figure_derivatives.json"
    derivatives = json.loads(derived_path.read_text()) if derived_path.exists() else {}
    if allowed and derivatives.get("source_svg_sha256") != {path: entry["after"] for path, entry in allowed.items()}:
        raise ValueError("Approved SVG labels changed since derivative rendering; rerun refresh_figures.py")
    permitted = {**{path: entry["after"] for path, entry in allowed.items()}, **derivatives.get("output_sha256", {})}
    metadata_path = root / "audit/source_data_metadata_updates.json"
    metadata_updates = json.loads(metadata_path.read_text()) if metadata_path.exists() else []
    for entry in metadata_updates:
        if entry["path"] != "manuscript/SourceData/SOURCE_DATA_INDEX.tsv":
            raise ValueError("Unapproved copied metadata update path")
        original, target = base / entry["path"], root / entry["path"]
        if digest(original.read_bytes()) != entry["before"] or digest(target.read_bytes()) != entry["after"] or not target.read_bytes().startswith(original.read_bytes()):
            raise ValueError("Inherited SourceData index changed outside appended v20 provenance")
        permitted[entry["path"]] = entry["after"]
    asset_roots = ["figures", "manuscript/Figure_PDF", "manuscript/Figure_PNG", "manuscript/SourceData"]
    asset_failures, asset_counts = [], {}
    blueprints = fixture["asset_sha256"] if fixture else {
        str(path.relative_to(base)): digest(path.read_bytes()) for folder in asset_roots
        for path in (base / folder).rglob("*") if path.is_file() and path.name != '.DS_Store' and '__pycache__' not in path.parts}
    display_ledger = validate_display_mapping(root, base, blueprints)
    blueprints = {display_asset_mapping(relative): checksum for relative, checksum in blueprints.items()}
    for folder in asset_roots:
        records = {rel: checksum for rel, checksum in blueprints.items() if rel.startswith(folder + "/") and protected_asset(rel, root)}
        asset_counts[folder] = len(records)
        for relative, checksum in records.items():
            target = root / relative
            if not target.is_file():
                asset_failures.append(str(target))
            elif checksum != digest(target.read_bytes()):
                if permitted.get(relative) != digest(target.read_bytes()):
                    asset_failures.append(str(target))
        source_names = {str(Path(rel).relative_to(folder)) for rel in records}
        target_names = {str(path.relative_to(root / folder)) for path in (root / folder).rglob("*") if path.is_file() and path.name != '.DS_Store' and '__pycache__' not in path.parts}
        if not source_names.issubset(target_names):
            asset_failures.append(f"{folder}: frozen source files missing")
    if input_failures or asset_failures:
        raise ValueError(f"Frozen input or copied asset mismatch: {input_failures + asset_failures}")
    extra_assets = {folder: sorted(str(path.relative_to(root)) for path in (root / folder).rglob("*") if path.is_file() and str(path.relative_to(root)) not in blueprints) for folder in asset_roots}
    return {"v19_frozen_files": len(inputs), "root_v19_frozen_files": len(root_records),
            "both_v19_manifests_unchanged": True if check_live else None,
            "original_v19_live_verification": check_live, "local_baseline_fixture_verified": fixture is not None,
            "v19_unchanged": True if check_live else None, "copied_asset_counts": asset_counts,
            "new_v20_assets": extra_assets,
            "approved_label_edits": allowed, "registered_derivatives": derivatives.get("output_sha256", {}),
            "figure_display_mapping": display_ledger,
            "registered_metadata_updates": metadata_updates,
            "unregistered_copied_asset_changes": []}


def numeric_tokens(text: str) -> set[str]:
    return set(re.findall(r"(?<![\w])\d(?:[\d,]*\d)?(?:\.\d+)?(?:/\d(?:[\d,]*\d)?(?:\.\d+)?)?", text))


def protected_facts(sources: dict[str, str], root: Path | None = None) -> dict:
    """Inventory frozen-source facts without enforcing obsolete prose placement.

    Exact paragraph/render preservation protects every source value. Scientific
    review owns whether rewritten/relocated claims remain source-supported.
    Root-owned requirements can add explicit cross-document factual anchors.
    """
    text = "\n".join(sources.values())
    anchors = ["137", "138", "202", "339", "126", "163", "70", "56", "109", "54", "289", "19", "385", "185",
               "248", "0.230", "53/137", "2.26", "0.97", "5.45", "0.044", "0.00564", "0.690476", "0.699",
               "205/681,884", "439/551,367", "359/546,836"]
    tokens = numeric_tokens(text)
    records = {anchor: anchor in tokens for anchor in anchors}
    records.update({sequence: sequence in text for sequence in [_private_literal('private_45bbaa37a9607bb7'), _private_literal('private_b3445b7fc78bf2f0')]})
    requirements_path = root / "audit/protected_fact_requirements.json" if root else None
    requirements = json.loads(requirements_path.read_text()) if requirements_path and requirements_path.exists() else {}
    required = requirements.get("numeric_anchors", []) + requirements.get("sequence_anchors", [])
    missing = [anchor for anchor in required if anchor not in tokens and anchor not in text]
    if missing:
        raise ValueError(f"Missing current root-required factual anchors: {missing}")
    return {"baseline_anchor_presence_for_scientific_review": records,
            "current_root_required_anchors": required, "required_anchors_present": True,
            "frozen_source_numeric_tokens": {kind: sorted(numeric_tokens(source)) for kind, source in sources.items()},
            "frozen_source_sha256": {kind: digest(source) for kind, source in sources.items()},
            "statistical_anchor_domains": {"0.230": "248-test sequence-cluster family minimum q; retained multiplicity correction",
                "0.044": "study-cohort raw Shannon individual P, rounded from 0.0443804706294422",
                "0.00564": "Q1 individual prevalence comparison P",
                "0.690476": "Q1 primary exact two-sided rank-sum P for ten quantifiable CRC samples, five pN0 and five pN+",
                "0.699": "Q1 older twelve-estimate nodal sensitivity P, including near-LOQ A1/B1; rounded from 0.6991341991341992"},
            "historical_metric_q_required_in_prose": False,
            "interpretation": "Every final-source paragraph/value must survive HTML/PDF/Word exactly. Baseline token omissions/relocations are reviewed scientifically, rather than forcing old prose into the new narrative."}


def require_prose_freeze(root: Path, sources: dict[str, str], selected: list[str]) -> dict:
    path = root / "audit/prose_freeze.json"
    if not path.is_file():
        raise ValueError("Final v20 prose freeze is required before production assembly")
    freeze = json.loads(path.read_text())
    for kind in selected:
        name = "main" if kind == "manuscript" else "supplement"
        if freeze.get(name) != "FROZEN" or freeze.get(name + "_sha256") != digest(sources[kind]):
            raise ValueError(f"{kind}: source is not frozen or differs from root's final prose freeze")
    return {"path": str(path.relative_to(root)), "sha256": digest(path.read_bytes()), "selected_sources_frozen": selected}


def metric_statistics_receipt(root: Path) -> dict:
    """Bind individual metric P values to frozen tables without changing their q columns."""
    folder = root / "manuscript/SourceData/nodal"
    specifications = {
        "expanded12_unadjusted.tsv": ("metric", "direct rank comparisons", 12),
        "expanded12_adjusted.tsv": ("feature", "covariate-adjusted logistic ORs; P tests the metric coefficient", 12),
        "Figure2_descriptive_tests_v4.tsv": ("feature", "three direct summary comparisons", 3),
    }
    result = {}
    for filename, (feature_column, interpretation, expected_size) in specifications.items():
        path = folder / filename
        rows = list(csv.DictReader(path.open(), delimiter="\t"))
        groups = {}
        records = []
        for row in rows:
            group = (row.get("cohort", ""), row.get("analysis", ""))
            groups.setdefault(group, []).append(row[feature_column])
            if not 0 <= float(row["P"]) <= 1 or int(row["family_size"]) != expected_size:
                raise ValueError(f"Invalid archived individual metric P/family size: {filename}")
            records.append({key: row[key] for key in ["cohort", "analysis", feature_column, "P", "q", "OR", "CIlo", "CIhi", "covariates"] if key in row})
        if any(len(features) != expected_size or len(set(features)) != expected_size for features in groups.values()):
            raise ValueError(f"Archived metric table has incomplete or duplicated groups: {filename}")
        result[str(path.relative_to(root))] = {"sha256": digest(path.read_bytes()),
            "interpretation": interpretation, "individual_P_records": records,
            "archival_q_columns_preserved": True, "q_required_in_v20_metric_presentation": False}
    return {"tables": result,
            "policy": "V20 presents individual P for repertoire/technical summary metrics. Archived 12/3-summary q values are retained as provenance. Covariate adjustment of an OR does not imply multiplicity adjustment. Specified sequence-cluster, antigen and state families retain their own corrections."}


def extra_figure_sections(root: Path, kind: str, source: str) -> list[str]:
    """All preserved and newly registered sections come from the current inventory."""
    expected = expected_figure_counts(root)[kind]
    if len(parse_source(source)[3]) != expected:
        raise ValueError(f'{kind}: final legends do not match the {expected} registered figure entries')
    return []


def paragraph_diffs(baseline: str, revised: str) -> list[dict]:
    before = [p.strip() for p in re.split(r"\n\s*\n", baseline) if p.strip()]
    after = [p.strip() for p in re.split(r"\n\s*\n", revised) if p.strip()]
    matcher = difflib.SequenceMatcher(a=before, b=after, autojunk=False)
    return [{"operation": op, "baseline_paragraphs": [i1 + 1, i2], "revised_paragraphs": [j1 + 1, j2],
             "original": before[i1:i2], "revised": after[j1:j2]}
            for op, i1, i2, j1, j2 in matcher.get_opcodes() if op != "equal"]


def self_test(base: Path) -> None:
    """Exercise only immutable v19 input; no draft or deliverable is written."""
    refs = {r["id"]: r for r in csv.DictReader((base / "manuscript/references_resolved.tsv").open(), delimiter="\t")}
    sources = {kind: (base / "manuscript" / filename).read_text() for kind, filename in DOCS.items()}
    all_keys = list(dict.fromkeys(cite_keys(sources["manuscript"]) + cite_keys(sources["supplement"])))
    numbers = {key: i + 1 for i, key in enumerate(all_keys)}
    for kind, filename in DOCS.items():
        output, record = assemble_document(sources[kind], sources[kind], (base / "manuscript" / f"CRC_TCR_{kind}_v19.html").read_text(), numbers, refs, kind)
        assert record["reference_mapping_exact"] and "<svg" in output
    parser = Markdown({"EXAMPLE": 1}, {})
    assert '<strong>bold</strong>' in parser.blocks("# Heading\n\n**bold** and *italic*, `code` [@EXAMPLE].")
    assert "<table>" in parser.blocks("| A | B |\n|---|---|\n| 1 | 2 |")
    assert "&lt;" in parser.inline("P < 0.001")
    assert '<em>i</em>' in parser.inline(r"Index \(i\)")
    assert '<sup>-19</sup>' in parser.inline('1.02×10^-19')
    protected_facts(sources)
    print(json.dumps({"self_test": "passed", "documents": 2, "figure_blocks": sum(len(parse_source(value)[3]) for value in sources.values()), "citation_keys": len(all_keys)}))


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--root", type=Path, default=ROOT)
    cli.add_argument("--base", type=Path, default=DEFAULT_BASE)
    cli.add_argument("--self-test", action="store_true")
    cli.add_argument("--only", choices=list(DOCS), help="Assemble only a frozen document; read the other source solely for citation numbering")
    cli.add_argument("--check-historical-baseline", action="store_true", help="Also verify the original workspace v19 files; unnecessary for a standalone archive rebuild")
    args = cli.parse_args()
    root, base = args.root.resolve(), args.base.resolve()
    if root == base:
        raise ValueError("Output root cannot be v19")
    if args.self_test:
        self_test(base)
        return
    sources = {kind: (root / "manuscript" / filename).read_text() for kind, filename in DOCS.items()}
    baselines = {kind: (base / "manuscript" / filename).read_text() for kind, filename in DOCS.items()}
    selected = [args.only] if args.only else list(DOCS)
    prose_freeze = require_prose_freeze(root, sources, selected)
    previous_path = root / "audit/build_validation.json"
    previous = json.loads(previous_path.read_text()) if args.only and previous_path.exists() else {}
    retained = {kind: record for kind, record in previous.get("documents", {}).items()
                if kind not in selected and record["markdown_sha256"] == digest(sources[kind])
                and (root / "manuscript" / f"CRC_TCR_{kind}_v20.html").is_file()
                and digest((root / "manuscript" / f"CRC_TCR_{kind}_v20.html").read_bytes()) == record["html_sha256"]}
    validation = {"phase": "assembled", "assembled_kinds": selected,
        "assembly_scope": "partial" if args.only else "complete",
        "assets": validate_frozen_assets(root, base, args.check_historical_baseline), "documents": retained,
        "prose_freeze": prose_freeze}
    reference_path = root / "manuscript/references_resolved.tsv"
    refs = {r["id"]: r for r in csv.DictReader(reference_path.open(), delimiter="\t")}
    baseline_refs = {r["id"]: r for r in csv.DictReader((base / "manuscript/references_resolved.tsv").open(), delimiter="\t")}
    reference_policy = validate_reference_rows(root, baseline_refs, refs)
    for key in set(refs) - set(baseline_refs):
        if not refs[key].get("formatted_reference") or not re.match(r"https?://", refs[key].get("url", "")):
            raise ValueError(f"Incomplete added reference: {key}")
    all_keys = list(dict.fromkeys(cite_keys(sources["manuscript"]) + cite_keys(sources["supplement"])))
    if set(all_keys) - refs.keys():
        raise ValueError(f"Unresolved reference keys: {set(all_keys) - refs.keys()}")
    numbers = {key: i + 1 for i, key in enumerate(all_keys)}
    validation["protected_facts"] = protected_facts(sources, root)
    if retained and previous.get("citations", {}).get("key_to_number") != numbers:
        raise ValueError("Combined citation numbering changed; run complete assembly instead of retaining the other document")
    validation["metric_statistics"] = metric_statistics_receipt(root)
    validation["citations"] = {"all_keys_resolve": True, **reference_policy,
        "new_registered_keys": sorted(set(refs) - set(baseline_refs)),
        "key_set_changes": {kind: {"added": sorted(set(cite_keys(sources[kind])) - set(cite_keys(baselines[kind]))), "removed": sorted(set(cite_keys(baselines[kind])) - set(cite_keys(sources[kind])))} for kind in DOCS},
        "numbering": "combined first citation: main then supplement", "key_to_number": numbers}
    pending = {}
    for kind in selected:
        filename = DOCS[kind]
        baseline_html = (base / "manuscript" / f"CRC_TCR_{kind}_v19.html").read_text()
        accepted = current_figure_sections(root, base, kind, baseline_html, SECTION, FIGURE)
        output, record = assemble_document(sources[kind], baselines[kind], baseline_html, numbers, refs, kind, extra_figure_sections(root, kind, sources[kind]), accepted, expected_figure_counts(root)[kind])
        pending[kind] = output
        record["markdown_sha256"] = digest(sources[kind])
        record["html_sha256"] = digest(output)
        validation["documents"][kind] = record
    # All substantive validation completes before any deliverable is written.
    for kind in selected:
        filename = DOCS[kind]
        html_path = root / "manuscript" / f"CRC_TCR_{kind}_v20.html"
        if not html_path.exists() or html_path.read_text() != pending[kind]:
            html_path.write_text(pending[kind])
        write_json(root / "audit" / f"{kind}_paragraph_changes.json", paragraph_diffs(baselines[kind], sources[kind]))
        (root / "audit" / f"{kind}.diff").write_text("".join(difflib.unified_diff(baselines[kind].splitlines(True), sources[kind].splitlines(True), fromfile="v19/" + filename, tofile="v20/" + filename)))
    write_json(root / "audit/build_validation.json", validation)
    write_json(root / "audit/citation_numbering.json", numbers)
    write_json(root / "audit/source_sha256.json", {str(path.relative_to(root)): digest(path.read_bytes()) for path in [reference_path, *[root / "manuscript" / DOCS[kind] for kind in validation["documents"]]]})
    print(json.dumps({"assembly": "passed", "scope": validation["assembly_scope"],
        "figure_counts": {kind: len(record["figure_blocks"]) for kind, record in validation["documents"].items()}, "references": len(all_keys)}))


if __name__ == "__main__":
    main()
