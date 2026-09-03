"""
The three axes an arm varies, read back from the config copied beside its outputs.

run_workflow.sh copies the arm's configs/*.json to predictions/<name_tag>/config.json
at submit time. Stages 04 and 05 read it from there — not from configs/ — so a result
row is labelled with what actually ran, even after the repository's configs move on.

The axes:
    engine          af3 | of3          which codebase produced the structures
    weights         af3_native | openbind
    use_templates   true | false       whether templates reached the model

They exist so that no table can be read as comparing two arms that in fact differed
in something unstated. Anything missing comes back as "" rather than as a default:
an arm whose template state is unknown must not look like a templates-off arm.

Imported by 04_collect_profiling.py and 05_score_dockq.py, which sit in this
directory (so a plain `import arm_metadata` resolves).
"""

import json
from pathlib import Path

# The three axes the comparison varies.
AXIS_COLUMNS = ("engine", "weights", "use_templates")

# What both result tables carry straight after `run`. `tag` is a SHORT display label
# for figures — the arm's pipeline in a few words, e.g. "AF3 code / openbind" — kept
# deliberately free of the template state. Stage 06 appends that only when the two
# arms being plotted actually disagree on it, so a legend never repeats context that
# is identical across every series. Falls back to the arm name when a config has none.
ARM_COLUMNS = ("tag", *AXIS_COLUMNS)


def read_arm(run_dir: Path) -> dict:
    """{engine, weights, use_templates, label} for one prediction directory.

    Values are strings; a key the config does not carry yields "". `use_templates`
    is normalised to "true"/"false" so the column is greppable.
    """
    run_dir = Path(run_dir)
    # `tag` defaults to the directory name so a figure always has SOME short label,
    # even for an arm with no config of ours (the paper's own predictions, say).
    arm = {"engine": "", "weights": "", "use_templates": "",
           "label": "", "tag": run_dir.name}
    config_path = run_dir / "config.json"
    if not config_path.is_file():
        return arm
    try:
        config = json.loads(config_path.read_text())
    except (OSError, json.JSONDecodeError):
        return arm

    for key in ("engine", "weights", "label", "tag"):
        value = config.get(key)
        if value is not None:
            arm[key] = str(value)
    if isinstance(config.get("use_templates"), bool):
        arm["use_templates"] = str(config["use_templates"]).lower()
    return arm
