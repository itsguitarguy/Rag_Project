"""Small text helpers shared across modules."""
import re

STOP = set("""a an the and or but if of to in on at for with by from is are was were be been it this that
these those as not no do does did has have had which who whom what when where why how can could should
would will may might than then there their they them its into about over under""".split())


def tokenize(text):
    """Lowercase, split into alphanumeric words, drop stop-words."""
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP]


def normalize(s):
    """Collapse whitespace + lowercase, for comparing quotes against source text."""
    return re.sub(r"\s+", " ", s).strip().lower()
