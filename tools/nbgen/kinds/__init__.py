"""Every model type: one module per type holds its models. A new type = a new module + a TYPES entry."""

from ..config import TYPES
from . import omni, text, vision_language

_BY_KIND = {"text": text, "vision-language": vision_language, "omni": omni}
assert set(_BY_KIND) == set(TYPES), "every entry of TYPES needs a module in nbgen/kinds, and the other way round"

MODELS = [model for kind in TYPES for model in _BY_KIND[kind].MODELS]  # README order: type, then list order
for _model in MODELS:
    assert _model["kind"] in _BY_KIND and _model in _BY_KIND[_model["kind"]].MODELS
