"""PicoThink — Embeddable AI model with online training."""

__version__ = "0.1.0"


def __getattr__(name):
    if name == "Config":
        from pico_think.config import Config
        return Config
    if name == "PicoThink":
        from pico_think.model import PicoThink
        return PicoThink
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
