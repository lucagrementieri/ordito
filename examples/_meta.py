"""Metadata every example module declares: its id, title, summary and the examples it credits."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Meta:
    """What the gallery page shows around an example's code and image."""

    id: str
    """Short identifier (``"T1"``), also the image file stem and the page anchor."""
    title: str
    summary: str
    """One or two sentences of Markdown placed above the code."""
    credits: tuple[tuple[str, str], ...] = field(default_factory=tuple)
    """``(label, url)`` pairs: the reference libraries' examples this one is modelled on."""
    notes: str = ""
    """Optional Markdown placed below the image."""

    @property
    def anchor(self) -> str:
        """The page anchor and the image stem."""
        return self.id.lower()
