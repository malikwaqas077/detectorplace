"""A Plotly chart that reports which point was clicked, including in 3D scenes.

Streamlit's built-in st.plotly_chart selection only works for 2D charts, so clicking a detector in the
3D view would do nothing. This small local component draws the same figure and sends the clicked
point (its text label, e.g. "FD-02") back to Python. plotly.js is served from the package itself, so it
works offline.
"""
from __future__ import annotations

from pathlib import Path

import streamlit.components.v1 as components

_component = components.declare_component("plot3d_click", path=str(Path(__file__).parent / "plot3d_component"))


def clickable_chart(fig, height=650, key=None):
    """Draw `fig`; returns the last click as a dict {tag, trace, x, y, z, nonce} or None."""
    return _component(fig=fig.to_json(), height=height, key=key, default=None)
