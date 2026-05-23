"""Template-only Streamlit router stub.

This file is a template only and not referenced by any central entrypoint in
this repo yet.
Rename/copy it to i_ching_app.py (or app.py) when ready to establish a central
router.
"""

from __future__ import annotations

import streamlit as st


def render() -> None:
    """Render a minimal sidebar router with Atlas Bridge as the only page."""
    page = st.sidebar.selectbox("Page", ["Atlas Bridge"])

    if page == "Atlas Bridge":
        import apps.streamlit.atlas_bridge as atlas_bridge

        atlas_bridge.render()


if __name__ == "__main__":
    render()
