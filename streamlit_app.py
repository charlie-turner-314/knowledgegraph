from __future__ import annotations

import streamlit as st

from app.core.config import settings
from app.core import logging as _logging  # noqa: F401 ensures logging configured
from app.data.db import init_db
from app.ui import manage, query, review, upload, visualization


TAB_MAP = {
    "Ingest": upload.render,
    "Review": review.render,
    "Visualize": visualization.render,
    "Query": query.render,
    "Manage": manage.render,
}


def bootstrap() -> None:
    """Perform one-time bootstrap tasks."""

    init_db()


def main() -> None:
    st.set_page_config(page_title=settings.app_name, layout="wide")
    bootstrap()

    st.sidebar.title(settings.app_name)
    st.sidebar.caption("MVP workbench for SME-curated knowledge graphs")

    tab_name = st.sidebar.radio("Workflow", list(TAB_MAP.keys()), index=0)
    render_tab = TAB_MAP[tab_name]
    render_tab()


if __name__ == "__main__":
    main()
