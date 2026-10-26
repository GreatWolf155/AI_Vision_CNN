"""CNN Vision - Streamlit site (stage 3 of the project).

Started by main.py. To run only the site:  streamlit run streamlit_app.py
"""

import os

import streamlit as st

from app_pages import common

# Page files are found next to this file, also when it is started from deploy/streamlit_app.py
PAGES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app_pages")

common.init_state()
mobile = common.is_mobile()

st.set_page_config(
    page_title="CNN Vision",
    page_icon="🔮",
    layout="centered" if mobile else "wide",
    initial_sidebar_state="collapsed" if mobile else "expanded",
)
common.inject_css(mobile)

pages = [
    st.Page(os.path.join(PAGES_DIR, "classify.py"), title="Classify", icon="🔍", default=True),
    st.Page(os.path.join(PAGES_DIR, "statistics.py"), title="Model statistics", icon="📊"),
    st.Page(os.path.join(PAGES_DIR, "api_guide.py"), title="REST API", icon="🔌"),
]
# The sidebar always holds the page menu and the layout setting - in the phone layout too,
# so the sidebar is never empty when it is opened
page = st.navigation(pages, position="sidebar")
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    common.view_mode_picker()

if mobile:
    # The phone layout keeps the sidebar folded away, so also show the pages as a row of links
    with st.container(horizontal=True, horizontal_alignment="distribute"):
        st.page_link(pages[0], label="Classify", icon="🔍")
        st.page_link(pages[1], label="Statistics", icon="📊")
        st.page_link(pages[2], label="API", icon="🔌")

page.run()
