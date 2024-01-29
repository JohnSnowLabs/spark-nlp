import streamlit as st

# Import for keyboard shortcuts
import streamlit.components.v1 as components

def load_keyboard_class():
    """This class enables to render some elements as if they were <kbd>.
    Without this class, currently <kbd> looks the same as <code> in Streamlit.
    Usage:
      load_keyboard_class()
      st.write('<span class="kbdx"> Press here </span>', unsafe_allow_html=True)
    """
    st.write(
        """<style>
        .kbdx {
        background-color: #eee;
        border-radius: 3px;
        border: 1px solid #b4b4b4;
        box-shadow: 0 1px 1px rgba(0, 0, 0, .2), 0 2px 0 0 rgba(255, 255, 255, .7) inset;
        color: #333;
        display: inline-block;
        font-size: .85em;
        font-weight: 700;
        line-height: 1;
        padding: 2px 4px;
        white-space: nowrap;
    }
    </style>""",
        unsafe_allow_html=True,
    )




def keyboard_to_url(
    key: str = None,
    key_code: int = None,
    url: str = None,
):
    """Map a keyboard key to open a new tab with a given URL.
    Args:
        key (str, optional): Key to trigger (example 'k'). Defaults to None.
        key_code (int, optional): If key doesn't work, try hard-coding the key_code instead. Defaults to None.
        url (str, optional): Opens the input URL in new tab. Defaults to None.
    """

    assert not (
        key and key_code
    ), """You can not provide key and key_code.
    Either give key and we'll try to find its associated key_code. Or directly
    provide the key_code."""

    assert (key or key_code) and url, """You must provide key or key_code, and a URL"""

    if key:
        key_code_js_row = f"const keyCode = '{key}'.toUpperCase().charCodeAt(0);"
    if key_code:
        key_code_js_row = f"const keyCode = {key_code};"

    components.html(
        f"""
<script>
const doc = window.parent.document;
buttons = Array.from(doc.querySelectorAll('button[kind=primary]'));
{key_code_js_row}
doc.addEventListener('keydown', function(e) {{
    e = e || window.event;
    var target = e.target || e.srcElement;
    // Only trigger the events if they're not happening in an input/textarea/select/button field
    if ( !/INPUT|TEXTAREA|SELECT|BUTTON/.test(target.nodeName) ) {{
        switch (e.keyCode) {{
            case keyCode:
                window.open('{url}', '_blank').focus();
                break;
        }}
    }}
}});
</script>
""",
        height=0,
        width=0,
    )