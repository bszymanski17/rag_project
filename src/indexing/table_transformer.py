import pandas as pd

def html_to_markdown(html_content: str) -> str:
    """Converts raw HTML table into Markdown table."""
    if not html_content:
        return ""
    dfs = pd.read_html(html_content)
    return dfs[0].to_markdown(index=False) if dfs else ""