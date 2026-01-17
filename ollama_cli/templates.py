"""HTML export templates for Ollama CLI Pro."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Dict, List


def format_html_content(content: str) -> str:
    """Format content for HTML export with code highlighting."""
    code_pattern = r"```(\w*)?\n?(.*?)```"
    parts = []
    last_end = 0

    for match in re.finditer(code_pattern, content, re.DOTALL):
        if match.start() > last_end:
            text_part = content[last_end : match.start()]
            text_part = (
                text_part.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
            )
            text_part = re.sub(r"`([^`]+)`", r"<code>\1</code>", text_part)
            text_part = text_part.replace("\n", "<br>")
            parts.append(text_part)

        lang = match.group(1) or ""
        code = match.group(2)
        code = code.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        parts.append(
            f'<div class="code-container"><button class="copy-btn" onclick="copyCode(this)">Kopyala</button><pre><code class="language-{lang}">{code}</code></pre></div>'
        )
        last_end = match.end()

    if last_end < len(content):
        text_part = content[last_end:]
        text_part = (
            text_part.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        )
        text_part = re.sub(r"`([^`]+)`", r"<code>\1</code>", text_part)
        text_part = text_part.replace("\n", "<br>")
        parts.append(text_part)

    return "".join(parts)


def generate_html_export(
    messages: List[Dict],
    model: str,
    title: str,
    theme: Dict[str, str],
    total_tokens: int = 0,
) -> str:
    """Generate styled HTML output with syntax highlighting.

    Args:
        messages: List of chat messages
        model: Model name
        title: Chat title
        theme: Theme colors dict
        total_tokens: Total token count

    Returns:
        Complete HTML document as string
    """
    model_short = model.split(":")[0]
    date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html = f"""<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            min-height: 100vh;
            color: #e2e8f0;
            line-height: 1.6;
        }}
        .container {{
            max-width: 900px;
            margin: 0 auto;
            padding: 2rem;
        }}
        .header {{
            text-align: center;
            padding: 3rem 2rem;
            background: linear-gradient(135deg, {theme["primary"]}22, {theme["secondary"]}22);
            border-radius: 20px;
            margin-bottom: 2rem;
            border: 1px solid {theme["primary"]}44;
        }}
        .header h1 {{
            font-size: 2rem;
            background: linear-gradient(135deg, {theme["primary"]}, {theme["secondary"]});
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }}
        .header .meta {{
            color: {theme["muted"]};
            font-size: 0.9rem;
        }}
        .header .model-badge {{
            display: inline-block;
            background: {theme["primary"]}33;
            color: {theme["primary"]};
            padding: 0.3rem 1rem;
            border-radius: 20px;
            font-size: 0.85rem;
            margin-top: 1rem;
            border: 1px solid {theme["primary"]}55;
        }}
        .message {{
            margin-bottom: 1.5rem;
            animation: fadeIn 0.3s ease;
        }}
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(10px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        .message-user {{
            display: flex;
            justify-content: flex-end;
        }}
        .message-assistant {{
            display: flex;
            justify-content: flex-start;
        }}
        .bubble {{
            max-width: 80%;
            padding: 1rem 1.5rem;
            border-radius: 20px;
            position: relative;
        }}
        .bubble-user {{
            background: linear-gradient(135deg, {theme["user"]}, {theme["user"]}dd);
            border-bottom-right-radius: 5px;
            color: white;
        }}
        .bubble-assistant {{
            background: #1e293b;
            border: 1px solid #334155;
            border-bottom-left-radius: 5px;
        }}
        .role-label {{
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.5rem;
            opacity: 0.7;
        }}
        pre {{
            background: #0f172a;
            border-radius: 10px;
            padding: 1rem;
            overflow-x: auto;
            margin: 1rem 0;
            border: 1px solid #334155;
        }}
        code {{
            font-family: 'Fira Code', 'Consolas', monospace;
            font-size: 0.9rem;
        }}
        .stats {{
            display: flex;
            justify-content: center;
            gap: 2rem;
            padding: 2rem;
            background: #1e293b;
            border-radius: 15px;
            margin-top: 2rem;
            border: 1px solid #334155;
        }}
        .stat {{
            text-align: center;
        }}
        .stat-value {{
            font-size: 1.5rem;
            font-weight: bold;
            color: {theme["primary"]};
        }}
        .stat-label {{
            font-size: 0.8rem;
            color: {theme["muted"]};
            text-transform: uppercase;
        }}
        .footer {{
            text-align: center;
            padding: 2rem;
            color: {theme["muted"]};
            font-size: 0.85rem;
        }}
        .code-container {{
            position: relative;
        }}
        .copy-btn {{
            position: absolute;
            top: 8px;
            right: 8px;
            background: {theme["primary"]}44;
            border: 1px solid {theme["primary"]}66;
            color: {theme["primary"]};
            padding: 4px 8px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.75rem;
            opacity: 0;
            transition: opacity 0.2s;
        }}
        .code-container:hover .copy-btn {{
            opacity: 1;
        }}
        .copy-btn:hover {{
            background: {theme["primary"]}66;
        }}
        .hljs {{
            background: transparent !important;
        }}
    </style>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/atom-one-dark.min.css">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>\U0001f4ac {title}</h1>
            <div class="meta">{date}</div>
            <div class="model-badge">\U0001f916 {model}</div>
        </div>
        <div class="messages">
"""

    for msg in messages:
        if msg["role"] == "system":
            continue

        content = msg.get("content", "")
        if isinstance(content, list):
            content = "<em>[Gorsel icerik]</em>"
        else:
            content = format_html_content(content)

        if msg["role"] == "user":
            html += f"""
            <div class="message message-user">
                <div class="bubble bubble-user">
                    <div class="role-label">Sen</div>
                    <div class="content">{content}</div>
                </div>
            </div>
"""
        else:
            html += f"""
            <div class="message message-assistant">
                <div class="bubble bubble-assistant">
                    <div class="role-label">{model_short}</div>
                    <div class="content">{content}</div>
                </div>
            </div>
"""

    msg_count = len([m for m in messages if m["role"] != "system"])
    html += f"""
        </div>
        <div class="stats">
            <div class="stat">
                <div class="stat-value">{msg_count}</div>
                <div class="stat-label">Mesaj</div>
            </div>
            <div class="stat">
                <div class="stat-value">{total_tokens:,}</div>
                <div class="stat-label">Token</div>
            </div>
        </div>
        <div class="footer">
            <p>Ollama CLI Pro v5.1 ile olusturuldu</p>
        </div>
    </div>
    <script>
        hljs.highlightAll();

        function copyCode(btn) {{
            const code = btn.nextElementSibling.querySelector('code');
            navigator.clipboard.writeText(code.textContent).then(() => {{
                btn.textContent = 'Kopyalandi!';
                setTimeout(() => btn.textContent = 'Kopyala', 2000);
            }});
        }}
    </script>
</body>
</html>
"""
    return html
