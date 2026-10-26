"""REST API page: every endpoint, in the style of the Swagger docs, built from the API's own OpenAPI spec."""

import html
import re

import streamlit as st

from app_pages import common

mobile = common.is_mobile()
common.sidebar_footer()

base = common.PUBLIC_URL or common.LOCAL_URL

# Swagger's colors for each HTTP method: (badge, row background)
METHOD_COLORS = {
    "get": ("#61affe", "#ebf3fb"),
    "post": ("#49cc90", "#e8f6f0"),
    "put": ("#fca130", "#fbf1e6"),
    "patch": ("#50e3c2", "#e5fbf6"),
    "delete": ("#f93e3e", "#fae7e7"),
    "options": ("#0d5aa7", "#e4ebf3"),
}


@st.cache_resource(show_spinner=False)
def openapi_spec():
    from api import app

    return app.openapi()


def resolve(spec, schema):
    # Follow a "$ref" to the schema it points at
    while "$ref" in schema:
        name = schema["$ref"].split("/")[-1]
        schema = spec["components"]["schemas"][name]
    return schema


def type_name(spec, schema):
    schema = resolve(spec, schema)
    if "anyOf" in schema:
        names = [type_name(spec, s) for s in schema["anyOf"] if s.get("type") != "null"]
        return " | ".join(names)
    # Uploads: OpenAPI 3.0 marks them with format "binary", 3.1 with a contentMediaType
    if schema.get("format") == "binary" or "contentMediaType" in schema:
        return "file"
    if schema.get("type") == "array":
        return f"array of {type_name(spec, schema.get('items', {}))}"
    return schema.get("type", schema.get("title", "object"))


def inline_markdown(text):
    # The little markdown the descriptions use: `code` and **bold**
    text = html.escape(text or "")
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    return re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)


def curl_example(method, path, operation):
    command = ["curl"]
    if method != "get":
        command.append(f"-X {method.upper()}")
    if common.PUBLIC_URL:
        command.append('-H "ngrok-skip-browser-warning: 1"')
    if "requestBody" in operation:
        command.append('-F "file=@photo.jpg"')
    example_path = path.replace("{model_id}", "fruits").replace("{graph}", "confusion_matrix")
    command.append(f'"{base}{example_path}"')
    return " \\\n     ".join(command)


def operation_html(spec, method, path, operation):
    badge, background = METHOD_COLORS.get(method, ("#999", "#f4f4f4"))
    parts = []

    description = operation.get("description")
    if description:
        parts.append(f'<p class="op-desc">{inline_markdown(description)}</p>')

    parameters = operation.get("parameters", [])
    if parameters:
        rows = []
        for p in parameters:
            schema = p.get("schema", {})
            default = schema.get("default")
            required = '<span class="req">* required</span>' if p.get("required") else ""
            rows.append(
                f"<tr><td><code>{p['name']}</code>{required}</td><td>{type_name(spec, schema)}"
                f'<div class="where">({p["in"]})</div></td>'
                f"<td>{inline_markdown(p.get('description') or schema.get('description', ''))}"
                + (f'<div class="where">Default: <code>{default}</code></div>' if default is not None else "")
                + "</td></tr>"
            )
        parts.append('<div class="op-section">Parameters</div><table class="op-table"><tr><th>Name</th><th>Type</th>'
                     "<th>Description</th></tr>" + "".join(rows) + "</table>")

    body = operation.get("requestBody")
    if body:
        content_type, content = next(iter(body["content"].items()))
        schema = resolve(spec, content["schema"])
        required = set(schema.get("required", []))
        rows = "".join(
            f"<tr><td><code>{name}</code>{'<span class=\"req\">* required</span>' if name in required else ''}</td>"
            f"<td>{type_name(spec, field)}</td><td>{inline_markdown(field.get('description', ''))}</td></tr>"
            for name, field in schema.get("properties", {}).items()
        )
        parts.append(f'<div class="op-section">Request body <span class="ctype">{content_type}</span></div>'
                     f'<table class="op-table"><tr><th>Field</th><th>Type</th><th>Description</th></tr>{rows}</table>')

    responses = "".join(
        f'<tr><td><span class="code code-{code[0]}">{code}</span></td><td>{inline_markdown(r.get("description", ""))}'
        + (f' <span class="ctype">{", ".join(r["content"])}</span>' if r.get("content") else "")
        + "</td></tr>"
        for code, r in operation.get("responses", {}).items()
    )
    parts.append(f'<div class="op-section">Responses</div><table class="op-table"><tr><th>Code</th>'
                 f"<th>Description</th></tr>{responses}</table>")

    parts.append(f'<div class="op-section">Example</div><pre class="op-curl">{html.escape(curl_example(method, path, operation))}</pre>')

    tag = (operation.get("tags") or ["default"])[0]
    docs_link = f"/api/docs#/{tag}/{operation.get('operationId', '')}"
    parts.append(f'<a class="op-try" href="{docs_link}" target="_blank">Try it out in the interactive docs ↗</a>')

    styled_path = re.sub(r"(\{[^}]+\})", r"<span class='param'>\1</span>", html.escape(path))
    return (
        f'<details class="op" style="--badge:{badge};--bg:{background}">'
        f'<summary><span class="method">{method.upper()}</span><span class="path">{styled_path}</span>'
        f'<span class="summary">{html.escape(operation.get("summary", ""))}</span><span class="chev"></span></summary>'
        f'<div class="op-body">{"".join(parts)}</div></details>'
    )


spec = openapi_spec()
info = spec["info"]

# Group the endpoints by tag, in the order they are first seen
groups = {}
for path, item in spec["paths"].items():
    for method, operation in item.items():
        if method not in METHOD_COLORS:
            continue
        tag = (operation.get("tags") or ["default"])[0]
        groups.setdefault(tag, []).append(operation_html(spec, method, path, operation))

oas = ".".join(spec.get("openapi", "3.1").split(".")[:2])
header = (
    f'<div class="api-head"><span class="api-title">{html.escape(info["title"])}</span>'
    f'<span class="pill pill-version">{html.escape(info["version"])}</span>'
    f'<span class="pill pill-oas">OAS {oas}</span></div>'
    f'<a class="api-spec" href="{html.escape(base)}/api/openapi.json" target="_blank">/api/openapi.json</a>'
    f'<div class="api-base">Base address: <code>{html.escape(base)}</code>'
    + (" &nbsp;·&nbsp; from code, add the header <code>ngrok-skip-browser-warning: 1</code>" if common.PUBLIC_URL else "")
    + "</div>"
)
sections = "".join(
    f'<details class="tag" open><summary><span class="tag-name">{html.escape(tag)}</span><span class="chev"></span></summary>'
    f'<div class="tag-body">{"".join(ops)}</div></details>'
    for tag, ops in groups.items()
)

st.markdown(
    """<style>
.api-head { display: flex; align-items: center; flex-wrap: wrap; gap: 8px; margin-top: 4px; }
.api-title { font-size: 2.1rem; font-weight: 750; color: #3b4151; margin-right: 4px; line-height: 1.2; }
.pill { color: #fff; border-radius: 57px; padding: 2px 9px; font-size: .78rem; font-weight: 700; }
.pill-version { background: #7d8492; }
.pill-oas { background: #89bf04; }
.api-spec { font-size: .85rem; color: #4990e2 !important; text-decoration: none; }
.api-base { margin: 14px 0 8px; color: #3b4151; font-size: .92rem; }
.api-base code, .op-table code, .op-desc code { background: #eceafb; color: #4c3fd1; padding: 1px 6px; border-radius: 6px; }
details.tag { margin-top: 26px; }
details.tag > summary { display: flex; align-items: center; justify-content: space-between; cursor: pointer;
    list-style: none; border-bottom: 1px solid #d8dde7; padding: 6px 10px 10px; }
details.tag > summary::-webkit-details-marker, details.op > summary::-webkit-details-marker { display: none; }
.tag-name { font-size: 1.5rem; font-weight: 700; color: #3b4151; }
.chev { width: 11px; height: 11px; border-right: 2px solid #3b4151; border-bottom: 2px solid #3b4151;
    transform: rotate(45deg); transition: transform .15s; margin: 0 8px 4px auto; flex: none; }
details[open] > summary > .chev { transform: rotate(-135deg); margin-bottom: -4px; }
.tag-body { padding-top: 6px; }
details.op { border: 1px solid var(--badge); background: var(--bg); border-radius: 4px; margin: 10px 0;
    box-shadow: 0 0 3px rgba(0,0,0,.12); }
details.op > summary { display: flex; align-items: center; gap: 10px; padding: 5px; cursor: pointer; list-style: none;
    flex-wrap: wrap; }
.method { background: var(--badge); color: #fff; font-weight: 700; font-size: .9rem; border-radius: 3px;
    min-width: 80px; text-align: center; padding: 6px 0; text-shadow: 0 1px 0 rgba(0,0,0,.1); flex: none; }
.path { font-family: 'Source Code Pro', Consolas, monospace; font-weight: 600; font-size: 1rem; color: #3b4151;
    word-break: break-all; }
.path .param { color: #3b4151; }
.summary { color: #3b4151; font-size: .9rem; }
details.op > summary .chev { border-color: #3b4151; }
.op-body { background: #fff; border-top: 1px solid var(--badge); padding: 14px 18px 16px; border-radius: 0 0 4px 4px; }
.op-desc { margin: 0 0 6px; color: #3b4151; }
.op-section { font-weight: 700; color: #3b4151; margin: 14px 0 6px; padding-bottom: 4px;
    border-bottom: 1px solid #e8e8ee; font-size: .95rem; }
.op-table { width: 100%; border-collapse: collapse; font-size: .88rem; color: #3b4151; }
.op-table th { text-align: left; font-weight: 600; padding: 6px 8px; border-bottom: 1px solid #e8e8ee; }
.op-table td { padding: 7px 8px; vertical-align: top; border-bottom: 1px solid #f1f1f5; }
.req { color: #f93e3e; font-size: .72rem; margin-left: 6px; white-space: nowrap; display: inline-block; }
.where { color: #8a8fa0; font-size: .78rem; font-style: italic; }
.ctype { background: #41444e; color: #fff; font-size: .72rem; border-radius: 4px; padding: 1px 6px; margin-left: 6px;
    font-weight: 600; }
.code { font-weight: 700; }
.code-2 { color: #49cc90; } .code-4 { color: #f93e3e; } .code-5 { color: #fca130; }
.op-curl { background: #333; color: #fff; border-radius: 4px; padding: 10px 12px; font-size: .82rem;
    white-space: pre-wrap; word-break: break-all; margin: 0; }
.op-try { display: inline-block; margin-top: 12px; padding: 6px 14px; border: 2px solid var(--badge);
    border-radius: 4px; color: #3b4151 !important; font-weight: 700; font-size: .85rem; text-decoration: none; }
@media (max-width: 640px) {
    .api-title { font-size: 1.6rem; }
    .method { min-width: 64px; font-size: .8rem; padding: 5px 0; }
    .path { font-size: .88rem; flex: 1; }
    .summary { flex-basis: 100%; padding: 0 2px 2px; font-size: .85rem; }
    details.op > summary .chev { display: none; }
    .op-body { padding: 10px 10px 12px; }
    .op-table th:nth-child(2), .op-table td:nth-child(2) { white-space: nowrap; }
}
</style>""",
    unsafe_allow_html=True,
)
st.html(header + sections)
