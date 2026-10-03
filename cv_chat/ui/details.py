"""The Details dropdown under each answer: how the question was routed, what was searched and used, and where the time went."""
import streamlit as st

ROUTE_NAMES = {"chat": "Chat message", "simple": "Simple question", "complex": "Complex question"}
STAGE_NAMES = {
    "route": "Router", "expand": "Query expansion", "search": "Search", "agent_model": "Agent (model)", "tool": "Agent (tool)",
    "model": "Model (starting)", "cache": "Cache", "filter": "Safety filter", "generate": "Write answer",
}


def fmt(ms: float | None) -> str:
    if ms is None:
        return "n/a"
    return f"{ms:.0f} ms" if ms < 1000 else f"{ms / 1000:.1f} s"


def render(trace: dict | None, sources: list[dict]) -> None:
    if not trace:
        return
    events = trace["events"]
    route = ROUTE_NAMES.get(trace["route"], trace["route"] or "Unknown")
    searches = [e for e in events if e["stage"] == "search"]
    with st.expander(f"Details · {route} · {fmt(trace['total_ms'])}", icon=":material/analytics:"):
        columns = st.columns(4)
        columns[0].metric("Route", route)
        columns[1].metric("Total", fmt(trace["total_ms"]), help="From sending the question to the last word of the answer")
        columns[2].metric("First word", fmt(trace["first_token_ms"]), help="How long until the answer started to appear")
        columns[3].metric("Searches", len(searches), help="Searches run on Azure AI Search (cached ones included)")

        tabs = ["Timeline", "Router", "Searches"]
        has_agent = any(e["stage"] in ("agent_model", "tool") for e in events)
        if has_agent:
            tabs.append("Agent")
        tabs += ["Excerpts used", "Settings"]
        tab = dict(zip(tabs, st.tabs(tabs)))
        with tab["Timeline"]:
            _timeline(events, trace["total_ms"])
        with tab["Router"]:
            _router(events, route)
        with tab["Searches"]:
            _searches(events, trace["settings"]["query_expansion"])
        if has_agent:
            with tab["Agent"]:
                _agent(events)
        with tab["Excerpts used"]:
            _excerpts(sources)
        with tab["Settings"]:
            _settings(trace)


def _detail(event: dict) -> str:
    info, stage = event["info"], event["stage"]
    if stage == "route":
        return f"{event['label']} · \"{info.get('query', '')}\"" + (" · cached" if info.get("cached") else "") + (" · failed, used the question as it is" if info.get("failed") else "")
    if stage == "search":
        return f"\"{event['label']}\" · {info['results']} chunks" + (" · cached" if info["cached"] else "")
    if stage == "agent_model":
        return f"{event['label']}: {info.get('decision') or info.get('error', '')}"
    if stage == "tool":
        return event["label"] + (f" · {info['cv']}" if info.get("cv") else "")
    if stage == "expand":
        return f"{len(info['queries'])} extra queries" if info["queries"] else "no extra queries (call failed)" if info.get("failed") else "no extra queries"
    if stage == "filter":
        return "left out: " + ", ".join(info["left_out"])
    if stage == "model":
        return f"{event['label']} · waiting for its first word"
    if stage == "generate":
        return f"{event['label']} · writing the answer"
    return event["label"]


def _timeline(events: list[dict], total: float | None) -> None:
    if not events:
        st.caption("Nothing was recorded.")
        return
    rows = [
        {"Step": STAGE_NAMES.get(e["stage"], e["stage"]), "Detail": _detail(e), "Time": e["ms"], "Share": min(e["ms"] / total, 1) if total else 0}
        for e in events
    ]
    st.dataframe(
        rows, hide_index=True, width="stretch",
        column_config={
            "Time": st.column_config.NumberColumn("Time", format="%d ms"),
            "Share": st.column_config.ProgressColumn("Share of total", min_value=0, max_value=1, format="percent"),
        },
    )
    st.caption("Steps that run in parallel (query expansion searches) overlap, so shares can add up to more than 100%.")


def _router(events: list[dict], route: str) -> None:
    router = next((e for e in events if e["stage"] == "route"), None)
    if not router:
        st.caption("No router result was recorded.")
        return
    info = router["info"]
    st.markdown(f"**Classified as:** {route}  \n**Standalone question:** {info.get('query', '')}  \n"
                f"**Sections to search:** {', '.join(info['sections']) if info.get('sections') else 'all'}  \n"
                f"**Router call:** {fmt(router['ms'])}" + (" (cached)" if info.get("cached") else ""))
    if info.get("failed"):
        st.warning("The router call failed, so the question was searched as written.", icon=":material/warning:")
    st.caption("chat: no search · simple: one search · complex: handled by the search agent")


def _searches(events: list[dict], expansion: bool) -> None:
    expand = next((e for e in events if e["stage"] == "expand"), None)
    if expand and expand["info"]["queries"]:
        st.markdown("**Reworded queries** (" + fmt(expand["ms"]) + "): " + " · ".join(f"`{q}`" for q in expand["info"]["queries"]))
    elif expansion:
        st.caption("Query expansion was on but produced no extra queries.")
    searches = [e for e in events if e["stage"] == "search"]
    if not searches:
        st.caption("No search was needed for this message.")
    for event in searches:
        info = event["info"]
        parts = [f"{info['results']} chunks", fmt(event["ms"])]
        if info["cached"]:
            parts.append("cached")
        else:
            parts.append(f"embedding {fmt(info['embed_ms'])}")
        if info["sections"]:
            parts.append("sections: " + ", ".join(info["sections"]) + (" (no hits, searched everything)" if info["fallback"] else ""))
        if info["cvs"]:
            parts.append(f"limited to {info['cvs']} CV{'' if info['cvs'] == 1 else 's'}")
        if info["filter"]:
            parts.append(f"filter: `{info['filter']}`")
        st.markdown(f"- `{event['label']}` · " + " · ".join(parts))


def _agent(events: list[dict]) -> None:
    for event in events:
        if event["stage"] == "agent_model":
            st.markdown(f"- **{event['label']}** · {fmt(event['ms'])} · " + (event["info"].get("decision") or event["info"].get("error", "")))
        elif event["stage"] == "tool":
            st.markdown(f"  - tool `{event['label']}`" + (f" on {event['info']['cv']}" if event["info"].get("cv") else "") + f" · {fmt(event['ms'])}")
        elif event["stage"] == "search":
            st.markdown(f"  - search `{event['label']}` · {event['info']['results']} chunks · {fmt(event['ms'])}" + (" · cached" if event["info"]["cached"] else ""))


def _excerpts(sources: list[dict]) -> None:
    if not sources:
        st.caption("No CV excerpts were used.")
        return
    files = {s["file_name"] for s in sources}
    st.caption(f"{len(sources)} excerpt{'' if len(sources) == 1 else 's'} from {len(files)} CV{'' if len(files) == 1 else 's'} went to the model.")
    for s in sources:
        page = f" · p.{s['page']}" if s.get("page") else ""
        who = f" · {s['candidate_name']}" if s.get("candidate_name") else ""
        st.markdown(f"- **{s['file_name']}**{who} · {s['section']}{page}")


def _settings(trace: dict) -> None:
    settings, models = trace["settings"], trace["models"]
    st.markdown(
        f"- Chat model deployment: `{models['chat']}`\n- Embedding deployment: `{models['embedding']}`\n"
        f"- Query expansion: {'on' if settings['query_expansion'] else 'off'}\n"
        f"- Answer cache: {'on' if settings['answer_cache'] else 'off'}"
    )
