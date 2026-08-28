#!/usr/bin/env python3
"""v6 Task 6 probe set, qwen36 arm.

Covers plan items 1, 2, 3 and 7. Items 4 (Kimi canary) and 5 (Harmony) need
model swaps and run separately.

Each phase prints a MARKER line so the server log can be sliced per phase
without relying on timestamps.

Item 2 is extended beyond the staged probe with a NESTED-ARGUMENT tool call.
Task 3's deferred minors record "NO settled-path coverage of nested-object/array
tool arguments through tojson" as a Task 6 candidate — the staged probe only
exercised a flat integer, so nested is the gap.
"""
import json
import os
import sys
import time
import urllib.request

URL = "http://127.0.0.1:8081/v1/chat/completions"

# Set PROBE_MODEL to reuse this set for the Kimi canary and the Harmony arm.
# Item 3 (max_tokens cut mid-thinking) is a no-op on Kimi, which has no thinking
# channel — it runs and simply produces an ordinary truncated turn. Harmless,
# and worth leaving in so the three models run an identical script.
MODEL = os.environ.get("PROBE_MODEL", "qwen36")

# Read-think-type gap between turns. The settle runs BETWEEN requests, so a
# back-to-back probe measures the worst case by construction: every settle is
# racing the next request. Item 6 asks whether real think time absorbs it, and
# that is unanswerable at 0. Set PROBE_THINK_S to simulate a reading user.
THINK_S = float(os.environ.get("PROBE_THINK_S", "0"))
_sent = [0]

FLAT_TOOL = {
    "type": "function",
    "function": {
        "name": "get_measurement",
        "description": "Return the recorded tide measurement for a named site.",
        "parameters": {
            "type": "object",
            "properties": {
                "site": {"type": "string", "description": "The site name."},
                "precision": {"type": "integer", "description": "Decimal places."},
            },
            "required": ["site", "precision"],
        },
    },
}

NESTED_TOOL = {
    "type": "function",
    "function": {
        "name": "plan_survey",
        "description": "Schedule a survey across sites with per-site options.",
        "parameters": {
            "type": "object",
            "properties": {
                "window": {
                    "type": "object",
                    "description": "Start and end of the survey window.",
                    "properties": {
                        "start": {"type": "string"},
                        "end": {"type": "string"},
                    },
                },
                "sites": {
                    "type": "array",
                    "description": "Site names to include.",
                    "items": {"type": "string"},
                },
                "depths": {
                    "type": "array",
                    "description": "Depths in metres.",
                    "items": {"type": "number"},
                },
            },
            "required": ["window", "sites", "depths"],
        },
    },
}


def post(messages, tools=None, max_tokens=900):
    if THINK_S and _sent[0]:
        time.sleep(THINK_S)
    _sent[0] += 1
    body = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    if tools:
        body["tools"] = tools
    req = urllib.request.Request(
        URL, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=3600) as r:
        resp = json.load(r)
    return resp, time.time() - t0


def marker(text):
    """Print-only. Deliberately does NOT send a request: a marker request would
    create its own cache entry and add eviction pressure to the very cache
    under test. Slice the server log by these UTC stamps instead."""
    print(f"\n===== {text} @ {time.strftime('%H:%M:%SZ', time.gmtime())} =====",
          flush=True)


def show(label, resp, wall):
    ch = resp["choices"][0]
    msg = ch["message"]
    u = resp.get("usage", {})
    calls = msg.get("tool_calls") or []
    print(f"{label}: wall={wall:.1f}s finish={ch.get('finish_reason')} "
          f"prompt={u.get('prompt_tokens')} completion={u.get('completion_tokens')} "
          f"reasoning_chars={len(msg.get('reasoning_content') or '')} calls={len(calls)}",
          flush=True)
    for c in calls:
        print(f"    args={c['function']['arguments']!r}", flush=True)
    return msg, calls


def assistant_from(msg, calls):
    a = {"role": "assistant", "content": msg.get("content") or ""}
    if msg.get("reasoning_content"):
        a["reasoning_content"] = msg["reasoning_content"]
    if calls:
        a["tool_calls"] = calls
    return a


# ---- item 1: plain multi-turn, two independent runs -------------------------
def plain(run_id):
    turns = [
        "Name one country in South America. One short sentence.",
        "Name a different one. One short sentence.",
        "And a third. One short sentence.",
    ]
    msgs = []
    for i, t in enumerate(turns):
        msgs.append({"role": "user", "content": f"[run{run_id}] {t}"})
        resp, wall = post(msgs)
        msg, calls = show(f"plain{run_id}-turn{i+1}", resp, wall)
        msgs.append(assistant_from(msg, calls))


# ---- item 2: tool loop, flat then nested arguments --------------------------
def tool_loop(tool, label, ask, result):
    msgs = [{"role": "user", "content": ask}]
    resp, wall = post(msgs, tools=[tool])
    msg, calls = show(f"{label}-turn1(call)", resp, wall)
    if not calls:
        print(f"{label}: NO TOOL CALL — loop not exercised", flush=True)
        return
    msgs.append(assistant_from(msg, calls))
    msgs.append({"role": "tool", "tool_call_id": calls[0]["id"], "content": result})
    resp, wall = post(msgs, tools=[tool])
    msg, calls2 = show(f"{label}-turn2(midloop)", resp, wall)
    msgs.append(assistant_from(msg, calls2))
    msgs.append({"role": "user", "content": "Thanks. Restate that in one sentence."})
    resp, wall = post(msgs, tools=[tool])
    show(f"{label}-turn3(postloop)", resp, wall)
    return msgs


# ---- item 3: degenerate, max_tokens cut mid-thinking ------------------------
def degenerate():
    msgs = [{"role": "user",
             "content": "Explain in detail why the sky is blue, reasoning carefully first."}]
    resp, wall = post(msgs, max_tokens=200)
    msg, calls = show("degenerate-turn1(cut)", resp, wall)
    msgs.append(assistant_from(msg, calls))
    msgs.append({"role": "user", "content": "Never mind. Name a colour."})
    resp, wall = post(msgs)
    show("degenerate-turn2(after-cut)", resp, wall)


# ---- item 7: arbitration, join vs abort -------------------------------------
def arbitration(loop_msgs, tool):
    """Fire immediately after a completed turn so the settle is still running.

    ⚠ REQUIRES a pending settle to arbitrate against. On a runner where
    supportsPartialRewind is false (GDN or ring-backed layers) no settle ever
    starts, so this probe cannot produce a join/abort verdict — it will run and
    tell you nothing. Check for `normalize kind=settle` in the log first.

    ⚠ `tools` MUST be passed. Omitting it makes these different conversations
    from the cache's point of view (the entry keys on tools), not continuations
    — which is what invalidated the 2026-08-27 19:45Z run.
    """
    if not loop_msgs:
        print("arbitration: skipped, no loop context", flush=True)
        return
    join = list(loop_msgs) + [{"role": "user", "content": "One more word on that."}]
    resp, wall = post(join, tools=[tool])
    show("arbitrate-JOIN(continuation)", resp, wall)

    edited = [dict(m) for m in loop_msgs]
    for m in edited:
        if m["role"] == "user":
            m["content"] = m["content"] + " (edited)"
            break
    edited.append({"role": "user", "content": "And now?"})
    resp, wall = post(edited, tools=[tool])
    show("arbitrate-ABORT(edited-history)", resp, wall)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "plain"):
        marker("ITEM1-PLAIN-RUN-A"); plain("A")
        marker("ITEM1-PLAIN-RUN-B"); plain("B")
    if which in ("all", "tools"):
        marker("ITEM2-TOOLLOOP-FLAT")
        tool_loop(FLAT_TOOL, "flat",
                  'Use get_measurement for site "harbour" with precision 2, then state the value.',
                  "1.83")
        marker("ITEM2-TOOLLOOP-NESTED")
        nested_msgs = tool_loop(
            NESTED_TOOL, "nested",
            'Use plan_survey for sites "harbour" and "quarry", depths 1.5 and 3.25, '
            'window start "2026-09-01" end "2026-09-07", then confirm the plan.',
            '{"scheduled": true, "id": "SV-11"}')
    # Harmony/gpt-oss cannot run the NESTED loop (server 500s:
    # structured_output_failure, decoded_calls=0 — the model emits no call for
    # that schema), so item 7 has no nested context there. Arbitration still
    # REQUIRES tools — omitting them keys different entries and recreates the
    # 19:45Z invalidation — so run it against FLAT instead.
    if which == "flat-arb":
        marker("ITEM2-TOOLLOOP-FLAT")
        flat_msgs = tool_loop(FLAT_TOOL, "flat",
                              'Use get_measurement for site "harbour" with precision 2, '
                              'then state the value.', "1.83")
        marker("ITEM7-ARBITRATION")
        arbitration(flat_msgs, FLAT_TOOL)
    if which in ("all", "degenerate"):
        marker("ITEM3-DEGENERATE"); degenerate()
    if which in ("all", "arbitration"):
        marker("ITEM7-ARBITRATION")
        arbitration(nested_msgs if which == "all" else None, NESTED_TOOL)
    print("\nprobe set done", flush=True)
