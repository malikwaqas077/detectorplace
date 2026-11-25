"""Design assistant: ask questions about the proposed layout in plain English.

"Why is FD-02 here?"  "What if FD-02 fails?"  "Can I move FD-02 2 metres left?"

Every answer comes from re-running the real coverage calculations on the plant model; nothing is
made up. If an answer depends on part of the drawing that was read with low confidence, the
answer says so.

Two front ends share the same calculation functions (the "tools"):
  * a built-in question parser that works offline with no API key (used for live demos), and
  * Claude (claude-opus-5-5) with tool use, used when an Anthropic API key is configured. Claude turns
    a free-form question into tool calls and puts the results into words.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import replace

import numpy as np

from .coverage import flame_matrix, gas_matrix
from .detectors import FlameModel, GasModel
from .evaluate import Design, coverage, counts
from .plant import RES

LEAK_RADIUS_M = 1.5      # a leak point is "watched" if a detector covers a target within this distance
MODEL = "claude-opus-5-5"


class DesignAssistant:
    def __init__(self, ctx, conf_items=None, conf_threshold=0.85):
        self.ctx = ctx
        self.p, self.plant, self.cand = ctx["problem"], ctx["plant"], ctx["cand"]
        self.design = ctx["opt"]["chosen"]["design"]
        self.rows = ctx["explain"]["schedule"]
        self.by_tag = {r["tag"]: r for r in self.rows}
        self.conf_items = conf_items or []
        self.conf_threshold = conf_threshold
        self.cf, self.cg = counts(self.p, self.design)
        self.base_cov = coverage(self.p, self.design)
        # Which targets sit next to each leak point (for "which leaks go unwatched").
        self._leak_fire = [np.hypot(*(self.p.fire.xy - xy).T) <= LEAK_RADIUS_M for xy in self.plant.leaks]
        self._leak_gas = [np.hypot(*(self.p.gas.xy - xy).T) <= LEAK_RADIUS_M for xy in self.plant.leaks]

    # ------------------------------------------------------------------ helpers
    def _row_vec(self, r):
        return (self.p.Mf if r["type"].startswith("Flame") else self.p.Mg)[r["cand"]].astype(np.int16)

    def _watched(self, cf, cg):
        """Leak points watched by at least one detector, given per-target detector counts."""
        out = []
        for i in range(len(self.plant.leaks)):
            f, g = self._leak_fire[i], self._leak_gas[i]
            if (f.any() and (cf[f] > 0).any()) or (g.any() and (cg[g] > 0).any()):
                out.append(f"L{i + 1}")
        return out

    def _leaks_seen_by(self, r):
        v = self._row_vec(r) > 0
        masks = self._leak_fire if r["type"].startswith("Flame") else self._leak_gas
        return [f"L{i + 1}" for i, m in enumerate(masks) if m.any() and v[m].any()]

    def _cov_rows(self, cov):
        return [{"target": q.label, "required": f"{q.target:.0%}", "before": f"{self.base_cov[q.key]:.0%}",
                 "after": f"{cov[q.key]:.0%}", "met": bool(cov[q.key] >= q.target)} for q in self.p.reqs]

    def _low_conf_for(self, tag):
        return [f"{c['tag']} ({c['type']}, {c['confidence']:.0%})" for c in self.conf_items
                if c["confidence"] < self.conf_threshold and tag in c.get("relies", [])]

    def _resolve(self, tag):
        if not tag:
            return None
        t = tag.upper().replace(" ", "").replace("_", "-")
        m = re.fullmatch(r"(FD|GD|D|F|G)?-?0*(\d+)", t)
        if not m:
            return None
        kind, n = m.group(1) or "D", int(m.group(2))
        for pre in (["FD"] if kind in ("FD", "F") else ["GD"] if kind in ("GD", "G") else ["FD", "GD"]):
            if f"{pre}-{n:02d}" in self.by_tag:
                return self.by_tag[f"{pre}-{n:02d}"]
        return None

    # ------------------------------------------------------------------ tools
    def list_detectors(self):
        return {"detectors": [{"tag": r["tag"], "type": r["type"], "x_m": round(r["x"], 1), "y_m": round(r["y"], 1),
                               "mount_m": r["mount_m"]} for r in self.rows]}

    def design_summary(self):
        fixed = self.ctx["catalogue"]["cost"]["fixed"]
        chosen = self.ctx["opt"]["chosen"]
        return {"flame_detectors": len(self.design.flame), "gas_detectors": len(self.design.gas),
                "installed_cost_gbp": round(chosen["cost"] + fixed),
                "saving_vs_rule_of_thumb_gbp": round(self.ctx["opt"]["greedy"]["cost"] - chosen["cost"]),
                "coverage": [{"target": q.label, "required": f"{q.target:.0%}", "achieved": f"{self.base_cov[q.key]:.0%}",
                              "met": bool(self.base_cov[q.key] >= q.target)} for q in self.p.reqs],
                "leak_points_watched": f"{len(self._watched(self.cf, self.cg))} of {len(self.plant.leaks)}"}

    def explain_detector(self, tag):
        r = self._resolve(tag)
        if r is None:
            return {"error": f"No detector called {tag}. Detectors are " + ", ".join(self.by_tag)}
        v = self._row_vec(r) > 0
        cnt = self.cf if r["type"].startswith("Flame") else self.cg
        masks = self._leak_fire if r["type"].startswith("Flame") else self._leak_gas
        only = [f"L{i + 1}" for i, m in enumerate(masks) if m.any() and v[m].any() and not (cnt[m][v[m]] > 1).any()]
        return {"tag": r["tag"], "type": r["type"], "position_m": [round(r["x"], 1), round(r["y"], 1)],
                "mounted_at_m": r["mount_m"],
                "facing_deg": None if r["azimuth_deg"] is None else round(r["azimuth_deg"]),
                "area_seen_m2": r["points_seen"], "area_only_it_sees_m2": r["unique_points"],
                "leak_points_it_watches": self._leaks_seen_by(r), "leak_points_only_it_watches": only,
                "reason": r["why"], "installed_cost_gbp": round(r["cost"]),
                "depends_on_low_confidence_drawing_items": self._low_conf_for(r["tag"])}

    def detector_fails(self, tag):
        r = self._resolve(tag)
        if r is None:
            return {"error": f"No detector called {tag}."}
        v = self._row_vec(r)
        cf, cg = (self.cf - v, self.cg) if r["type"].startswith("Flame") else (self.cf, self.cg - v)
        cov = coverage(self.p, self.design, cf, cg)
        before, after = set(self._watched(self.cf, self.cg)), set(self._watched(cf, cg))
        lost = sorted(before - after, key=lambda s: int(s[1:]))
        return {"tag": r["tag"], "coverage_if_it_fails": self._cov_rows(cov),
                "targets_broken": [q.label for q in self.p.reqs if cov[q.key] < q.target],
                "leak_points_left_unwatched": lost,
                "depends_on_low_confidence_drawing_items": self._low_conf_for(r["tag"])}

    def move_detector(self, tag, dx_m=0.0, dy_m=0.0):
        """dx_m > 0 moves right (east), dy_m > 0 moves down the drawing (south)."""
        r = self._resolve(tag)
        if r is None:
            return {"error": f"No detector called {tag}."}
        nx, ny = r["x"] + float(dx_m), r["y"] + float(dy_m)
        where = f"({nx:.1f} m, {ny:.1f} m)"
        if not (0.5 <= nx <= self.plant.width - 0.5 and 0.5 <= ny <= self.plant.depth - 0.5):
            return {"tag": r["tag"], "new_position_m": [nx, ny], "possible": False,
                    "reason": f"{where} is outside the plant boundary."}
        iy, ix = int(ny / RES), int(nx / RES)
        if self.plant.heights[iy, ix] > 0:
            return {"tag": r["tag"], "new_position_m": [nx, ny], "possible": False,
                    "reason": f"{where} is inside a piece of equipment ({self.plant.heights[iy, ix]:.0f} m tall)."}
        k = r["cand"]
        old = self._row_vec(r)
        if r["type"].startswith("Flame"):
            one = replace(self.cand, f_x=np.array([nx]), f_y=np.array([ny]), f_z=self.cand.f_z[[k]],
                          f_az=self.cand.f_az[[k]], f_tilt=self.cand.f_tilt[[k]], f_cost=self.cand.f_cost[[k]])
            new = flame_matrix(self.plant, one, self.p.fire, FlameModel())[0].astype(np.int16)
            cf, cg = self.cf - old + new, self.cg
        else:
            one = replace(self.cand, g_xy=np.array([[nx, ny]]), g_cost=self.cand.g_cost[[k]])
            new = gas_matrix(self.plant, one, self.p.gas, GasModel())[0].astype(np.int16)
            cf, cg = self.cf, self.cg - old + new
        cov = coverage(self.p, self.design, cf, cg)
        before, after = set(self._watched(self.cf, self.cg)), set(self._watched(cf, cg))
        broken = [q.label for q in self.p.reqs if cov[q.key] < q.target]
        return {"tag": r["tag"], "moved_by_m": [dx_m, dy_m], "new_position_m": [round(nx, 1), round(ny, 1)],
                "possible": True, "area_seen_before_m2": int((old > 0).sum()), "area_seen_after_m2": int((new > 0).sum()),
                "coverage_after_move": self._cov_rows(cov), "targets_broken": broken,
                "leak_points_lost": sorted(before - after, key=lambda s: int(s[1:])),
                "leak_points_gained": sorted(after - before, key=lambda s: int(s[1:])),
                "recommendation": "keep it where it is" if broken or (before - after) else "the move is acceptable"}

    def low_confidence_items(self):
        low = [c for c in self.conf_items if c["confidence"] < self.conf_threshold]
        return {"threshold": f"{self.conf_threshold:.0%}",
                "items": [{"item": c["tag"], "what": c["type"], "confidence": f"{c['confidence']:.0%}",
                           "detectors_relying_on_it": c.get("relies", [])} for c in low]}

    TOOLS = [
        {"name": "design_summary", "description": "Overall result: detector counts, installed cost, saving, coverage "
         "against every target and how many leak points are watched.",
         "input_schema": {"type": "object", "properties": {}, "additionalProperties": False}},
        {"name": "list_detectors", "description": "All detectors in the layout with tag, type and position.",
         "input_schema": {"type": "object", "properties": {}, "additionalProperties": False}},
        {"name": "explain_detector", "description": "Why one detector is placed where it is: what it sees, which "
         "leak points only it watches, its cost, and any low-confidence drawing items it depends on.",
         "input_schema": {"type": "object", "properties": {"tag": {"type": "string", "description": "e.g. FD-02 or GD-01"}},
                          "required": ["tag"], "additionalProperties": False}},
        {"name": "detector_fails", "description": "Re-run coverage with one detector out of service. Returns coverage "
         "before and after for every target, targets broken and leak points left unwatched.",
         "input_schema": {"type": "object", "properties": {"tag": {"type": "string"}}, "required": ["tag"],
                          "additionalProperties": False}},
        {"name": "move_detector", "description": "Re-run coverage with one detector moved. dx_m > 0 is right/east, "
         "dx_m < 0 is left/west, dy_m > 0 is down/south on the drawing, dy_m < 0 is up/north.",
         "input_schema": {"type": "object", "properties": {"tag": {"type": "string"}, "dx_m": {"type": "number"},
                                                           "dy_m": {"type": "number"}},
                          "required": ["tag", "dx_m", "dy_m"], "additionalProperties": False}},
        {"name": "low_confidence_items", "description": "Drawing items the AI read with low confidence, and the "
         "detectors that rely on them. An engineer should check these.",
         "input_schema": {"type": "object", "properties": {}, "additionalProperties": False}},
    ]

    def call(self, name, args):
        fn = {"design_summary": self.design_summary, "list_detectors": self.list_detectors,
              "explain_detector": self.explain_detector, "detector_fails": self.detector_fails,
              "move_detector": self.move_detector, "low_confidence_items": self.low_confidence_items}.get(name)
        if fn is None:
            return {"error": f"Unknown tool {name}"}
        try:
            return fn(**(args or {}))
        except TypeError as e:
            return {"error": f"Bad arguments for {name}: {e}"}

    # ------------------------------------------------------------------ built-in (offline) front end
    _DIRS = {"left": (-1, 0), "west": (-1, 0), "right": (1, 0), "east": (1, 0),
             "up": (0, -1), "north": (0, -1), "down": (0, 1), "south": (0, 1)}

    HELP = ("I can answer questions about this layout, for example: *Why is FD-02 here?*, "
            "*What if FD-02 fails?*, *Can I move FD-02 2 m left?*, *What does it cost?*, "
            "*Which items should an engineer check?*")

    def route(self, question, selected=None):
        """Understand a question with simple rules and run the matching calculation.
        Returns (tool_name, result), or (None, None) if the question isn't understood."""
        q = question.lower()
        m = re.search(r"\b(fd|gd|d|f|g)[\s-]?0*(\d{1,2})\b", q)
        tag = f"{m.group(1).upper()}-{int(m.group(2)):02d}" if m else selected
        mv = re.search(r"(\d+(?:\.\d+)?)\s*(?:m|metres?|meters?)?\s*(?:to the\s+)?(left|right|up|down|north|south|east|west)", q)
        if mv and re.search(r"move|shift|put|relocat", q):
            d = float(mv.group(1))
            ux, uy = self._DIRS[mv.group(2)]
            return "move_detector", self.move_detector(tag, ux * d, uy * d)
        if re.search(r"fail|lose|lost|broken|out of service|offline|remove|maintenance|goes down|stops|without", q):
            return "detector_fails", self.detector_fails(tag)
        if re.search(r"confiden|unsure|uncertain|check|trust|misread", q):
            return "low_confidence_items", self.low_confidence_items()
        if re.search(r"why|explain|purpose|what does|tell me about|reason|important", q) and tag:
            return "explain_detector", self.explain_detector(tag)
        if re.search(r"cost|price|cheap|saving|how much|coverage|target|summary|overall|how many", q):
            return "design_summary", self.design_summary()
        if re.search(r"list|which detectors|all detectors", q):
            return "list_detectors", self.list_detectors()
        if tag:
            return "explain_detector", self.explain_detector(tag)
        return None, None

    def answer_offline(self, question, selected=None, level="technical"):
        name, res = self.route(question, selected)
        return self.HELP if name is None else self.describe(name, res, level)

    @staticmethod
    def _conf_note(res):
        low = res.get("depends_on_low_confidence_drawing_items")
        return ("\n\n⚠️ **Check first:** this answer depends on drawing items read with low confidence: "
                + ", ".join(low) + ". An engineer should confirm them on the drawing.") if low else ""

    @staticmethod
    def _join(items):
        items = list(items)
        return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]

    def describe_simple(self, name, res):
        note = ""
        low = res.get("depends_on_low_confidence_drawing_items")
        if low:
            note = ("\n\n⚠️ This depends on part of the drawing the AI wasn't fully sure about ("
                    + self._join(x.split(" (")[0] for x in low) + "). An engineer should check it first.")
        if name == "explain_detector":
            if res["leak_points_only_it_watches"]:
                s = (f"**{res['tag']}** is the only detector watching leak point(s) "
                     f"**{self._join(res['leak_points_only_it_watches'])}**. Without it they would be unwatched.")
            elif res["area_only_it_sees_m2"] > 0:
                s = (f"**{res['tag']}** covers **{res['area_only_it_sees_m2']} m²** of the danger area that no other "
                     "detector can see. Without it there would be a blind spot.")
            else:
                s = (f"**{res['tag']}** is a back-up. Everything it sees is also seen by another detector, so two "
                     "detectors can agree before the alarm goes off.")
            if res["leak_points_it_watches"] and not res["leak_points_only_it_watches"]:
                s += f" It also watches leak point(s) {self._join(res['leak_points_it_watches'])}."
            return s + note
        if name == "detector_fails":
            broken, lost = res["targets_broken"], res["leak_points_left_unwatched"]
            if not broken and not lost:
                return (f"If **{res['tag']}** stops working, **every safety target is still met**. The other detectors "
                        "cover for it." + note)
            parts = []
            if broken:
                parts.append(f"**{len(broken)} of {len(res['coverage_if_it_fails'])} safety targets** would be missed")
            if lost:
                parts.append(f"leak point(s) **{self._join(lost)}** would go unwatched")
            return f"If **{res['tag']}** stops working, " + " and ".join(parts) + ". So it is an important detector." + note
        if name == "move_detector":
            dx, dy = res.get("moved_by_m", [0, 0])
            how = " and ".join(t for t in (f"{abs(dx):g} m {'right' if dx > 0 else 'left'}" if dx else "",
                                            f"{abs(dy):g} m {'down' if dy > 0 else 'up'}" if dy else "") if t) or "there"
            if not res["possible"]:
                return f"**{res['tag']}** can't go {how}: {res['reason']}"
            if res["recommendation"].startswith("keep"):
                why = []
                if res["leak_points_lost"]:
                    why.append(f"it would lose sight of leak point(s) **{self._join(res['leak_points_lost'])}**")
                if res["targets_broken"]:
                    why.append(f"{self._join(res['targets_broken'])} would drop below target")
                return f"Moving **{res['tag']}** {how} isn't a good idea: {' and '.join(why)}. **Keep it where it is.**"
            return (f"Yes, **{res['tag']}** can move {how}. All safety targets are still met after re-checking "
                    "the line of sight.")
        if name == "design_summary":
            met = sum(c["met"] for c in res["coverage"])
            return (f"**{res['flame_detectors'] + res['gas_detectors']} detectors** cost about "
                    f"**£{res['installed_cost_gbp'] / 1000:,.0f}k** to install, "
                    f"£{res['saving_vs_rule_of_thumb_gbp'] / 1000:,.0f}k less than the usual rule of thumb. "
                    f"**{met} of {len(res['coverage'])}** safety targets are met.")
        if name == "low_confidence_items":
            if not res["items"]:
                return "The AI was confident about everything it read from the drawing."
            return (f"The AI wasn't sure about **{len(res['items'])} item(s)**: "
                    + self._join(f"{i['item']} ({i['confidence']})" for i in res["items"])
                    + ". An engineer should check these on the drawing.")
        return self.describe(name, res, "technical")

    def describe(self, name, res, level="technical"):
        """Put a tool result into words. level="simple" gives a short plain-English answer for anyone;
        "technical" gives the full figures for an engineer."""
        if "error" in res:
            return res["error"]
        if level == "simple":
            return self.describe_simple(name, res)
        if name == "explain_detector":
            face = "all round" if res["facing_deg"] is None else f"facing {res['facing_deg']}°"
            s = (f"**{res['tag']}** ({res['type']}) is at x {res['position_m'][0]} m, y {res['position_m'][1]} m, "
                 f"mounted at {res['mounted_at_m']:.0f} m, {face}. {res['reason']}")
            if res["leak_points_only_it_watches"]:
                s += (f" It is the only detector that watches leak point(s) "
                      f"{', '.join(res['leak_points_only_it_watches'])}.")
            elif res["leak_points_it_watches"]:
                s += f" It watches leak point(s) {', '.join(res['leak_points_it_watches'])}."
            s += f" Installed cost about £{res['installed_cost_gbp']:,}."
            return s + self._conf_note(res)
        if name == "detector_fails":
            ch = [f"{c['target']}: {c['before']} → **{c['after']}**" + ("" if c["met"] else " ❌ below target")
                  for c in res["coverage_if_it_fails"] if c["before"] != c["after"]]
            s = f"If **{res['tag']}** fails, " + ("coverage changes:\n- " + "\n- ".join(ch) if ch else
                                                  "no coverage figure changes; other detectors already see everything it sees.")
            if res["leak_points_left_unwatched"]:
                s += f"\n\nLeak point(s) **{', '.join(res['leak_points_left_unwatched'])}** would go unwatched."
            s += ("\n\nResult: **" + ", ".join(res["targets_broken"]) + "** would fall below target, so this detector "
                  "is critical." if res["targets_broken"] else "\n\nResult: every safety target is still met.")
            return s + self._conf_note(res)
        if name == "move_detector":
            if not res["possible"]:
                return f"Can't move **{res['tag']}** there: {res['reason']}"
            ch = [f"{c['target']}: {c['before']} → **{c['after']}**" + ("" if c["met"] else " ❌ below target")
                  for c in res["coverage_after_move"] if c["before"] != c["after"]]
            s = (f"Moved **{res['tag']}** to x {res['new_position_m'][0]} m, y {res['new_position_m'][1]} m and re-ran "
                 f"the line-of-sight check. It would see {res['area_seen_after_m2']} m² instead of "
                 f"{res['area_seen_before_m2']} m².")
            s += ("\n- " + "\n- ".join(ch)) if ch else " No coverage target changes."
            if res["leak_points_lost"]:
                s += f"\n\nIt would no longer see leak point(s) **{', '.join(res['leak_points_lost'])}**."
            if res["leak_points_gained"]:
                s += f"\n\nIt would newly cover leak point(s) {', '.join(res['leak_points_gained'])}."
            return s + f"\n\n**Recommendation: {res['recommendation']}.**"
        if name == "design_summary":
            met = sum(c["met"] for c in res["coverage"])
            s = (f"The layout uses **{res['flame_detectors']} flame and {res['gas_detectors']} gas detectors** for about "
                 f"**£{res['installed_cost_gbp']:,}** installed, £{res['saving_vs_rule_of_thumb_gbp']:,} cheaper than "
                 f"the rule-of-thumb layout. {met} of {len(res['coverage'])} safety targets are met and "
                 f"{res['leak_points_watched']} leak points are watched.\n- ")
            return s + "\n- ".join(f"{c['target']}: {c['achieved']} (needs {c['required']})" for c in res["coverage"])
        if name == "low_confidence_items":
            if not res["items"]:
                return f"Every drawing item was read with at least {res['threshold']} confidence."
            return (f"These items were read with under {res['threshold']} confidence. An engineer should check them:\n- "
                    + "\n- ".join(f"**{i['item']}** {i['what']}, {i['confidence']}"
                                  + (f", used by {', '.join(i['detectors_relying_on_it'])}" if i["detectors_relying_on_it"]
                                     else "") for i in res["items"]))
        if name == "list_detectors":
            return "Detectors: " + ", ".join(f"{d['tag']} ({d['type']})" for d in res["detectors"])
        return json.dumps(res)

    # ------------------------------------------------------------------ Claude front end (optional)
    SYSTEM = ("You are the design assistant inside DetectorPlace, a fire and gas detector layout tool. An engineer is "
              "asking about the layout on screen. Answer only from tool results: call a tool for every fact and never "
              "estimate coverage yourself. Detector tags look like FD-01 (flame) and GD-01 (gas); 'D2' means FD-02 "
              "unless only a gas detector matches. Leak points are L1, L2, and so on. If a result lists low-confidence drawing items, "
              "say an engineer should check them first. You never change the design; the engineer decides.")

    @staticmethod
    def claude_available():
        try:
            import anthropic  # noqa: F401
        except ImportError:
            return False
        return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))

    STYLE = {"simple": "Answer in one or two short sentences of everyday English for a non-specialist.",
             "technical": "Answer for a fire and gas engineer in two to four sentences with the key figures in bold."}

    def answer_claude(self, question, history=(), selected=None, level="technical"):
        """Manual tool-use loop. Returns (answer_text, tool_calls_made)."""
        import anthropic
        client = anthropic.Anthropic()
        msgs = [{"role": m["role"], "content": m["content"]} for m in history if m["role"] in ("user", "assistant")]
        hint = f" (The engineer has {selected} selected.)" if selected else ""
        msgs.append({"role": "user", "content": question + hint})
        calls = []
        for _ in range(6):
            resp = client.beta.messages.create(
                model=MODEL, max_tokens=4000, system=self.SYSTEM + " " + self.STYLE[level], tools=self.TOOLS,
                messages=msgs,
                output_config={"effort": "low"},
                betas=["server-side-fallback-2026-07-01"], fallbacks="default")
            if resp.stop_reason == "refusal":
                return "I can't answer that one. Try asking about a specific detector.", calls
            msgs.append({"role": "assistant", "content": resp.content})
            uses = [b for b in resp.content if b.type == "tool_use"]
            if resp.stop_reason != "tool_use" or not uses:
                return "".join(b.text for b in resp.content if b.type == "text").strip(), calls
            results = []
            for b in uses:
                out = self.call(b.name, b.input)
                calls.append((b.name, b.input))
                results.append({"type": "tool_result", "tool_use_id": b.id, "content": json.dumps(out)})
            msgs.append({"role": "user", "content": results})
        return "That needed too many steps. Try a more specific question.", calls

    def answer(self, question, history=(), selected=None, use_claude=None, level="technical"):
        """Returns (text, engine). Falls back to the built-in engine if Claude is unavailable or errors."""
        if use_claude is None:
            use_claude = self.claude_available()
        if use_claude:
            try:
                text, _ = self.answer_claude(question, history, selected, level)
                if text:
                    return text, "Claude"
            except Exception:
                pass
        return self.answer_offline(question, selected, level), "built-in"
