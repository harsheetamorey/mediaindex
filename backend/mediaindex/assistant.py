"""Ask assistant: answers questions about the photo library in plain language.

- "How many dogs?"           -> object counts from the detector (detect.py), with the photos they come from
- "Show me beach sunsets"    -> EmbeddingGemma search (ranked by similarity, as in Search)
- "Which of those have people?" / "...are at night?" -> narrows or re-ranks the previous answer's photos
- "Describe the second one"  -> the local chat model looks at that one photo (needs Ollama + Gemma 4)
- anything else              -> the local chat model replies, given only facts from the library

Numbers always come from the database, never from the chat model: factual answers are fixed sentences, and a free
chat reply that mentions a number not present in the facts it was given is replaced by a fixed summary.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable

from .detect import LABELS
from .llm import ChatUnavailable, OllamaChat
from .store import Store

ACTIONS = ("count", "search", "refine", "describe", "chat")

DISPLAY = {"diningtable": "dining table", "pottedplant": "potted plant", "tvmonitor": "TV", "aeroplane": "airplane",
           "motorbike": "motorbike", "sports ball": "ball"}
PLURAL = {"person": "people", "bus": "buses", "sheep": "sheep", "skis": "skis", "mouse": "mice", "knife": "knives",
          "bench": "benches", "sandwich": "sandwiches", "wine glass": "wine glasses", "toothbrush": "toothbrushes",
          "hot dog": "hot dogs", "tvmonitor": "TVs", "scissors": "scissors", "cell phone": "phones",
          "diningtable": "dining tables", "pottedplant": "potted plants", "aeroplane": "airplanes",
          "sports ball": "balls", "couch": "couches", "broccoli": "broccoli"}
SYNONYMS = {
    "person": "people persons human humans man men woman women kid kids child children boy boys girl girls guy guys "
              "lady ladies adult adults baby babies crowd folks",
    "dog": "dogs puppy puppies doggy doggies pup pups hound",
    "cat": "cats kitten kittens kitty kitties",
    "car": "cars automobile automobiles taxi taxis cab cabs",
    "bicycle": "bicycles bike bikes cycle cycles",
    "motorbike": "motorbikes motorcycle motorcycles scooter scooters",
    "aeroplane": "aeroplanes airplane airplanes plane planes aircraft jet jets",
    "sofa": "sofas couch couches",
    "tvmonitor": "tv tvs television televisions monitor monitors",
    "diningtable": "table tables",
    "pottedplant": "plant plants houseplant houseplants",
    "cell phone": "phone phones smartphone smartphones mobile",
    "sports ball": "ball balls football soccer basketball",
    "handbag": "handbags bag bags purse purses",
    "truck": "trucks lorry lorries",
    "boat": "boats ship ships",
    "cow": "cows cattle",
    "bird": "birds seagull seagulls gull gulls pigeon pigeons duck ducks",
    "horse": "horses pony ponies",
    "teddy bear": "teddy teddies",
    "wine glass": "glass glasses",
    "cup": "cups mug mugs",
    "laptop": "laptops computer computers",
}


def _build_index() -> dict[str, str]:
    idx: dict[str, str] = {}
    for label in LABELS:
        idx[label] = label
        idx[DISPLAY.get(label, label)] = label
        idx[PLURAL.get(label, label + "s")] = label
    for label, words in SYNONYMS.items():
        for w in words.split():
            idx.setdefault(w, label)
    return idx


WORD_TO_LABEL = _build_index()
FILLER = set("""a an the any all my our your of in on at with do does did i we you have has had are is there was were
photo photos picture pictures image images pic pics shot shots library folder collection total altogether
contain contains containing show see find can could please how many number count counted some me which those
these them it they that this one ones keep only just hmm ok okay so"""
             .split())


def resolve_labels(subject: str) -> list[str]:
    """Map a phrase ("puppies", "cats or dogs", "a TV") to the detector's labels it names, in order."""
    words = re.findall(r"[a-z]+", subject.lower())
    found, i = [], 0
    while i < len(words):
        two = " ".join(words[i:i + 2])
        if len(words) > i + 1 and two in WORD_TO_LABEL:  # two-word labels first ("teddy bear", "traffic light")
            label, i = WORD_TO_LABEL[two], i + 2
        else:
            label, i = WORD_TO_LABEL.get(words[i]), i + 1
        if label and label not in found:
            found.append(label)
    return found


def resolve_label(subject: str) -> str | None:
    labels = resolve_labels(subject)
    return labels[0] if labels else None


def display(label: str, n: int = 2) -> str:
    return DISPLAY.get(label, label) if n == 1 else PLURAL.get(label, DISPLAY.get(label, label) + "s")


ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8,
            "ninth": 9, "tenth": 10, "1st": 1, "2nd": 2, "3rd": 3, "last": -1}
COUNT_RE = re.compile(r"\b(how many|number of|count)\b", re.I)
COMPARE_RE = re.compile(r"\b(more|fewer|less|most)\b.+\bor\b", re.I)
DESCRIBE_RE = re.compile(r"\b(describe|tell me (more )?about|what'?s (in|happening in)|what is (in|happening in)|"
                         r"what do you see)\b", re.I)
REFINE_RE = re.compile(r"\b(which of|of those|of these|among (them|those|these)|the ones|which ones|only keep|keep only|"
                       r"just keep|those|these|them)\b", re.I)
REFERENCE_RE = re.compile(r"\b(this|that|it|one|photo|picture|image)\b", re.I)
CHAT_RE = re.compile(r"^\s*(hi|hello|hey|thanks|thank you|help|who are you|what can you do)\b", re.I)


@dataclass
class Plan:
    action: str
    subject: str = ""
    item: int | None = None
    routed_by: str = "rules"


def _item(text: str) -> int | None:
    t = text.lower()
    for w, n in ORDINALS.items():
        if re.search(rf"\b{w}\b", t):
            return n
    m = re.search(r"(?:#|number |no\.? ?|photo |picture |image )(\d{1,3})\b", t)
    return int(m.group(1)) if m else None


def _strip(text: str) -> str:
    kept = [w for w in re.findall(r"[\w'-]+", text) if w.lower() not in FILLER]
    return " ".join(kept).strip()


def rule_plan(message: str, has_previous: bool) -> Plan | None:
    """Deterministic routing for clear phrasings. Returns None when the question needs the chat model."""
    m = message.strip()
    if COUNT_RE.search(m):
        return Plan("count", _strip(COUNT_RE.split(m, 1)[-1]))
    if COMPARE_RE.search(m) and len(resolve_labels(m)) >= 2:  # "more cats or dogs?"
        return Plan("count", _strip(m))
    if DESCRIBE_RE.search(m) and has_previous and (_item(m) or REFERENCE_RE.search(m)):
        return Plan("describe", item=_item(m) or 1)
    if has_previous and REFINE_RE.search(m):
        return Plan("refine", _strip(REFINE_RE.sub(" ", m)))
    if CHAT_RE.search(m):
        return Plan("chat")
    if re.match(r"^\s*(show|find|search|look for|get|give me|any|photos? of|pictures? of|images? of)\b", m, re.I):
        return Plan("search", _strip(m) or m)
    return None


ROUTER_PROMPT = """You turn a question about the user's photo library into JSON:
{"action": "count" | "search" | "refine" | "describe" | "chat", "subject": "<thing or scene>", "item": <number or 0>}
- count: how many of something ("how many dogs", "are there lots of cars?")
- search: find or show photos of something ("sunsets", "my dog at the beach")
- refine: narrow down the photos from the previous answer ("which are outdoors?", "only the night ones");
  subject = what to keep, e.g. "umbrellas" for "only keep the ones with umbrellas"
- describe: describe one photo from the previous answer; item = its position (1 = first, -1 = last)
- chat: greetings, thanks, questions about what you can do, or anything else
Reply with JSON only."""

ROUTER_SCHEMA = {"type": "object", "properties": {"action": {"type": "string", "enum": list(ACTIONS)},
                                                  "subject": {"type": "string"}, "item": {"type": "integer"}},
                 "required": ["action", "subject", "item"]}


@dataclass
class Turn:
    role: str
    text: str
    asset_ids: list[str] = field(default_factory=list)
    action: str | None = None  # for assistant turns: which kind of answer it was


SearchFn = Callable[[str, list[str] | None, int], list[str]]  # text, library_ids, limit -> asset ids by similarity
RankFn = Callable[[str, list[str]], list[str]]  # text, candidate asset ids -> the same ids, most similar first


class Assistant:
    def __init__(self, store: Store, detector_key: str, search: SearchFn, rank_within: RankFn,
                 image_b64: Callable[[str], str], chat: OllamaChat | None = None, result_limit: int = 12):
        self.store = store
        self.key = detector_key
        self.search = search
        self.rank_within = rank_within
        self.image_b64 = image_b64
        self.chat = chat
        self.limit = result_limit

    # ---- routing --------------------------------------------------------------------
    def plan(self, message: str, history: list[Turn], use_chat: bool) -> Plan:
        previous = self._previous(history)
        p = rule_plan(message, bool(previous))
        if p is not None or not use_chat:
            return p or Plan("search", _strip(message) or message)
        context = [f"{t.role}: {t.text}" for t in history[-4:]]
        prompt = (("Earlier conversation:\n" + "\n".join(context) + "\n\n") if context else "") + \
                 f"The previous answer {'showed ' + str(len(previous)) + ' photos' if previous else 'showed no photos'}." \
                 f"\nQuestion: {message}"
        try:
            raw = self.chat.chat([{"role": "system", "content": ROUTER_PROMPT}, {"role": "user", "content": prompt}],
                                 schema=ROUTER_SCHEMA, max_tokens=60)
            d = json.loads(raw)
            action = d.get("action")
            if action not in ACTIONS or (action in ("refine", "describe") and not previous):
                raise ValueError(action)
            return Plan(action, str(d.get("subject") or "").strip() or _strip(message), int(d.get("item") or 1),
                        routed_by="chat model")
        except (ChatUnavailable, ValueError, TypeError, json.JSONDecodeError):
            return Plan("search", _strip(message) or message)

    @staticmethod
    def _previous(history: list[Turn]) -> list[str]:
        for t in reversed(history):
            if t.role == "assistant" and t.asset_ids and t.action != "describe":  # the last list, not one photo
                return t.asset_ids
        return []

    # ---- answering ------------------------------------------------------------------
    def ask(self, message: str, history: list[Turn], library_ids: list[str] | None = None,
            asset_id: str | None = None) -> dict:
        status = self.chat.status() if self.chat else {"available": False, "reason": "no chat model configured"}
        use_chat = bool(status.get("available"))
        if asset_id:  # the user clicked a photo and asked about it
            plan = Plan("describe", item=None)
        else:
            plan = self.plan(message, history, use_chat)
        handler = getattr(self, "_" + plan.action)
        out = handler(plan, message, history, library_ids, use_chat, asset_id)
        out.update(action=plan.action, routed_by=plan.routed_by, chat_model=status)
        out.setdefault("asset_ids", [])
        return out

    def _coverage(self, library_ids):
        return self.store.detection_coverage(self.key, library_ids)

    def _count(self, plan, message, history, library_ids, use_chat, asset_id):
        cov = self._coverage(library_ids)
        labels = resolve_labels(plan.subject) or resolve_labels(message)
        facts = {"subject": plan.subject, "labels": labels, "coverage": cov}
        if not labels:
            ids = self.search(plan.subject or message, library_ids, self.limit)
            return {"answer": (f"I can count common things like people, dogs, cats, cars and birds, but not "
                               f"“{plan.subject or message}”. Here are the closest photos instead."),
                    "asset_ids": ids, "facts": facts}
        if cov["checked"] == 0:
            return {"answer": ("Your photos haven't been checked for objects yet. Press “Count objects” above, "
                               "then ask again."), "facts": facts}
        parts, ids = [], []
        for label in labels[:4]:
            c = self.store.count_objects(self.key, label, library_ids)
            facts[label] = {"photos": c["photos"], "objects": c["objects"]}
            ids += [a for a in c["asset_ids"] if a not in ids]
            if c["photos"] == 0:
                parts.append(f"I didn't find any {display(label)} in the {cov['checked']} photos I checked.")
            else:
                parts.append(f"{display(label).capitalize()} appear in {c['photos']} of {cov['checked']} photos "
                             f"({c['objects']} {display(label, c['objects'])} counted).")
        text = " ".join(parts)
        unchecked = cov["photos"] - cov["checked"]
        if unchecked:
            text += f" {unchecked} newer photos haven't been checked yet."
        text += " Counts come from automatic detection and can miss small or crowded objects."
        return {"answer": text, "asset_ids": ids[:48], "facts": facts}

    def _search(self, plan, message, history, library_ids, use_chat, asset_id):
        q = plan.subject or message
        ids = self.search(q, library_ids, self.limit)
        if not ids:
            return {"answer": "Nothing is indexed yet. Add a folder on the left first.", "facts": {"query": q}}
        return {"answer": f"Here are the closest matches for “{q}”, best first. Similar isn't always a match, "
                          f"so the last ones may be off.", "asset_ids": ids, "facts": {"query": q}}

    def _refine(self, plan, message, history, library_ids, use_chat, asset_id):
        prev = self._previous(history)
        q = plan.subject or message
        label = resolve_label(q)
        if label is not None and self._coverage(library_ids)["checked"]:
            keep = set(self.store.count_objects(self.key, label, library_ids)["asset_ids"])
            ids = [a for a in prev if a in keep]
            return {"answer": f"{len(ids)} of those {len(prev)} photos {'has' if len(ids) == 1 else 'have'} "
                              f"{display(label)}.",
                    "asset_ids": ids, "facts": {"label": label, "of": len(prev), "kept": len(ids)}}
        ids = self.rank_within(q, prev)
        return {"answer": f"Here {'is that photo' if len(prev) == 1 else f'are those {len(prev)} photos'} again, closest to “{q}” first. I can't be sure which "
                          f"ones truly match, so have a look.", "asset_ids": ids, "facts": {"query": q, "of": len(prev)}}

    def _describe(self, plan, message, history, library_ids, use_chat, asset_id):
        prev = self._previous(history)
        if asset_id is None:
            if not prev:
                return {"answer": "Ask for some photos first, then ask me to describe one of them.", "facts": {}}
            n = plan.item or 1
            if n == -1:
                n = len(prev)
            if not 1 <= n <= len(prev):
                return {"answer": f"The last answer had {len(prev)} photos; pick a number from 1 to {len(prev)}.",
                        "facts": {}}
            asset_id = prev[n - 1]
        asset = self.store.get_asset(asset_id)
        if asset is None or asset["media_type"] != "image":
            return {"answer": "I can only describe photos.", "facts": {}}
        objects = self.store.detections_for(asset["content_hash"], self.key)
        facts = {"asset_id": asset_id, "file": asset["rel_path"], "objects": objects}
        if use_chat:
            q = message.strip() or "Describe this photo."
            try:
                text = self.chat.chat([{"role": "system", "content": "Describe the photo truthfully in 1-3 sentences. "
                                        "Only mention what you can see."},
                                       {"role": "user", "content": q, "images": [self.image_b64(asset_id)]}],
                                      max_tokens=160)
                return {"answer": text, "asset_ids": [asset_id], "facts": facts, "described_by": "chat model"}
            except ChatUnavailable:
                pass
        found = ", ".join(f"{n} {display(k, n)}" for k, n in (objects or {}).items())
        text = (f"I found {found} in {asset['rel_path']}." if found else
                f"I didn't detect any common objects in {asset['rel_path']}." if objects is not None else
                f"{asset['rel_path']} hasn't been checked for objects yet.")
        return {"answer": text + " (Turn on the local chat model to get a written description.)",
                "asset_ids": [asset_id], "facts": facts}

    def _chat(self, plan, message, history, library_ids, use_chat, asset_id):
        cov = self._coverage(library_ids)
        summary = self.store.object_summary(self.key, library_ids)  # all 80 types at most: small enough
        fixed = ("I can count things in your photos (“how many dogs?”), find photos (“beach at sunset”), narrow "
                 "down the last answer (“which of those have people?”) and describe a photo (“describe the "
                 "second one”).")
        if summary:
            fixed += " Most common in your photos: " + ", ".join(
                f"{display(s['label'])} ({s['photos']} photos)" for s in summary[:5]) + "."
        if not use_chat:
            return {"answer": fixed, "facts": {"coverage": cov}}
        facts = {"photos": cov["photos"], "photos_checked_for_objects": cov["checked"],
                 "objects_found": [{"thing": display(s["label"]), "photos": s["photos"], "count": s["objects"]}
                                   for s in summary]}
        system = ("You are the assistant inside MediaIndex, a private photo search app that runs on this computer. "
                  "You can: count common objects in photos, find photos by description, narrow down the last "
                  "results, and describe a photo. Answer in 1-3 short sentences. Use only these facts and never "
                  "invent numbers or photos: " + json.dumps(facts))
        msgs = [{"role": "system", "content": system}]
        msgs += [{"role": t.role, "content": t.text} for t in history[-4:] if t.role in ("user", "assistant")]
        msgs.append({"role": "user", "content": message})
        try:
            text = self.chat.chat(msgs, max_tokens=160)
        except ChatUnavailable:
            return {"answer": fixed, "facts": facts}
        allowed = {str(v) for v in re.findall(r"\d+", json.dumps(facts) + " " + message)}
        if any(n not in allowed for n in re.findall(r"\d+", text)):
            return {"answer": fixed, "facts": facts, "note": "chat reply replaced: it mentioned numbers not in the facts"}
        return {"answer": text, "facts": facts}
