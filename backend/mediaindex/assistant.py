"""Ask assistant: answers questions about the photo library in plain language.

- "How many dogs?"           -> object counts from the detector (detect.py), with the photos they come from
- "Show me beach sunsets"    -> EmbeddingGemma search (ranked by similarity, as in Search)
- "Which of those have people?" / "...are at night?" -> narrows or re-ranks the previous answer's photos
- "Describe the second one"  -> the local chat model looks at that one photo (needs Ollama + Gemma 4)
- "Remove the car from photo 2" / "replace the dog with a cat" -> a new, edited copy of that photo (edit.py);
  the original is never changed. The edit itself runs as a background job (api_ask.py).
- "Make an image of a red bicycle"  -> a new, made-up image from the local image model (creative.py), labelled as such
- "Make a video of those" / "turn my beach photos into a reel" -> a slideshow video of exactly those photos (slideshow.py)
- anything else              -> the local chat model replies, given only facts from the library

Numbers always come from the database, never from the chat model: factual answers are fixed sentences, and a free
chat reply that mentions a number not present in the facts it was given is replaced by a fixed summary.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .detect import LABELS, SURE
from .llm import ChatUnavailable, OllamaChat
from .store import Store

ACTIONS = ("count", "search", "refine", "describe", "explain", "summary", "chat")

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


IRREGULAR_PLURALS = {"people", "persons", "men", "women", "children", "kids", "mice", "geese", "cattle", "sheep"}


def is_plural(subject: str) -> bool:
    """ "cars", "the people", "puppies" -> True; "car", "bus", "wine glass" -> False."""
    for w in re.findall(r"[a-z]+", subject.lower()):
        if w in IRREGULAR_PLURALS or w in {PLURAL.get(lab, DISPLAY.get(lab, lab) + "s") for lab in LABELS} - {"sheep"}:
            return True
        if w.endswith("s") and w not in WORD_TO_LABEL.values() and \
                (w[:-1] in WORD_TO_LABEL or w[:-2] in WORD_TO_LABEL or w[:-3] + "y" in WORD_TO_LABEL):
            return True
    return False


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
REFINE_RE = re.compile(r"\b(which of|of those|of these|among (them|those|these)|the ones|which ones|which one|the one with|"
                       r"the one that|only keep|keep only|just keep|only the|just the|those|these|them)\b", re.I)
TOP_REFINE = 3  # a follow-up the detector can't check ("which one has the peacock?"): the closest 3 first, the rest "maybe"
# colour of a detected object, measured in its own box (PIL HSV, 0-255): name -> test on hue, saturation, value
COLOUR_TESTS = {
    "red": lambda h, s, v: ((h < 10) | (h > 240)) & (s > 110) & (v > 60),
    "orange": lambda h, s, v: (h >= 10) & (h < 26) & (s > 120) & (v > 90),
    "yellow": lambda h, s, v: (h >= 26) & (h < 46) & (s > 110) & (v > 100),
    "green": lambda h, s, v: (h >= 46) & (h < 120) & (s > 70) & (v > 50),
    "blue": lambda h, s, v: (h >= 130) & (h < 190) & (s > 80) & (v > 50),
    "purple": lambda h, s, v: (h >= 190) & (h <= 240) & (s > 70) & (v > 50),
    "white": lambda h, s, v: (s < 40) & (v > 190),
    "black": lambda h, s, v: v < 50,
    "grey": lambda h, s, v: (s < 40) & (v >= 70) & (v <= 190),
}
COLOUR_WORDS = {**{c: c for c in COLOUR_TESTS}, "gray": "grey", "silver": "grey"}
# share of the box in that colour: kept / "maybe". Set on 13 car photos: the 4 red cars 0.13-0.47 (a small car in
# a big box, or deep shadow, gives less), every other car <= 0.01. Only red has been checked against photos.
COLOUR_SURE, COLOUR_MAYBE = 0.10, 0.04
REFERENCE_RE = re.compile(r"\b(this|that|it|one|photo|picture|image)\b", re.I)
EXPLAIN_RE = re.compile(r"\b(why|where('?s| is| are)?|not an?|isn'?t|is not|are not|aren'?t|none of|wrong|are you sure|really|"
                        r"(don'?t|do not|can'?t|cannot) see|no (dog|cat|one)s?)\b", re.I)
NEGATION_RE = re.compile(r"\b(not|isn'?t|aren'?t|no|wrong|none)\b", re.I)
REMOVE_VERBS = r"remove|erase|delete|get rid of|take out|take away|edit out|clean up|cut out"
REMOVE_RE = re.compile(rf"\b(?:{REMOVE_VERBS})\s+(?P<obj>.+)$", re.I)
REPLACE_RE = re.compile(r"\b(?:replace|swap|change|turn)\s+(?P<obj>.+?)\s+(?:with|for|into|to)\s+(?P<new>.+)$", re.I)
# where the edit should happen: "... from the second one", "... in photo 2", "... in it"
WHERE_RE = re.compile(r"\s+(?:from|in|on|off|of)\s+(?:the\s+|this\s+|that\s+|my\s+)?(?:photo|picture|image|pic|one|it|"
                      r"this|that|#?\d|first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|last|1st|2nd|"
                      r"3rd)\b.*$", re.I)
GENERATE_RE = re.compile(r"^\s*(?:please\s+|can you\s+|could you\s+)*(?:generate|create|make|draw|paint|imagine|design)"
                         r"\s+(?:me\s+)?(?:an?\s+|some\s+)?(?:new\s+)?(?:(?:wide|tall|square|landscape|portrait)\s+)?"
                         r"(?:image|picture|photo|drawing|painting|illustration|artwork|art|pic)s?\s+"
                         r"(?:of|showing|with|that shows)\s+(?P<what>.+)$", re.I)
VIDEO_RE = re.compile(r"\b(?:make|create|generate|turn|build|put|render)\b.*\b(?P<kind>video|slideshow|slide show|reel|"
                      r"short|movie|montage|story)s?\b", re.I)
VERTICAL_RE = re.compile(r"\b(reel|short|story|vertical|portrait|tiktok|instagram|phone)s?\b", re.I)
SQUARE_RE = re.compile(r"\bsquare\b", re.I)
PRONOUN_RE = re.compile(r"^\s*(those|these|them|they|it|this|that|the ones|those ones|these ones|the photos|the pictures)?\s*$",
                        re.I)
ALL_RE = re.compile(r"\b(all|every|each|both)\b", re.I)
# after "I didn't change it: … too big": the user says to go ahead anyway
TOO_BIG_RE = re.compile(r"\btoo big to (replace|remove)\b", re.I)
CONFIRM_RE = re.compile(r"^\s*(?:yes|yeah|yep|ok|okay|sure|fine|go ahead|go for it|do it|please do|try (?:it|anyway)|"
                        r"it'?s (?:ok|okay|fine)|that'?s (?:ok|okay|fine)|i don'?t mind|(?:do|try) it anyway|anyway|"
                        r"make (?:it|most of it) up)\b", re.I)
# after a replacement: "put a truck instead", "try a bus", "make it a lorry": the same edit with something else
REDO_RE = re.compile(r"^\s*(?:(?:can|could|would)\s+(?:you|u)\s+|please\s+|now\s+|instead\s+|ok\s+|okay\s+)*"
                     r"(?:put|try|use|make it|do|give me|paint|show me)\s+(?P<new>(?:an?|some|the)\s+[\w\s-]+?)"
                     r"(?:\s+(?:instead|there|here|in it|in there))?\s*$", re.I)
REDO_WITH_RE = re.compile(r"^\s*(?:(?:can|could|would)\s+(?:you|u)\s+|please\s+|now\s+|ok\s+|okay\s+)*"
                          r"(?:replace|swap|change)\s+(?:it\s+|that\s+|this\s+|them\s+|those\s+|these\s+)?(?:with|for|to|into)\s+"
                          r"(?P<new>.+?)(?:\s+instead)?\s*$", re.I)
# after a removal: "it should look seamless", "there's a gap", "try again": the same removal, another try
RETRY_RE = re.compile(r"\b(seamless|seamlessly|gap|smudge|smudges|smear|smeared|blur|blurry|blocky|patch|shadow|"
                      r"cleaner|better|neater|neat|again|redo|retry|another try|not clean|still see)\b", re.I)
PRONOUN_OBJ = {"this", "that", "it", "them", "this one", "that one", "the thing", "the object"}
COLOURS = ("white black red blue green yellow orange purple pink brown grey gray silver gold beige dark light navy "
           "maroon teal").split()
VEHICLES = {"car", "truck", "bus"}
VEHICLE_WORDS = re.compile(r"\b(car|cars|van|vans|minivan|suv|jeep|truck|trucks|lorry|lorries|pickup|pick-up|bus|buses|"
                           r"taxi|cab|coach|limo|limousine|sedan|hatchback|wagon|ambulance|camper)\b", re.I)
CHAT_RE = re.compile(r"^\s*(hi|hello|hey|thanks|thank you|help|who are you|what can you do)\b", re.I)


@dataclass
class Plan:
    action: str
    subject: str = ""
    item: int | None = None
    routed_by: str = "rules"
    new: str = ""  # edit: what to put in place of `subject` ("" = remove it)
    big: bool = False  # edit: the user said to go ahead although it changes most of the photo
    seed: int = 0  # edit: > 0 when the user asked to try a removal again


def _parts(message: str) -> list[str]:
    return re.split(r"[.;!?]|,\s|\s+-\s+", message.strip())


def clean_new(new: str) -> str:
    """ "a van here. van should be white color" -> "a white van": the first phrase, plus a colour said after it."""
    parts = re.split(r"[.;!?]|,\s|\s+(?:but|and make|so that|which|that should|it should|should be)\b", new, maxsplit=1)
    head = parts[0]
    head = re.sub(r"\s+(?:here|there|in it|in this|in that|in there|instead|please|too|now)\s*$", "", head,
                  flags=re.I).strip()
    rest = parts[1] if len(parts) > 1 else ""
    colour = next((c for c in re.findall(r"[a-z]+", rest.lower()) if c in COLOURS), None)
    if colour and not any(w in COLOURS for w in head.lower().split()):
        m = re.match(r"^(an?|the|some)\s+(.*)$", head, re.I)
        head = f"{'an' if colour[0] in 'aeiou' else 'a'} {colour} {m.group(2)}" if m and m.group(1).lower() in ("a", "an") \
            else f"{m.group(1)} {colour} {m.group(2)}" if m else f"{colour} {head}"
    return head or new.strip()


def same_kind(label: str, new: str) -> bool:
    """Replacing a car with a van: the detector will still call it a car, so "still there" means nothing."""
    return label in VEHICLES and bool(VEHICLE_WORDS.search(new))


def create_plan(message: str) -> Plan | None:
    """ "make an image of a red bike" -> generate; "make a video of my beach photos" -> video (new = its shape)."""
    m = message.strip().rstrip(".!?")
    if (g := GENERATE_RE.search(m)) is not None:
        shape = "wide" if re.search(r"\b(wide|landscape)\b", m, re.I) else \
            "tall" if re.search(r"\b(tall|portrait)\b", m, re.I) else "square"
        return Plan("generate", g.group("what").strip(), new=shape)
    if (v := VIDEO_RE.search(m)) is not None:
        shape = "vertical" if VERTICAL_RE.search(m) else "square" if SQUARE_RE.search(m) else "landscape"
        rest = m[:v.start("kind")] + " " + m[v.end("kind"):]  # what the video is of, around the word "video"
        rest = re.sub(r"\b(make|create|generate|turn|build|put|render|me|a|an|the|out of|of|from|with|into|my|our|all|"
                      r"photos?|pictures?|pics?|images?|shots?|please|can you|could you|vertical|square|landscape|"
                      r"portrait|for|instagram|tiktok|youtube|phone)\b", " ", rest, flags=re.I)
        rest = re.sub(r"\s+", " ", rest).strip(" ,")
        return Plan("video", "" if PRONOUN_RE.match(rest) else rest, new=shape)
    return None


def edit_plan(message: str) -> Plan | None:
    """ "remove the car from photo 2" -> edit(subject="car", item=2); "replace the dog with a cat" -> new="a cat"."""
    m = message.strip().rstrip(".!?")
    r = REPLACE_RE.search(m)
    if r:
        obj, new = WHERE_RE.sub("", r.group("obj")), WHERE_RE.sub("", r.group("new"))
    elif (r := REMOVE_RE.search(m)):
        obj, new = WHERE_RE.sub("", r.group("obj")), ""
    else:
        return None
    obj = re.sub(r"^\s*(?:all the|both the|all of the|the|a|an|all|every|each|both|that|this|those|these)\s+", "", obj, flags=re.I)
    obj = re.sub(r"\s+(?:too|also|as well|please|now|again)\s*$", "", obj, flags=re.I).strip()
    new = clean_new(new) if new else ""
    if not obj:
        return None
    return Plan("edit", obj, _item(m), new=new)


def _items(text: str) -> list[int]:
    """All photo numbers in a correction: "photos 1, 2 and 4", "the first and the last", "#3"."""
    t = text.lower()
    found = [n for w, n in ORDINALS.items() if re.search(rf"\b{w}\b", t)]
    found += [int(x) for x in re.findall(r"\b(\d{1,3})\b", t)]
    return list(dict.fromkeys(found))


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


def rule_plan(message: str, has_previous: bool, has_focus: bool | None = None) -> Plan | None:
    """Deterministic routing for clear phrasings. Returns None when the question needs the chat model.

    has_previous: the conversation has a list of photos; has_focus: it has any photos (a list or one described photo).
    """
    m = message.strip()
    if (has_previous if has_focus is None else has_focus) and EXPLAIN_RE.search(m) and not COUNT_RE.search(m):
        return Plan("explain", _strip(m))  # "why did you say it's a dog?", "that's not a cat", "where is the dog?"
    if (c := create_plan(m)) is not None:  # "make an image of…", "make a video of those"
        return c
    e = edit_plan(m)  # only ever routed by these rules, never by the chat model: edits must be asked for in words
    if e is not None and ((has_previous if has_focus is None else has_focus) or REMOVE_RE.search(m)):
        return e  # "remove the car from the second one", "replace the dog with a cat"
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
    if re.match(r"^\s*(show|find|search|look for)\b", m, re.I):  # unambiguous only; other wordings go to the chat model
        return Plan("search", _strip(m) or m)
    return None


ROUTER_PROMPT = """You turn a question about the user's photo library into JSON:
{"action": "count" | "search" | "refine" | "describe" | "explain" | "summary" | "chat",
 "subject": "<thing or scene>", "item": <number or 0>}
- count: how many of something ("how many dogs", "are there lots of cars?")
- search: find or show photos of something ("sunsets", "my dog at the beach")
- refine: narrow down the photos from the previous answer ("which are outdoors?", "only the night ones");
  subject = what to keep, e.g. "umbrellas" for "only keep the ones with umbrellas"
- describe: describe one photo from the previous answer; item = its position (1 = first, -1 = last)
- explain: the user doubts or asks about an earlier answer ("why do you say that's a dog?", "where is it?")
- summary: what is in the library as a whole ("what's in my photos?", "give me an overview", "what do I have?")
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
    edit_id: str | None = None  # for assistant turns that showed an edited copy


USER = "user"  # corrections typed by the user ("photo 3 isn't a dog") always win

SearchFn = Callable[[str, list[str] | None, int], list[str]]  # text, library_ids, limit -> asset ids by similarity
RankFn = Callable[[str, list[str]], list[str]]  # text, candidate asset ids -> the same ids, most similar first


class Assistant:
    def __init__(self, store: Store, detector_key: str, search: SearchFn, rank_within: RankFn,
                 image_b64: Callable[[str], str], chat: OllamaChat | None = None, result_limit: int = 12,
                 load_photo: Callable[[str], object] | None = None):
        self.store = store
        self.key = detector_key
        self.search = search
        self.rank_within = rank_within
        self.image_b64 = image_b64
        self.chat = chat
        self.limit = result_limit
        self.load_photo = load_photo  # asset id -> PIL image (colour follow-ups)

    # ---- routing --------------------------------------------------------------------
    def plan(self, message: str, history: list[Turn], use_chat: bool) -> Plan:
        previous = self._previous(history)
        answered = bool(self._focus(history)) or any(t.role == "assistant" for t in history)
        p = rule_plan(message, bool(previous), answered)
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
            if action not in ACTIONS or (action in ("refine", "describe") and not previous) or \
                    (action == "explain" and not self._focus(history)):
                raise ValueError(action)
            return Plan(action, str(d.get("subject") or "").strip() or _strip(message), int(d.get("item") or 1),
                        routed_by="chat model")
        except (ChatUnavailable, ValueError, TypeError, json.JSONDecodeError):
            return Plan("search", _strip(message) or message)

    @staticmethod
    def _previous(history: list[Turn]) -> list[str]:
        for t in reversed(history):
            if t.role == "assistant" and t.asset_ids and t.action not in ("describe", "edit"):  # a list, not one photo
                return t.asset_ids
        return []

    @staticmethod
    def _focus(history: list[Turn]) -> list[str]:
        """The photos the conversation is about now: the last answer that showed any (a list or one photo)."""
        for t in reversed(history):
            if t.role == "assistant" and t.asset_ids:
                return t.asset_ids
        return []

    @staticmethod
    def _topic_label(message: str, history: list[Turn]) -> str | None:
        """The object being discussed: named in this message, or in the latest earlier question that named one."""
        label = resolve_label(message)
        if label:
            return label
        for t in reversed(history):
            if t.role == "user" and (label := resolve_label(t.text)):
                return label
        return None

    def _without_corrections(self, label: str, ids: list[str]) -> list[str]:
        """Drop photos the user said don't contain `label`."""
        hashes = {aid: a["content_hash"] for aid in ids if (a := self.store.get_asset(aid)) is not None}
        user = self.store.verifications(list(set(hashes.values())), label, USER)
        return [aid for aid in ids if user.get(hashes.get(aid), True)]

    # ---- answering ------------------------------------------------------------------
    def ask(self, message: str, history: list[Turn], library_ids: list[str] | None = None,
            asset_id: str | None = None) -> dict:
        status = self.chat.status() if self.chat else {"available": False, "reason": "no chat model configured"}
        use_chat = bool(status.get("available"))
        if not asset_id and (out := self.follow_up_edit(message, history, library_ids, use_chat)) is not None:
            plan = out.pop("plan")
        else:
            if asset_id:  # the user clicked a photo and asked about it: describe it, or edit it
                plan = edit_plan(message) or Plan("describe", item=None)
            else:
                plan = self.plan(message, history, use_chat)
            handler = getattr(self, "_" + plan.action)
            out = handler(plan, message, history, library_ids, use_chat, asset_id)
        out.update(action=plan.action, routed_by=plan.routed_by, chat_model=status)
        out.setdefault("asset_ids", [])
        box_labels = out.pop("box_labels", False)  # False: no boxes; None: every object; list: these labels
        out["boxes"] = {}
        if box_labels is not False:
            for aid in out["asset_ids"][:48]:
                a = self.store.get_asset(aid)
                if a is not None and (b := self.store.boxes_for(a["content_hash"], self.key, box_labels)):
                    out["boxes"][aid] = b
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
        parts, ids, maybe = [], [], []
        for label in labels[:4]:
            c = self.store.count_objects(self.key, label, library_ids, sure=SURE)
            kept = self._without_corrections(label, c["asset_ids"])
            unsure = [a for a in self._without_corrections(label, c["maybe_ids"]) if a not in ids and a not in kept]
            objects = sum(c["counts"][a] for a in kept)
            facts[label] = {"photos": len(kept), "objects": objects, "maybe_photos": len(unsure)}
            ids += [a for a in kept if a not in ids]
            maybe += [a for a in unsure if a not in maybe]
            if not kept:
                parts.append(f"I didn't find any {display(label)} in the {cov['checked']} photos I checked.")
            else:
                parts.append(f"{display(label).capitalize()} appear in {len(kept)} of {cov['checked']} photos "
                             f"({objects} {display(label, objects)} counted).")
            if unsure:
                parts.append(f"{len(unsure)} more {'photo' if len(unsure) == 1 else 'photos'} might have {display(label)}, "
                             f"but the detector isn't sure (shown last, marked “maybe”).")
        text = " ".join(parts)
        unchecked = cov["photos"] - cov["checked"]
        if unchecked:
            text += f" {unchecked} newer photos haven't been checked yet."
        text += " Counts come from automatic detection and can be wrong; tell me if one is (“photo 3 isn't a dog”)."
        ids, maybe = ids[:48], maybe[:24]
        return {"answer": text, "asset_ids": ids + maybe, "maybe_ids": maybe, "facts": facts, "box_labels": labels[:4]}

    def _search(self, plan, message, history, library_ids, use_chat, asset_id):
        q = plan.subject or message
        labels = resolve_labels(q)
        checked = len(labels) == 1 and self._coverage(library_ids)["checked"] > 0
        ids = self.search(q, library_ids, self.limit * 4 if checked else self.limit)
        if not ids:
            return {"answer": "Nothing is indexed yet. Add a folder on the left first.", "facts": {"query": q}}
        if checked:  # like the Search tab: photos where the detector found it first, the rest marked "maybe"
            label = labels[0]
            sure = set(self._without_corrections(
                label, self.store.count_objects(self.key, label, library_ids, sure=SURE)["asset_ids"]))
            found = [a for a in ids if a in sure][:self.limit]
            rest = [a for a in ids if a not in sure][:max(0, self.limit - len(found))]
            text = (f"{len(found)} of these have {display(label)} (found by the object detector), closest to “{q}” "
                    f"first." if found else f"The object detector didn't find {display(label)} in the closest matches.")
            if rest:
                text += " The ones marked “maybe” are only similar: low confidence."
            return {"answer": text, "asset_ids": found + rest, "maybe_ids": rest,
                    "facts": {"query": q, "label": label, "found": len(found)}, "box_labels": [label] if found else False}
        return {"answer": f"Here are the closest matches for “{q}”, best first. Similar isn't always a match, "
                          f"so the last ones may be off.", "asset_ids": ids, "facts": {"query": q}}

    def _refine(self, plan, message, history, library_ids, use_chat, asset_id):
        prev = self._previous(history)
        q = plan.subject or message
        colour = next((COLOUR_WORDS[w] for w in re.findall(r"[a-z]+", message.lower()) if w in COLOUR_WORDS), None)
        topic = resolve_label(q) or self._topic_label(message, history)
        if colour and topic and self.load_photo is not None and self._coverage(library_ids)["checked"]:
            return self._by_colour(prev, colour, topic, library_ids)
        label = resolve_label(q)
        if label is not None and self._coverage(library_ids)["checked"]:
            keep = set(self.store.count_objects(self.key, label, library_ids, sure=SURE)["asset_ids"])
            ids = self._without_corrections(label, [a for a in prev if a in keep])
            return {"answer": f"{len(ids)} of those {len(prev)} photos {'has' if len(ids) == 1 else 'have'} "
                              f"{display(label)}.",
                    "asset_ids": ids, "facts": {"label": label, "of": len(prev), "kept": len(ids)}, "box_labels": [label]}
        ids = self.rank_within(q, prev)
        if len(ids) > TOP_REFINE:
            return {"answer": f"Closest to “{q}”: the first {TOP_REFINE}. The rest, marked “maybe”, are low "
                              f"confidence.", "asset_ids": ids, "maybe_ids": ids[TOP_REFINE:],
                    "facts": {"query": q, "of": len(prev), "top": TOP_REFINE}}
        return {"answer": f"Here {'is that photo' if len(prev) == 1 else f'are those {len(prev)} photos'} again, closest to “{q}” first. I can't be sure which "
                          f"ones truly match, so have a look.", "asset_ids": ids, "facts": {"query": q, "of": len(prev)}}

    def _by_colour(self, prev: list[str], colour: str, label: str, library_ids) -> dict:
        """ "only the red ones": the share of each detected object's box in that colour (pixels, not a model)."""
        from PIL import Image

        sure = set(self.store.count_objects(self.key, label, library_ids, sure=SURE)["asset_ids"])
        test = COLOUR_TESTS[colour]
        scored = []
        for aid in prev:
            a = self.store.get_asset(aid)
            if aid not in sure or a is None:
                continue
            boxes = [b["box"] for b in self.store.boxes_for(a["content_hash"], self.key, [label]) if b["score"] >= SURE]
            try:
                im = self.load_photo(aid).convert("RGB")
            except Exception:
                continue
            best = 0.0
            for x0, y0, x1, y1 in boxes:
                crop = im.crop((int(x0 * im.width), int(y0 * im.height), int(x1 * im.width), int(y1 * im.height)))
                crop.thumbnail((128, 128))
                h, s_, v = (np.asarray(c, dtype=np.int16) for c in crop.convert("HSV").split())
                best = max(best, float(test(h, s_, v).mean()))
            scored.append((best, aid))
        scored.sort(key=lambda x: -x[0])
        keep = [a for f, a in scored if f >= COLOUR_SURE]
        maybe = [a for f, a in scored if COLOUR_MAYBE <= f < COLOUR_SURE]
        thing = display(label, 2)
        if keep:
            text = f"{len(keep)} of those have {colour} {thing}, checked by colour inside each detected {display(label, 1)}."
        else:
            text = f"None of those clearly have {colour} {thing}."
        if maybe:
            text += f" {len(maybe)} more {'is' if len(maybe) == 1 else 'are'} partly {colour}, marked “maybe”."
        return {"answer": text, "asset_ids": keep + maybe, "maybe_ids": maybe, "box_labels": [label],
                "facts": {"label": label, "colour": colour, "kept": len(keep), "maybe": len(maybe)}}

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
        objects = self.store.detections_for(asset["content_hash"], self.key, sure=SURE)
        facts = {"asset_id": asset_id, "file": asset["rel_path"], "objects": objects}
        if use_chat:
            q = message.strip() or "Describe this photo."
            system = "Describe the photo truthfully in 1-3 sentences. Only mention what you can see."
            if objects:  # the counts the user was shown; keeps the description consistent with them
                found = ", ".join(f"{n} {display(k, n)}" for k, n in objects.items())
                system += (f" Hint (from an object detector): the photo contains {found}. Mention them if you can see "
                           "them, small ones too, using the same names. Write as if you simply see the photo: never "
                           "mention the hint, a detector or exact counts.")
            try:
                text = self.chat.chat([{"role": "system", "content": system},
                                       {"role": "user", "content": q, "images": [self.image_b64(asset_id)]}],
                                      max_tokens=160)
                return {"answer": text, "asset_ids": [asset_id], "facts": facts, "described_by": "chat model",
                        "box_labels": None}
            except ChatUnavailable:
                pass
        found = ", ".join(f"{n} {display(k, n)}" for k, n in (objects or {}).items())
        text = (f"I found {found} in {asset['rel_path']}." if found else
                f"I didn't detect any common objects in {asset['rel_path']}." if objects is not None else
                f"{asset['rel_path']} hasn't been checked for objects yet.")
        return {"answer": text + " (Turn on the local chat model to get a written description.)",
                "asset_ids": [asset_id], "facts": facts, "box_labels": None}

    @staticmethod
    def _last_edit(history: list[Turn]) -> str | None:
        """The edited copy the conversation is about now (the latest answer, if it showed one)."""
        for t in reversed(history):
            if t.role == "assistant":
                return t.edit_id
        return None

    def _edit(self, plan, message, history, library_ids, use_chat, asset_id):
        """Work out which photo and which object; the edit itself runs as a job (api_ask.py) using `edit_request`."""
        labels = resolve_labels(plan.subject)
        verb = "replace" if plan.new else "remove"
        if not labels and plan.subject.lower() in PRONOUN_OBJ:  # "replace this with a lorry": the photo's main object
            labels = self._main_object(asset_id, plan, history)
            if not labels:
                return {"answer": f"Which thing should I {verb}? Say for example “{verb} the car"
                                  f"{' with ' + plan.new if plan.new else ''}”.", "facts": {}}
        if not labels:
            if re.search(r"\b(photo|picture|image|pic|file|one)s?\b|^\s*#?\d+\s*$", plan.subject, re.I):
                return {"answer": "MediaIndex never deletes or changes your photos. To take something out of a photo, "
                                  "say what: “remove the car from photo 2”.", "facts": {}}
            return {"answer": f"I can only {verb} things the object detector recognises, like people, cars, dogs, "
                              f"cats, birds, bikes, boats or cups, not “{plan.subject}” yet.",
                    "facts": {"subject": plan.subject}}
        source: dict | None = None
        prev = self._previous(history)
        last_edit = self._last_edit(history)
        if asset_id:
            source = {"asset_id": asset_id}
        elif plan.item is not None and prev:
            n = len(prev) if plan.item == -1 else plan.item
            if not 1 <= n <= len(prev):
                return {"answer": f"The last answer had {len(prev)} photos; pick a number from 1 to {len(prev)}.",
                        "facts": {}}
            source = {"asset_id": prev[n - 1]}
        elif last_edit:  # "now remove the person too": keep editing the copy just made
            source = {"edit_id": last_edit}
        elif len(focus := self._focus(history)) == 1:
            source = {"asset_id": focus[0]}
        elif focus:
            return {"answer": f"Which photo? Say for example “{verb} the {display(labels[0], 1)} "
                              f"{'with ' + plan.new + ' ' if plan.new else ''}in photo 2”.", "asset_ids": focus,
                    "facts": {}}
        if source is None:
            return {"answer": f"First find the photo (for example “show me photos with {display(labels[0])}”), "
                              f"then tell me which one to edit.", "facts": {}}
        if "asset_id" in source:
            a = self.store.get_asset(source["asset_id"])
            if a is None or a["media_type"] != "image":
                return {"answer": "I can only edit photos.", "facts": {}}
        everything = bool(ALL_RE.search(plan.subject) or ALL_RE.search(message)) or is_plural(plan.subject)
        thing = display(labels[0], 2 if everything else 1)
        doing = f"Replacing the {thing} with {plan.new}" if plan.new else f"Removing the {thing}"
        return {"answer": f"{doing}… This makes a new copy; your original photo stays as it is.",
                "asset_ids": [source["asset_id"]] if "asset_id" in source else [],
                "edit_request": {**source, "label": labels[0], "all": everything, "new": plan.new, "big": plan.big,
                                 "seed": plan.seed},
                "facts": {"label": labels[0], "all": everything, "new": plan.new}}

    def _main_object(self, asset_id, plan, history) -> list[str]:
        """The biggest object the detector is sure of in the photo being talked about."""
        prev, focus = self._previous(history), self._focus(history)
        aid = asset_id or (prev[(len(prev) if plan.item == -1 else plan.item) - 1]
                           if plan.item is not None and prev and 1 <= (len(prev) if plan.item == -1 else plan.item)
                           <= len(prev) else focus[0] if len(focus) == 1 else None)
        a = self.store.get_asset(aid) if aid else None
        if a is None:
            return []
        boxes = [b for b in self.store.boxes_for(a["content_hash"], self.key) if b["score"] >= SURE]
        if not boxes:
            return []
        big = max(boxes, key=lambda b: (b["box"][2] - b["box"][0]) * (b["box"][3] - b["box"][1]))
        return [big["label"]]

    @staticmethod
    def _last_edit_ask(history: list[Turn]) -> tuple[int, Plan] | None:
        """The latest user message that asked for an edit, and its plan."""
        for i in range(len(history) - 1, -1, -1):
            t = history[i]
            if t.role == "user" and (p := edit_plan(t.text)) is not None:
                return i, p
        return None

    def follow_up_edit(self, message: str, history: list[Turn], library_ids, use_chat) -> dict | None:
        """ "go ahead" after a too-big refusal, or "put a truck instead" after a replacement: the same edit again, on
        the same photo as before (never on the copy it made), with the change asked for."""
        last = next((t for t in reversed(history) if t.role == "assistant"), None)
        if last is None or last.action != "edit" or (found := self._last_edit_ask(history)) is None:
            return None
        i, p = found
        if TOO_BIG_RE.search(last.text) and CONFIRM_RE.search(message):
            p.big = True
        elif (new := self._redo_new(message)) and (p.new or any(REDO_WITH_RE.match(x) for x in _parts(message))):
            p.new = new  # after a replacement: something else; after a removal: "replace them with a table instead"
        elif not p.new and RETRY_RE.search(message) and not edit_plan(message):
            p.seed = 1 + sum(1 for t in history[i + 1:] if t.role == "user")  # each "try again" a new picture
        else:
            return None
        before = history[:i]
        asked = history[i]
        out = self._edit(p, asked.text, before, library_ids, use_chat, asked.asset_ids[0] if len(asked.asset_ids) == 1
                         else None)
        out["plan"] = p
        return out

    @staticmethod
    def _redo_new(message: str) -> str:
        """ "wait no not that, replace with a tesla cybertruck" -> "a tesla cybertruck" (any phrase may say it)."""
        for part in reversed(_parts(message)):
            if (r := REDO_WITH_RE.match(part) or REDO_RE.match(part)):
                return clean_new(r.group("new"))
        return ""

    def _generate(self, plan, message, history, library_ids, use_chat, asset_id):
        """A new image from words; made up by the image model, so the answer says it isn't one of your photos."""
        what = plan.subject.strip()
        return {"answer": f"Making a new image of “{what}”… It is made up by a local image model, not one of your photos.",
                "create_request": {"op": "generate", "prompt": what, "shape": plan.new or "square"},
                "facts": {"prompt": what}}

    def _video(self, plan, message, history, library_ids, use_chat, asset_id):
        """A slideshow of exactly the photos asked for: the last answer's, or the closest matches for a subject."""
        if plan.subject:
            ids = self.search(plan.subject, library_ids, self.limit)
            source = f"the {len(ids)} closest matches for “{plan.subject}”"
        else:
            ids = self._previous(history) or self._focus(history)
            source = f"{'this photo' if len(ids) == 1 else f'these {len(ids)} photos'}"
        if not ids:
            return {"answer": "Which photos? Ask for some first (“show me beach photos”), then say “make a video of "
                              "those”, or name them: “make a video of my beach photos”.", "facts": {}}
        shape = {"vertical": "a vertical (9:16)", "square": "a square"}.get(plan.new, "a widescreen (16:9)")
        note = " Similar isn't always a match, so check the photos below." if plan.subject else ""
        return {"answer": f"Making {shape} video from {source}, in this order…{note}",
                "asset_ids": ids, "create_request": {"op": "video", "asset_ids": ids, "shape": plan.new or "landscape"},
                "facts": {"photos": len(ids), "shape": plan.new}}

    def _explain(self, plan, message, history, library_ids, use_chat, asset_id):
        """Show the evidence for an earlier answer: where the detector found the object, and how sure it was."""
        focus = self._focus(history)
        label = self._topic_label(message, history)
        if not focus:
            return {"answer": "The last answer didn't show any photos, so there's nothing to check.", "facts": {}}
        nums = _items(message)
        if nums and label is not None and len(focus) > 1 and NEGATION_RE.search(message):
            return self._correct(focus, nums, label)
        if len(focus) > 1:
            if label is None:
                return {"answer": "Those photos came from similarity search, which ranks photos without checking "
                                  "what is in them, so some may not match.", "asset_ids": focus, "facts": {}}
            return {"answer": f"I've outlined each {display(label, 1)} the detector found in these photos, with how sure "
                              f"it was. If one is wrong, tell me its number (“photo 3 isn't a {display(label, 1)}”) and "
                              f"I'll leave it out from now on.",
                    "asset_ids": focus, "facts": {"label": label}, "box_labels": [label]}
        asset = self.store.get_asset(focus[0])
        boxes = self.store.boxes_for(asset["content_hash"], self.key, [label] if label else None)
        if label is None:
            text = ("Here's everything the detector found in this photo, outlined." if boxes else
                    "The detector found no common objects in this photo.")
        elif not boxes:
            text = (f"The detector didn't find a {display(label, 1)} in this photo. If my description said "
                    f"otherwise, that came from the chat model, and it can be wrong.")
        else:
            best = max(b["score"] for b in boxes)
            size = max((b["box"][2] - b["box"][0]) * (b["box"][3] - b["box"][1]) for b in boxes)
            text = (f"The detector found {len(boxes)} {display(label, len(boxes))} here, outlined in the photo "
                    f"(confidence {best:.2f}{', sure' if best >= SURE else ', not sure: a “maybe”'}).")
            if size < 0.02:
                text += f" {'It is' if len(boxes) == 1 else 'They are'} small, so look closely. Detection can still be wrong."
        return {"answer": text, "asset_ids": [asset["id"]], "facts": {"label": label, "boxes": boxes},
                "box_labels": [label] if label else None}

    def _correct(self, focus: list[str], nums: list[int], label: str) -> dict:
        """The user says photos `nums` of the last answer don't contain `label`: remember that, for good."""
        picks = sorted({len(focus) if n == -1 else n for n in nums})
        bad = [n for n in picks if not 1 <= n <= len(focus)]
        if bad:
            return {"answer": f"The last answer had {len(focus)} photos; pick numbers from 1 to {len(focus)}.", "facts": {}}
        for n in picks:
            self.store.save_verification(self.store.get_asset(focus[n - 1])["content_hash"], label, USER, False)
        rest = [a for i, a in enumerate(focus, 1) if i not in picks]
        which = ", ".join(map(str, picks[:-1])) + (" and " if len(picks) > 1 else "") + str(picks[-1])
        return {"answer": f"Thanks, I've noted that photo{'s' if len(picks) > 1 else ''} {which} "
                          f"{'have' if len(picks) > 1 else 'has'} no {display(label, 1)}. "
                          f"{'They are' if len(picks) > 1 else 'It is'} left out of {display(label)} answers from now on.",
                "asset_ids": rest, "facts": {"removed": [focus[n - 1] for n in picks], "label": label},
                "box_labels": [label]}

    def _summary(self, plan, message, history, library_ids, use_chat, asset_id):
        """What's in the library: every number comes from the database, never from the chat model."""
        cov = self._coverage(library_ids)
        if cov["photos"] == 0:
            return {"answer": "There are no indexed photos yet. Add a folder on the left first.", "facts": {"coverage": cov}}
        if cov["checked"] == 0:
            return {"answer": f"You have {cov['photos']} photos. Press “Count objects” above to see what's in them.",
                    "facts": {"coverage": cov}}
        top = self.store.object_summary(self.key, library_ids, sure=SURE)[:8]
        text = f"You have {cov['photos']} photos."
        if top:
            text += " Most common in them: " + ", ".join(
                f"{display(t['label'])} ({t['photos']} {'photo' if t['photos'] == 1 else 'photos'})" for t in top) + "."
        else:
            text += " I didn't find any of the common objects I can count."
        if cov["checked"] < cov["photos"]:
            text += f" {cov['photos'] - cov['checked']} newer photos haven't been checked yet."
        examples = []  # one example photo for each of the most common things
        for t in top[:6]:
            ids = self._without_corrections(t["label"],
                                             self.store.count_objects(self.key, t["label"], library_ids, sure=SURE)["asset_ids"])
            if (pick := next((a for a in ids if a not in examples), None)) is not None:
                examples.append(pick)
        return {"answer": text, "asset_ids": examples, "facts": {"coverage": cov, "top": top},
                "box_labels": [t["label"] for t in top[:6]]}

    def _chat(self, plan, message, history, library_ids, use_chat, asset_id):
        cov = self._coverage(library_ids)
        summary = self.store.object_summary(self.key, library_ids, sure=SURE)
        fixed = ("I can count things in your photos (“how many dogs?”), find photos (“beach at sunset”), narrow "
                 "down the last answer (“which of those have people?”) and describe a photo (“describe the "
                 "second one”).")
        if summary:
            fixed += " Most common in your photos: " + ", ".join(
                f"{display(s['label'])} ({s['photos']} {'photo' if s['photos'] == 1 else 'photos'})"
                for s in summary[:5]) + "."
        if not use_chat:
            return {"answer": fixed, "facts": {"coverage": cov}}
        facts = {"photos": cov["photos"]}
        system = ("You are the assistant inside MediaIndex, a private photo search app that runs on this computer. "
                  "You can: count common objects in photos (\"how many dogs?\"), give an overview (\"what's in my "
                  "library?\"), find photos by description, narrow down the last results, and describe a photo. "
                  "Answer in 1-3 short sentences. Don't state any numbers or say what is in the photos: for that, "
                  "suggest one of those questions instead.")
        msgs = [{"role": "system", "content": system}]
        msgs += [{"role": t.role, "content": t.text} for t in history[-4:] if t.role in ("user", "assistant")]
        msgs.append({"role": "user", "content": message})
        try:
            text = self.chat.chat(msgs, max_tokens=160)
        except ChatUnavailable:
            return {"answer": fixed, "facts": facts}
        allowed = set(re.findall(r"\d+", message))  # the chat model gets no numbers, so it may not state any
        if any(n not in allowed for n in re.findall(r"\d+", text)):
            return {"answer": fixed, "facts": facts, "note": "chat reply replaced: it mentioned numbers"}
        return {"answer": text, "facts": facts}
