"""Prompts for MiniMax H3's reference mode (ref2va), in the six sections its prompt rewriter
uses: subject_definitions, summary, retention_analysis, detailed_description,
overall_soundscape, non_diegetic_music.

Labels: <Subject N> = something to reuse (a person from their pictures, the room);
<Picture N> = an input image. <Picture 1> is the scene as the viewer frames it (with or
without mannequins); a picture that only shows a person is cited inside that person's
subject line, not given its own line. Shots are
[Shot 1], ...; speakers get (S1), (S2) in the order they first speak, and their words go in
<d>[Language] ...</d>.
"""
from __future__ import annotations

import re

MAX_VOICES = 3  # MiniMaxH3ReferenceToVideo takes at most 3 standalone audio references
SAY_VERBS = "says|asks|shouts|whispers|replies|exclaims|calls|answers|laughs|adds|continues"
# Speech is tagged English; a prompt that already carries <d>[Language] ...</d> tags is left
# as written, so other languages are tagged by hand.
SPEECH_LANGUAGE = "English"


def _sentence(text: str) -> str:
    text = " ".join(str(text or "").split()).strip()
    return text if not text or text[-1] in ".!?" else text + "."


def _tags(tags: list[str]) -> str:
    return tags[0] if len(tags) == 1 else f"{', '.join(tags[:-1])} and {tags[-1]}"


# Speakers that are not in the cast: 'She says', 'the man asks'.
OTHER_SPEAKERS = r"[Ss]he|[Hh]e|[Tt]hey|(?:[Tt]he|[Aa]n?)\s+(?:young\s+|old\s+|older\s+)?(?:woman|man|girl|boy|person)"


def dialogue(text: str, names: list[str], speakers: set | None = None) -> str:
    """'Emma says: "Hi there"' -> 'Emma (S1) says, <d>[English] Hi there.</d>'. Speakers are
    numbered by first line spoken (cast names, then 'she' / 'the man' for people not in the
    cast; 'she' / 'he' after a cast name is that person); text already in <d> is left alone.
    speakers, if given, collects the cast names that speak."""
    if not text or "<d>" in text:
        return text
    ids: dict[str, int] = {}
    cast = "|".join(re.escape(n) for n in sorted(names, key=len, reverse=True))
    alt = (f"(?i:{cast})|" if cast else "") + OTHER_SPEAKERS
    # 'Emma says softly: "..."': up to three words between the verb and the quote; 'Emma sits
    # down, smiles and says: "..."': up to 80 characters of the same sentence before the verb.
    pattern = re.compile(rf"\b({alt})\b([^.!?\"“„]{{0,80}}?\s)?((?:{SAY_VERBS})(?:\s+[\w-]+){{0,3}}?)\s*[,:]?\s*"
                         rf"[\"“„]([^\"”“]+)[\"”]")

    named = re.compile(rf"\b(?i:{cast})\b") if cast else None

    def repl(m):
        who = m.group(1)
        key = next((n for n in names if n.lower() == who.lower()), " ".join(who.lower().split()))
        if key in ("she", "he", "they"):
            # 'Emma smiles. She says ...': the cast member named last before the pronoun.
            last = list(named.finditer(text, 0, m.start())) if named and key != "they" else []
            key = next((n for n in names if last and n.lower() == last[-1].group(0).lower()), "pronoun " + key)
        if speakers is not None and key in names:
            speakers.add(key)
        sid = ids.setdefault(key, len(ids) + 1)
        words = _sentence(m.group(4))
        lang = SPEECH_LANGUAGE
        middle = f" {m.group(2).strip()}" if (m.group(2) or "").strip() else ""
        return f"{who} (S{sid}){middle} {m.group(3).strip()}, <d>[{lang}] {words}</d>"
    return pattern.sub(repl, text)


def build(people: list[dict], places: list[str], detail: str, *, seconds: float = 0.0, soundscape: str = "",
          music: str = "", triggers: str = "", marked: bool = False, angles: list[str] = ()) -> str:
    """A ref2va clip prompt. <Picture 1> is the scene as the viewer shows it (a reference for
    the place and the framing, not a first frame); people: [{"name", "tags": ["<Picture 2>",
    ...], "description"}] in picture order follow it. marked = coloured mannequins in
    <Picture 1> mark where people stand; without them the prompt alone places everyone.
    A person with "voice": True gets "<Audio k> is Name's voice" when they speak.
    detail = the detailed description as written (a style sentence, then [Shot 1] ...), with
    people by name and speech in quotes; soundscape / music = those sections as written.
    angles = the tags of pictures of the place from other angles (a SceneRef Scene)."""
    subjects = ["<Subject 1> is the scene in <Picture 1>: the room, its furniture, wall art and light, and the "
                "camera view and framing." + (" The plain coloured mannequins in <Picture 1> only mark where the "
                                              "people stand; they are not in the video." if marked else "")
                + (f" {_tags(list(angles))} {'shows' if len(angles) == 1 else 'show'} the same place from "
                   f"{'another angle' if len(angles) == 1 else 'other angles'}; the camera stays as in <Picture 1>." if angles else "")]
    retention = ["<Subject 1> (appears in [Shot 1]): fully_preserved - the room, furniture, wall art and framing "
                 "stay as in <Picture 1>."]
    n = 2
    label = {}
    for p in people:
        lab = f"<Subject {n}>"
        label[p["name"]] = lab
        desc = _sentence(p.get("description") or "")
        desc = desc[:1].upper() + desc[1:]
        subjects.append(f"{lab} is {p['name']}, the person in {_tags(p['tags'])}: the same face, age, hair and "
                        f"build." + (f" {desc}" if desc else ""))
        retention.append(f"{lab} (appears in [Shot 1]): fully_preserved - {p['name']}'s identity, face, age, hair "
                         f"and build follow {_tags(p['tags'])}.")
        n += 1

    who = " and ".join(label[p["name"]] for p in people)
    shots = len(re.findall(r"\[Shot \d+\]", str(detail or ""))) or 1
    summary = ("[reference generation] The target video is "
               + ("a single continuous shot" if shots == 1 else f"{shots} shots")
               + (f" of about {seconds:.1f} seconds" if seconds else "") + " in <Subject 1>, seen as in <Picture 1>"
               + (f", with {who} as {'real people' if len(people) > 1 else 'a real person'}, recognisable from "
                  f"{'their pictures' if len(people) > 1 else 'the pictures'}." if who else "."))

    names = [p["name"] for p in people]
    body = " ".join(str(detail or "").split())
    # The style sentence opens the description, before [Shot 1]. The model sometimes writes it
    # after [Shot 1], or leaves [Shot 1] out.
    m = re.match(r"(?:\[Shot 1\]\s*)?(The target video (?:is|uses)[^.]*\.)\s*", body)
    look, body = (m.group(1), body[m.end():]) if m else ("", body)
    if not re.search(r"\[Shot 1\]", body):
        body = "[Shot 1] " + body
    speakers: set = set()
    body = dialogue(" ".join(x for x in (look, body) if x), names, speakers)
    # Written already in H3's form ('Emma (S1) says, <d>...'): the speakers from their tags.
    speakers |= {n for n in names if re.search(rf"\b{re.escape(n)} \(S\d+\)", body)}
    # A voice recording is cited (OmniChar's wording) for each person with one who speaks;
    # numbered in cast order, as SceneRefStudio wires them (only the tags left in the prompt).
    k = 0
    for i, p in enumerate(people):
        if p.get("voice"):
            k += 1
            if k <= MAX_VOICES and p["name"] in speakers:
                subjects[i + 1] += (f" <Audio {k}> is {p['name']}'s voice. {p['name']} speaks in this voice, lips "
                                    f"moving in sync with every word.")
    # Where each person is goes at the start of the first shot, after the style sentence.
    where = " ".join(places)
    m = re.search(r"\[Shot 1\]\s*", body)
    body = body[:m.end()] + (where + " " if where else "") + body[m.end():]
    # People by their labels, as the subject lines define them; spoken words stay as said.
    parts = re.split(r"(<d>.*?</d>)", body)
    for i in range(0, len(parts), 2):
        for name, lab in label.items():
            parts[i] = re.sub(rf"\b{re.escape(name)}\b", lab, parts[i])
    description = " ".join(x for x in ("".join(parts), _sentence(triggers)) if x)

    soundscape = " ".join(str(soundscape or "").split()) or (
        "Natural room tone of the scene, with the soft physical sounds of the people's movements and voices as "
        "described.")
    return "\n\n".join([
        "subject_definitions:\n" + "\n".join(subjects),
        "summary:\n" + summary,
        "retention_analysis:\n" + "\n".join(retention),
        "detailed_description:\n" + description,
        "overall_soundscape:\n" + soundscape,
        "non_diegetic_music:\n" + (" ".join(str(music or "").split()) or "N/A"),
    ])
