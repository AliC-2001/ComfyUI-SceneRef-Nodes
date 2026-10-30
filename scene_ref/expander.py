"""SceneRef Prompt Expander - a small vision model turns a short action into scene directions.

It looks at the shot (the composite, with the coloured figures), knows who plays which
figure, and writes what each person does, their expression and how they interact; on
request it adds room contents and decoration. The text goes into the Studio's
`action_text`. Off = the action passes through unchanged and no model is loaded.
The model is unloaded right after, so the image model gets the VRAM back.

write_prompt is the Studio's 'Write the prompt' button: the whole prompt for the chosen
backend, written before Run into the Studio's prompt box.
"""
from __future__ import annotations

import math

from . import scenes
from .characters import CAST_TYPE, FIGURE_COLORS, resolve_refs
from .nodes import SHOT_TYPE
from .studio import _place_phrase
from .vlm import AUTO, pick_encoder, vlm_files, vlm_text

VISION_MEGAPIXELS = 0.4

SYSTEM = (
    "You write short scene directions for an image generator. The picture is a 3D layout of a room; "
    "plain coloured mannequins mark where the people are. Describe the finished photograph: for every person, "
    "what they are doing, their expression and body language, and how they interact with each other and with "
    "the things around them. Every person stays exactly where they are and keeps exactly the pose given "
    "(sitting stays sitting, standing stays standing, no change of place, no leaning over to someone): if "
    "they interact from a distance, describe it that way (looking at, talking to, holding something out "
    "towards). Use every name exactly as given. Never mention mannequins, figures, colours of the figures, 3D, "
    "renders, layouts or the camera. Do not describe anyone's face, hair, age or body: reference photos show "
    "those. Keep every detail the user wrote (clothing, objects, mood) and repeat it. "
    "Never add, remove or change windows, doors, walls, pictures, furniture or the room itself, and do not add "
    "people. {room}Reply with the text only: plain sentences, at most {words} words."
)
ROOM_ON = ("Then add one or two sentences of small things that fit the room (for example plants, books, a lamp, "
           "cushions, a blanket, cups, small objects on tables and shelves), placed on the existing surfaces, and "
           "describe the light without inventing new windows. ")
ROOM_OFF = "Do not add objects. "

# For a MiniMax H3 clip that starts from the finished picture: playback order, camera, sound
# and speech in the forms H3's reference-mode prompts use ((S1), <d>[Language] ...</d>).
SYSTEM_VIDEO = (
    "You write the shot description of a short video for a video generator. The picture is a 3D layout of the "
    "scene the video plays in; plain coloured mannequins mark where the people are. Describe, in playback order, one "
    "continuous shot: what each person does and how it changes (first, then, by the end), their expressions and "
    "gestures, how they interact, small movements of objects, cloth and light, the camera (static unless the "
    "user asks for a move; name the move, its direction and speed), and the sounds as they happen (room tone, "
    "footsteps, cups, laughter). Everyone starts exactly where they are, in the pose given, and moves only as "
    "far as the action needs. Speech: the speaker's name, a verb and the words in double quotes, in the "
    "language they are said in, like: Emma says softly: \"Are you coming?\" Every quoted line the user wrote "
    "is said exactly once, word for word. Keep every detail the user wrote. Use every name exactly as given. Never mention mannequins, "
    "figures, colours of the figures, 3D, renders or layouts. Do not describe anyone's face, hair, age or body: "
    "reference photos show those. {room}Reply with the text only: plain sentences, at most {words} words."
)
ROOM_VIDEO_ON = "You may mention small things already in the room that move or make sound. "
MARKED = "plain coloured mannequins mark where the people are."
# Reference pictures often show people in other clothes than the scene needs, and MiniMax H3
# copies what they wear when the prompt names no outfit (or the vision model drops the one the
# action named). FLUX3_KEEP has the same rule.
DRESS = ("Dress everyone: name each person's complete outfit, the clothes the user wrote word for word, else "
         "ordinary clothes that fit the place and the weather; the reference pictures do not decide the clothes, "
         "and nobody is naked unless the user asks for it. ")
UNMARKED = ("nobody's place is marked in it: put each person where the user's text says, in a natural spot in "
            "this room.")
# A scene from pictures (SceneRef Scene): the 3D wording of the texts above, as for a photo.
PHOTO_WORDS = {
    "The picture is a 3D layout of a room; ": "The picture shows the place; ",
    "The picture is a 3D layout of the scene the video plays in; ": "The picture shows the place the video plays in; ",
    "The picture is a 3D layout of the scene (a render of the room, its furniture and light); ":
        "The picture shows the place the scene happens in (its layout, furniture and light); ",
    "If the user names no visual style, it is a realistic photograph.":
        "If the user names no visual style, keep the look of the picture.",
}

# ── the Studio's 'Write the prompt' ─────────────────────────────────────────────
# The model writes the scene part of the prompt; the Studio adds which picture is what and
# who replaces which mannequin (those must be exact), and the LoRA trigger words. Each
# backend's part follows its own prompt guide.
STUDIO_RULES = (
    "The picture is a 3D layout of the scene (a render of the room, its furniture and light); "
    + MARKED + " The lines that say which reference picture shows whom and which mannequin becomes whom are "
    "added separately: do not write them, do not mention images, pictures or references, and refer to people by "
    "name only. Every person stays exactly where they are and keeps exactly the pose given (sitting stays sitting, "
    "standing stays standing): if they interact from a distance, describe it that way (looking at, talking to, "
    "holding something out towards). Keep the camera angle and framing of the layout. Never mention mannequins, "
    "figures, colours of the figures, 3D, renders or layouts. Do not describe anyone's face, hair, age or body: "
    "the reference pictures show those. Keep every detail the user wrote (clothing, objects, light, time of day, "
    "mood, style) and repeat it. " + DRESS + "Never add, remove or change windows, doors, walls, wall pictures, "
    "furniture or the room itself, and do not add people. If the user names no visual style, it is a realistic "
    "photograph. {room}Reply with the prompt text only, no title, no explanation, at most {words} words."
)
GUIDES = {
    # BFL's FLUX.2 [klein] notes: prose like a novelist, named light, a closing style line.
    "klein": (
        "You write the scene description of a prompt for FLUX.2 Klein, an image model that reads prose, not "
        "keyword lists: what you write is exactly what it receives. Describe the scene like a novelist: the "
        "people and what they do first, their expressions and how they interact, then the setting. Describe the "
        "light by its source, direction and softness. End with one line: 'Style: <the visual style>. Mood: "
        "<the mood>.' "
    ),
    # Qwen-Image 2.1's prompt rewriter, shortened.
    "qwen": (
        "You write the scene description of a prompt for Qwen-Image 2.1. Lead with the people and their key "
        "action, pose or relationship, then the setting, the essential secondary elements, the visual style and "
        "the atmosphere. State relationships literally and early (on, under, beside, behind, holding, wearing, "
        "looking at). Keep every explicit fact: counts, colours, materials, clothing, and any visible text, "
        "word for word in double quotes. For a short idea add restrained details that fit it (light, texture, "
        "atmosphere); never invent people, objects or an aesthetic that contradicts it. One continuous paragraph "
        "of fluent natural English, no lists or headings, under 200 words. "
    ),
    # The Krea 2 prompt skill: named surfaces, named light, one off-centre detail, no slop.
    "krea2": (
        "You write the scene description of a prompt for Krea2, a prose model trained on dense captions. Name "
        "the materials and surfaces concretely (raw linen, worn oak, chipped enamel, brushed steel) and name the "
        "light and what it does (its source, its direction, what it catches, the shadows it throws). Add exactly "
        "one small, specific, observed-looking detail (a coffee ring on a table, one sleeve rolled higher than "
        "the other). Give hands a clear, single action. No quality words (beautiful, stunning, masterpiece, 8k, "
        "ultra-detailed, cinematic) and no negations: describe what is there instead. One paragraph, 80 to 180 "
        "words. "
    ),
}
# MiniMax H3's reference-mode prompt guide (ref-en): the model writes the three free sections;
# subject_definitions, summary and retention_analysis are built from the cast (exact labels).
MOVIE_GUIDE = (
    "You write part of a prompt for MiniMax H3, which makes a video with sound from reference pictures. The "
    "picture shows the scene the video plays in; " + MARKED + " The lines that say which reference picture shows "
    "whom are added separately: refer to people by name only and never mention images, pictures or references. "
    "Reply in exactly this form, with these three headings:\n"
    "detailed_description:\n"
    "<One or two sentences that set the visual style, like: The target video is in a realistic live-action style "
    "with warm indoor lighting and a slightly desaturated palette.> [Shot 1] <the shot>\n"
    "overall_soundscape:\n"
    "<1 to 4 sentences: the ambience and physical sounds through the whole video (room tone, footsteps, fabric, "
    "cups, breathing); no dialogue>\n"
    "non_diegetic_music:\n"
    "<1 to 3 sentences on background music only the audience hears: instruments, tempo, how it develops; or "
    "N/A if the user wants none or names none>\n\n"
    "The video lasts {seconds} seconds. In detailed_description describe, in playback order, the composition, "
    "where each person is and what they do and how it changes, their expressions and gestures, how they "
    "interact, small movements of objects, cloth and light, the camera and the sounds as they happen. Camera: "
    "a static shot unless the user asks for a move; write a move as a natural sentence with its type, size and "
    "speed (the camera pushes in with small amplitude at slow speed). One continuous shot unless the user asks "
    "for cuts; a later shot starts with [Shot 2] At 00:03.000, the shot cuts to ... and must start before the "
    "end. Speech: write the speaker's name, a verb and the words in double quotes in the language they are said "
    "in, like: Emma says softly: \"Are you coming?\" then describe the voice (pitch, tone, pace) in the next "
    "sentence. Every quoted line the user wrote is said exactly once, word for word. Concrete visual and sound "
    "details, not words like cinematic or beautiful. Everyone starts exactly where they are, in the pose given, "
    "and moves only as far as the action needs. Never mention mannequins, figures, colours of the figures, 3D, "
    "renders or layouts. Do not describe anyone's face, hair, age or body: the reference pictures show those. "
    "Keep every detail the user wrote (clothing, objects, light, time of day, mood, style). " + DRESS + "If the "
    "user names no visual style, it is realistic live action. {room}detailed_description has about {words} words."
)
SECTIONS = ("detailed_description", "overall_soundscape", "non_diegetic_music")


def people_lines(figures: list, cast: list, size: tuple = (0, 0)) -> list[str]:
    by_figure = {c.get("figure"): c for c in cast or [] if c.get("figure")}
    lines = []
    for f in figures or []:
        idx = int(f.get("index") or 0)
        color = f.get("color") or FIGURE_COLORS[(idx - 1) % len(FIGURE_COLORS)]
        ch = by_figure.get(idx)
        who = ch["name"] if ch else {"male": "a man", "female": "a woman"}.get(f.get("sex"), "a person")
        pose = str(f.get("pose_label") or f.get("pose") or "").split("(")[0].strip().lower()
        line = f"- {who} (the {color} figure), {_place_phrase(f, *size)}" + (
            f", pose: {pose}" if pose and "daz" not in pose else "")
        text = (ch.get("description") or ch.get("file_description") or "") if ch else ""
        if text:
            line += f". {' '.join(text.split())[:300]}"
        lines.append(line)
    # Characters not tied to a figure are still part of the story.
    for ch in cast or []:
        if not ch.get("figure"):
            text = ch.get("description") or ch.get("file_description") or ""
            lines.append(f"- {ch['name']}: {' '.join(text.split())[:300]}")
    return lines


def _small(image, megapixels: float = VISION_MEGAPIXELS):
    import comfy.utils
    img = image[:1]
    h, w = img.shape[1], img.shape[2]
    scale = math.sqrt(megapixels * 1e6 / (w * h))
    if scale >= 1:
        return img
    nw, nh = max(28, round(w * scale / 28) * 28), max(28, round(h * scale / 28) * 28)
    return comfy.utils.common_upscale(img.movedim(-1, 1), nw, nh, "area", "disabled").movedim(1, -1)


class SceneRefPromptExpander:
    DESCRIPTION = (
        "Optional: a small vision model (Qwen3-VL 8B) looks at the shot and turns your short action into "
        "fuller scene directions: what each person does, expressions, interaction, and (if you want) room "
        "contents and decoration. Wire 'text' into the Studio's action_text. Off = your action passes "
        "through unchanged. The model is unloaded afterwards."
    )
    CATEGORY = "SceneRef"
    FUNCTION = "run"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    OUTPUT_TOOLTIPS = ("The expanded action (or the action as typed, when off).",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "enabled": ("BOOLEAN", {"default": True, "label_on": "expand", "label_off": "off (pass through)"}),
                "action": ("STRING", {"multiline": True, "default": "", "tooltip": "What happens, short: "
                           "'Emma reads on the couch, Jack brings her tea.'"}),
                "add_room_details": ("BOOLEAN", {"default": True, "tooltip": "Let it add decoration and small "
                                     "objects (plants, books, lamps...) and describe the light."}),
                "max_words": ("INT", {"default": 110, "min": 20, "max": 400, "step": 10}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "control_after_generate": True}),
                "text_encoder": ([AUTO] + vlm_files(), {"default": AUTO, "tooltip": "The vision language "
                                 "model. Only files that can write text are listed (Qwen3-VL with its lm_head, "
                                 "from models/text_encoders or models/LLM)."}),
                "unload_after": ("BOOLEAN", {"default": True, "tooltip": "Free the model's VRAM as soon as "
                                 "the text is written."}),
            },
            "optional": {
                "shot": (SHOT_TYPE, {"tooltip": "The shot of the Scene Reference Viewer or SceneRef Scene: the "
                                     "model looks at the first picture."}),
                "cast": (CAST_TYPE, {"tooltip": "Who plays which figure."}),
            },
        }

    def run(self, enabled, action, add_room_details, max_words, seed, text_encoder, unload_after,
            shot=None, cast=None):
        action = resolve_refs(action, cast or [])
        text = expand(action, shot, cast, add_room_details, max_words, seed, text_encoder, unload_after)             if enabled else action.strip()
        return {"ui": {"text": [text]}, "result": (text,)}


def _sections(text: str) -> dict:
    """'detailed_description: ... overall_soundscape: ...' -> {name: text}; without headings
    everything is the detailed description. The model does not always start a heading on a
    line of its own ('... her breathing. overall_soundscape: ...'), so they are found anywhere."""
    import re

    marks = list(re.finditer(rf"\**\b({'|'.join(SECTIONS)})\b\**\s*:\s*", text, re.I))
    if not marks:
        return {"detailed_description": text.strip()}
    out = {}
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        out[m.group(1).lower()] = text[m.end():end].strip()
    return out


def write_prompt(action, shot, cast, *, backend, movie=False, room_details=True, text_encoder=AUTO, triggers="",
                 control_strength=0.75, image_aspect=None, movie_aspect=None, video_seconds=5.0,
                 upsampler=AUTO) -> str:
    """The Studio's 'Write the prompt': the complete prompt, as the Run will use it. The
    vision model writes the scene from the action, following the backend's prompt guide; the
    picture references, the mannequin lines (Klein / Qwen / Krea2) or the subject sections
    (H3) and the trigger words are added around it. The result goes into the prompt box, to
    be edited before Run."""
    from . import h3prompt
    from .studio import (KLEIN, KLEIN3, KREA2, QWEN, aspect_crop, build_prompt, h3_frames, H3_FPS, join_passes,
                         krea2_prompts, movie_layout, _shift_figures)

    figures = ((shot or {}).get("figures") or [[]])[0] or []
    cast = scenes.plain_cast(shot, cast)
    action = resolve_refs(action, cast)
    photo = scenes.is_pictures(shot)
    sw, sh = int(shot["width"]), int(shot["height"])
    left, top, cw, ch_ = aspect_crop(sw, sh, movie_aspect if movie else image_aspect)
    if (cw, ch_) != (sw, sh):
        figures = _shift_figures(figures, left, top)
    if movie:
        seconds = h3_frames(video_seconds) / H3_FPS
        text = expand(action, shot, cast, room_details, max_words=350, text_encoder=text_encoder, mode="video",
                      guide=MOVIE_GUIDE, seconds=seconds)
        parts = _sections(text)
        angles = len(scenes.angles_of(shot, movie=True))
        _pictures, tagged, places, marked = movie_layout(figures, cast, cw, ch_, angles)
        return h3prompt.build(tagged, places, parts.get("detailed_description") or action, seconds=seconds,
                              soundscape=parts.get("overall_soundscape", ""), music=parts.get("non_diegetic_music", ""),
                              triggers=triggers, marked=marked,
                              angles=[f"<Picture {i}>" for i in range(2, 2 + angles)])
    if backend == KLEIN3:
        return flux3_prompt(action, shot, cast, figures, (cw, ch_), room_details, text_encoder, triggers, upsampler)
    key = {KLEIN: "klein", QWEN: "qwen", KREA2: "krea2"}[backend]
    body = expand(action, shot, cast, room_details, max_words=200 if key != "krea2" else 180,
                  text_encoder=text_encoder, guide=GUIDES[key]).strip() or action.strip()
    if backend == KREA2:
        return join_passes(krea2_prompts(figures, cast, body, triggers, photo))
    # Qwen with the ControlNet on gets the empty room as image 1 (see SceneRefStudio.run).
    marked = not (backend == QWEN and control_strength > 0 and figures)
    return build_prompt(figures, cast, body, backend, triggers, marked=marked, size=(cw, ch_), photo=photo,
                        angles=len(scenes.angles_of(shot)))


# FLUX.3 Klein's request to BFL's upsampler: what only SceneRef knows, in plain words.
FLUX3_ROOM_ON = ("Add a few small things that fit the room (plants, books, a lamp, cushions, cups) on its existing "
                 "surfaces, and describe the light without new windows.")
FLUX3_ROOM_OFF = "Add no objects."
FLUX3_KEEP = ("Turn the render into a real photograph unless another style is asked for. Keep its windows, doors, "
              "walls, wall pictures and furniture, and add no people. The figures in the first image give only each "
              "person's place, size and pose: their bare grey or plain surfaces are not skin, clothing or hair, and "
              "none of them remains. The people take their faces, hair, age and build from their own pictures, and "
              "wear what those pictures show unless the request names other clothes; describe a complete outfit, "
              "with ordinary everyday clothes for what the pictures do not show.")
FLUX3_KEEP_PHOTO = ("Keep the first image's camera angle, framing, windows, doors, walls, wall pictures and furniture, "
                    "and its look unless another style is asked for; add no other people. The people take their "
                    "faces, hair, age and build from their own pictures, and wear what those pictures show unless "
                    "the request names other clothes; describe a complete outfit, with ordinary everyday clothes for "
                    "what the pictures do not show.")


def flux3_prompt(action, shot, cast, figures, size, room_details=True, text_encoder=AUTO, triggers="",
                 upsampler=AUTO) -> str:
    """FLUX.3 Klein's prompt: the scene (image 1) and the cast pictures go to BFL's prompt
    upsampler with SceneRef's request (which image shows whom, which mannequin becomes whom,
    the action); its caption, then the trigger words. Same inputs = the same caption, without
    loading the model again."""
    import hashlib

    import torch

    from . import flux3
    from .studio import KLEIN3, _ref_tensor, build_prompt

    what = " ".join(action.split()) or "Something natural for this room."
    photo = scenes.is_pictures(shot)
    place = f"The place: {shot['place'].rstrip('. ')}." if photo and shot.get("place") else ""
    body = " ".join(x for x in [what, place, FLUX3_ROOM_ON if room_details else FLUX3_ROOM_OFF,
                                FLUX3_KEEP_PHOTO if photo else FLUX3_KEEP] if x)
    angles = scenes.angles_of(shot)
    request = build_prompt(figures, cast, body, KLEIN3, "", marked=True, size=size, photo=photo,
                           angles=len(angles)).replace("\n\n", " ")
    pictures = [_small(shot["composite"], flux3.UPSAMPLER_MEGAPIXELS)]
    pictures += [_small(t, flux3.UPSAMPLER_MEGAPIXELS) for t in angles]
    pictures += [_small(_ref_tensor(im), flux3.UPSAMPLER_MEGAPIXELS) for ch in cast for im, _role in ch["refs"]]
    key = hashlib.sha1("\0".join([request, flux3.upsampler_name(upsampler, text_encoder)]).encode("utf-8"))
    for p in pictures:
        key.update((p * 255).round().to(dtype=torch.uint8).cpu().numpy().tobytes())
    key = "flux3:" + key.hexdigest()
    if key not in _CACHE:
        if len(_CACHE) > 32:
            _CACHE.clear()
        _CACHE[key] = flux3.upsample(request, pictures, text_encoder, choice=upsampler) or request
    return "\n\n".join(x for x in (_CACHE[key], triggers.strip()) if x)


_CACHE: dict = {}


def expand(action, shot, cast, add_room_details=True, max_words=110, seed=0, text_encoder=AUTO,
           unload_after=True, mode="image", guide=None, seconds=0.0) -> str:
    """The expanded action. Same inputs (text, people, picture, settings) = same text, without
    loading the model again, so re-rolling only the image seed in the Studio stays fast.
    guide = a backend's prompt guide (the Studio's 'Write the prompt'); none = the plain
    scene directions of the Prompt Expander node."""
    import hashlib

    import torch

    figures = (shot or {}).get("figures") or [[]]
    size = (int(shot["width"]), int(shot["height"])) if shot is not None else (0, 0)
    people = people_lines(figures[0] if figures else [], cast, size)
    user = []
    if scenes.is_pictures(shot) and shot.get("place"):
        user.append(f"The place: {shot['place']}")
    if people:
        user.append("People:\n" + "\n".join(people))
    user.append("What happens: " + (action.strip() or "(nothing specified: something natural for this room)"))
    user = "\n\n".join(user)
    if guide and mode == "video":
        system = guide.format(room=ROOM_VIDEO_ON if add_room_details else ROOM_OFF, words=max_words,
                              seconds=f"{seconds:.1f}")
    elif guide:
        system = guide + STUDIO_RULES.format(room=ROOM_ON if add_room_details else ROOM_OFF, words=max_words)
    elif mode == "video":
        system = SYSTEM_VIDEO.format(room=ROOM_VIDEO_ON if add_room_details else ROOM_OFF, words=max_words)
    else:
        system = SYSTEM.format(room=ROOM_ON if add_room_details else ROOM_OFF, words=max_words)
    if not (figures and figures[0]):
        # No mannequins in this shot: the people come from the cast and the text alone.
        system = system.replace(MARKED, UNMARKED)
    if scenes.is_pictures(shot):
        for old, new in PHOTO_WORDS.items():
            system = system.replace(old, new)
    image = _small(shot["composite"]) if shot is not None else None
    name = pick_encoder(text_encoder)

    key = hashlib.sha1("\0".join([user, system, name, str(max_words), str(seed)]).encode("utf-8"))
    if image is not None:
        key.update((image * 255).round().to(dtype=torch.uint8).cpu().numpy().tobytes())
    key = key.hexdigest()
    if key in _CACHE:
        return _CACHE[key]

    # A movie prompt also has its soundscape and music sections after the description.
    tokens = int(max_words * (3 if guide and mode == "video" else 2) + 64)
    text = vlm_text([(user, image)], system, name, tokens, seed, unload_after)[0]
    text = text or action.strip()
    if len(_CACHE) > 32:
        _CACHE.clear()
    _CACHE[key] = text
    return text
