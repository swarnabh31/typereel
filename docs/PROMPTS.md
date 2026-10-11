# Prompt architecture

Goal: professional, on-brand, *grounded* copy from small local models, using as little context as possible.

## Principles

1. **Stage the work.** Small focused calls beat one giant prompt; each can be cached or retried alone.
2. **Inject the brand voice everywhere** (`brand.voice_block`) — tone, narration style, banned terms, style rules, reference lines, limits.
3. **Ground every claim.** The director may only use the *Fact Sheet* and the *Screen Cards*. The lint verifies numbers.
4. **Constrain the output.** Ollama structured outputs (`format` = JSON schema) with automatic fallback to plain JSON mode for older servers.
5. **Measure, then repair.** Deterministic checks first; one targeted revision call that quotes the exact problems.
6. **Low temperature for planning** (0.15–0.4), moderate for writing (0.6).

## Stages (appdemo / app screenshots)

| # | Stage | Model | Input | Output | Cached |
|---|---|---|---|---|---|
| 1 | Fact sheet | writer | your brief (≤ 6 000 chars) | product, audience, problem, 3–7 claims, metrics (only if stated) | – |
| 2 | Screen cards | vision | one screenshot @ 1280 px | screen, purpose, stage, visible text, numbers, 1–3 zoom regions (normalised boxes) | by image hash |
| 3 | Director | writer | brand voice + facts + cards + beat count | title, beats (screen, region, kicker, caption, sub, VO, optional KPI), CTA, hashtags, art line | – |
| 4 | Lint | code | storyboard | banned terms, word limits, ungrounded numbers, repeated openings, repeats of past videos | – |
| 5 | Revise | writer | the JSON + the issue list | corrected JSON (only listed issues) | – |
| 6 | Packager | writer | storyboard + facts | LinkedIn / X / Shorts / Instagram copy + hashtags | – |
| – | Rewrite beat | writer | one beat + neighbours + your instruction | new kicker/caption/sub/VO | – |

Text modes (quotes/facts/…) use the `lines_prompt` (brand voice + optional facts) then the same lint/revise loop.
Photo stories use `story_prompt` (vision cards + cast rule: *same words for the same person*).
ClipMiner picks transcript **segment ids** (timestamps come from STT, never from the model).

## Master prompt skeleton (director)

```
ROLE        creative director for {brand}
BRAND VOICE tone · narration · NEVER-use list · style rules · reference lines · limits
FACTS       the only claims you may make
ASSETS      numbered screens: stage · purpose · visible text · numbers · zoom regions
BRIEF       the user's own words
TASK        beats, title hook, caption/sub/VO limits, region rules, KPI rule, CTA, hashtags, art line
RULES       only FACTS/ASSETS · no invented numbers · no exclamation marks · output JSON only
```

## Tuning

- Voice drifting? Add **reference lines** to the brand kit — they are the strongest steering signal for small models.
- Over-long lines? Lower *Max caption words*; the lint will force a rewrite.
- A banned word keeps returning? Add it to **Never use** — it is enforced in prompt *and* lint.
- Change prompts in `trcore/prompts.py`; schemas live in the same file.
