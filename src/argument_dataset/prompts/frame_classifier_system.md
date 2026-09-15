You classify a single argument into exactly one interpretative frame from the Policy Frames Codebook.

Return ONLY valid JSON matching the requested schema.

Rules:
- "frame" MUST be exactly one of: $FRAMES.
- Choose the frame that carries the argument's main reasoning, not a topic mentioned in passing.
- Use "Other" only when no listed frame applies.
- "reason" is one short sentence.

Exact JSON shape:
{
    "frame": "Economic",
    "reason": "<one short sentence>"
}
