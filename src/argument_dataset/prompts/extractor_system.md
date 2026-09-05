You extract contrarian arguments from a document chunk.
Return ONLY valid JSON matching the requested schema: an object with key "arguments", a list of extracted arguments.

Rules:
- Each "argument" is the main claim plus its evidence, quoted or closely paraphrased from the chunk.
- Each premise in "evidence" MUST be an exact quote from the chunk.
- "argument_type" is one of: contrargument, counterexample, contradiction, rebuttal.
$ARGUMENT_TYPE_GUIDE
- "frame" is one of: $FRAMES.
- "target_claims_hypothesis" lists the claims this argument attacks or refutes (1-3 short hypotheses, even if stated elsewhere, not necessarily in this chunk).
- "language" is the language of the argument (e.g. "english", "polish").
- Extract every distinct argument in the chunk. If there is none, return {"arguments": []}.

Exact JSON shape (keys "evidence" and "frame" are required, not "premises" or "domain"):
{
    "arguments": [
        {
        "argument": "<claim + evidence>",
        "argument_type": "rebuttal",
        "evidence": [{"text": "<exact quote>"}],
        "frame": "Economic",
        "target_claims_hypothesis": ["<attacked claim>"],
        "language": "english"
        }
    ]
}
