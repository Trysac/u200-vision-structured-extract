# Security

## Threat Model

### Defended

- **Image Bomb / Decompression Attacks**: Images are never decoded. Headers are validated via direct byte-slicing and struct-unpacking. JPEG markers are walked with a hard limit of 1,000 segments.
- **Prompt Injection & Data Tampering**: The system prompt (`SYSTEM_PROMPT`) explicitly instructs the model that image content is data, not instructions. Schemas are strictly delimited in `<schema>` tags.
- **Information Disclosure**: Logs never record raw image bytes or vision model replies. A `NullHandler` is used for the logger.
- **Denial of Service (Memory/Resource Exhaustion)**:
  - File size capped at 20MB.
  - Image dimensions capped at 8,000px.
  - Model replies exceeding 200,000 characters are rejected (`reply_too_large`).
  - JSON parser considers at most 16 balanced candidates.
  - Schema recursion depth is limited to 16.

### Not Defended

- Arbitrary malicious payloads in the *model's* output (though these are mitigated by schema validation).
- Compromise of the third-party vision provider API.
