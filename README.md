# trysac-vision-structured-extract

Extract schema-validated JSON from images through an injected vision model port: header-only image checks, a JSON Schema subset and a bounded re-ask loop.

## Problem

Teams send photos of forms, labels and screenshots to vision models and get back prose, half-valid JSON or invented fields, then write unbounded retry loops and leak image data into logs. Each project re-implements image checks, prompt layout, JSON extraction and validation. This library does that once: it checks images from their headers, sends a fixed prompt through a provider-neutral port and returns a value validated against a declared schema.

## Installation

```bash
pip install trysac-vision-structured-extract
```

## Usage

```python
from decimal import Decimal
from vision_structured_extract import RecordedVision, extract

schema = {
    "type": "object",
    "properties": {"total": {"type": "number"}},
    "required": ["total"],
    "additionalProperties": False,
}
vision = RecordedVision(['{"total": 10.5}'])
extraction = extract(vision, [], schema)
assert extraction.value == {"total": Decimal("10.5")}
```

## Public API Summary

- `prepare_image(data: bytes, *, max_bytes: int = 20_000_000, max_side: int = 8_000) -> ImageInput`: Inspects PNG/JPEG/WebP/GIF headers without decoding.
- `validate(value: object, schema: Mapping) -> list[Issue]`: Validates against a supported subset (type, enum, bounds, required, additionalProperties).
- `extract(...) -> Extraction`: Coordinates extraction via `VisionPort`, with a bounded retry loop (1-10 attempts).
- `VisionExtractError`: Base class for errors (`unsupported_image`, `image_too_large`, `too_many_images`, `schema_unsupported`, `no_json`, `invalid_json`, `schema_validation`, `reply_too_large`, `attempts_exhausted`).
- `AttemptsExhaustedError`: Raised when retry loops expire, containing the last observed `Issue`s.
- `RecordedVision` / `FakeVision`: Test doubles for recording and simulating model outputs.

## Why not <best-known alternative>?

Unlike provider-specific structured-output modes, this library is provider-neutral, allows for deterministic offline testing via dependency injection, and performs strict, local validation before accepting any model output.

[Security Details](docs/SECURITY.md)
