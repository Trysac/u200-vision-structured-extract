# Architecture

## Overview

`trysac-vision-structured-extract` provides a deterministic pipeline for extracting structured JSON data from images via multimodal LLMs.

```
[Raw Images] ──> prepare_image() ──> [ImageInput (header verified, base64)]
                                              │
                                              ▼
[JSON Schema] ──> validate() ───────> extract() <──> [VisionPort Protocol]
                                              │
                                              ▼
                                   [Validated Extraction]
```

## Core Components

1. **Image Inspector (`prepare_image`)**:
   Inspects image headers directly (PNG `IHDR`, JPEG `SOF0..SOF15` markers, WebP `VP8`, `VP8L`, `VP8X`, GIF `GIF87a`/`GIF89a`). Verifies payload size and dimensions without raster decompression.

2. **Schema Engine (`validate`)**:
   Validates parsed JSON against a strict schema subset with max depth 16. Rejects unsupported keywords to prevent unvalidated model output. Uses `Decimal` for all fractional numbers to prevent precision loss.

3. **Extraction & Re-ask Loop (`extract`)**:
   Interacts with an injected `VisionPort`. When a model reply fails JSON parsing or schema validation, it formats up to 20 issues into a clean re-ask prompt and repeats up to `max_attempts`.

4. **Test Doubles (`RecordedVision`, `FakeVision`)**:
   Enables fully deterministic, offline testing of multimodal extraction loops without requiring live provider API keys.
