# Header-only image validation and decimal-based JSON extraction

Status: accepted (generated, pending human review)

## Context

Extracting structured data from multimodal models requires robust input checking to prevent decompression bomb attacks and format confusion, as well as deterministic schema validation without introducing third-party dependencies.

## Decision

Inspect binary image headers directly for PNG, JPEG, WebP, and GIF without raster decompression. Use json.loads with parse_float=Decimal to prevent precision loss. Implement a provider-neutral VisionPort protocol and bounded re-ask retry loop.

## Consequences

Zero external dependencies; fast and safe execution; strict protection against malformed images and model hallucinations with exact structured feedback.
