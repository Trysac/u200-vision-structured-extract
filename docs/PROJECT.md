# Project Record

## Objective

Extract schema-validated JSON from images through an injected vision model port, providing header-only image checks, a JSON Schema subset and a bounded re-ask loop.

## Scope

- Header-only image validation for PNG, JPEG, WebP (VP8, VP8L, VP8X), and GIF.
- Strictly typed `VisionPort` protocol for multimodal LLM completions.
- Schema validation enforcing strict type support and bounded recursion depth (16).
- Extraction via balanced bracket scanner and deterministic feedback retry loop.

## Out-of-Scope

- Provider-specific adapters (SDKs).
- Image manipulation (resizing, cropping).
- Object detection (bounding boxes).
- Streaming responses.

## Factory run u200 (measured from API usage)

| Phase | Model | Batch | Input | Output | Cache write | Cache read | Cost (USD) |
|---|---|---|---|---|---|---|---|
| generation | gemini-3.8-flash | yes | 4830 | 14656 | 0 | 0 | 0.0293 |
| fix-1 | gemini-3.8-flash | no | 68020 | 2078 | 0 | 507603 | 0.0969 |
| fix-2 | gemini-3.8-flash | no | 16772 | 155 | 0 | 12517 | 0.0141 |
| audit | gemini-3.1-flash-lite | no | 51251 | 210 | 0 | 50779 | 0.0144 |
| docs | gemini-3.1-flash-lite | no | 149129 | 1310 | 0 | 0 | 0.0392 |

Total: $0.1939. Prices: Anthropic list prices.
