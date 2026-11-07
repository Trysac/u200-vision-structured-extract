# Testing Strategy

- **Zero Network/LLM dependency**: Tests exclusively use `RecordedVision` or `FakeVision` doubles.
- **Binary Fixtures**: Image headers (PNG, JPEG, WebP, GIF) are generated in-code using `struct` and byte literals, ensuring no external test files are required.
- **Coverage**:
  - `prepare_image`: Edge cases for unsupported formats (e.g., BMP), oversized payloads, invalid dimensions.
  - `validate`: Strict enforcement of schema subset, including recursion limits.
  - `extract`: Retry logic, error reporting (including truncation of >20 issues), and fallback between fenced blocks and balanced object detection.
