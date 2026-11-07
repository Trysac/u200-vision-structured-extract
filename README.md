# vision-structured-extract

[![PyPI version](https://img.shields.io/pypi/v/trysac-vision-structured-extract)](https://pypi.org/project/trysac-vision-structured-extract/)
[![License: MIT](https://img.shields.io/pypi/l/trysac-vision-structured-extract)](LICENSE)
[![CI](https://github.com/Trysac/u200-vision-structured-extract/actions/workflows/ci.yml/badge.svg)](https://github.com/Trysac/u200-vision-structured-extract/actions/workflows/ci.yml)

Extract schema-validated JSON from images through an injected vision model port: header-only image checks, a JSON Schema subset and a bounded re-ask loop

## Problem

Describe in two or three sentences the concrete problem this utility solves, who hits it, and what
they do today without it. Replace this paragraph when the first real feature lands.

## Install

```sh
pip install trysac-vision-structured-extract
# or
uv add trysac-vision-structured-extract
```

Requires Python 3.10 or newer. No runtime dependencies.

## Usage

```python
import vision_structured_extract

# normalize() trims surrounding whitespace and lower-cases the text.
print(vision_structured_extract.normalize("  Hello World "))  # -> "hello world"

# Blank input is rejected explicitly instead of returning an empty string.
try:
    vision_structured_extract.normalize("   ")
except ValueError as exc:
    print(f"rejected: {exc}")
```

## Documentation

- [Architecture](docs/ARCHITECTURE.md) and [decisions](docs/decisions/)
- [Testing](docs/TESTING.md) and [security](docs/SECURITY.md)
- [Project record](docs/PROJECT.md) and [stack](docs/STACK.md)
- [Changelog](CHANGELOG.md) and [contributing guide](CONTRIBUTING.md)

## License

[MIT](LICENSE) (c) 2026 Miguel Gomez
