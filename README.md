# candle-lora-macro
[![MIT License](https://img.shields.io/badge/License-MIT-informational)](LICENSE)
[![Continuous integration](https://github.com/EricLBuehler/candle-lora-macro/actions/workflows/ci.yml/badge.svg)](https://github.com/EricLBuehler/candle-lora-macro/actions/workflows/ci.yml)

Macros for candle-lora. These are designed to allow ergonomics similar to what the `peft` library allows.

These macros will provide utilities to:
1) Switch the concrete types for dynamic-dispatch types.
2) Swap the layers.

Currently, candle-lora-macro is able to:
1) Select layers for swapping.