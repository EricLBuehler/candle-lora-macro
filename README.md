# candle-lora-macro
[![MIT License](https://img.shields.io/badge/License-MIT-informational)](LICENSE)
[![Continuous integration](https://github.com/EricLBuehler/candle-lora-macro/actions/workflows/ci.yml/badge.svg)](https://github.com/EricLBuehler/candle-lora-macro/actions/workflows/ci.yml)

This library are designed to allow ergonomics similar to what the `peft` library allows. It does this by providing a single derive macro, called `AutoLora`. This macro defines a method
`get_lora_model` which selects and swaps all `Box<dyn ...LayerLike>` layers. Currently, it does not update struct types.