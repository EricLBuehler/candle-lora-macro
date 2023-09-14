# candle-lora-macro
[![MIT License](https://img.shields.io/badge/License-MIT-informational)](LICENSE)
[![Continuous integration](https://github.com/EricLBuehler/candle-lora-macro/actions/workflows/ci.yml/badge.svg)](https://github.com/EricLBuehler/candle-lora-macro/actions/workflows/ci.yml)

This library is designed to allow ergonomics similar to what the `peft` library's `get_peft_model` method allows. It does this by providing a single derive macro, called `AutoLoraConvert`. This macro defines a method
`get_lora_model` which selects and swaps all `Box<dyn ...LayerLike>` layers. 

In addition, `candle-lora-macro` also provides an attribute macro called `replace_layer_fields`. This replaces `Linear`, `Conv1d`, `Conv2d`, and `Embedding` concrete types with their `candle-lora` `Box<dyn ...LayerLike>` counterpart. This means that `candle-lora` can be added to any `candle` model with minimal code changes.