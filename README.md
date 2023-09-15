# candle-lora-macro
[![MIT License](https://img.shields.io/badge/License-MIT-informational)](LICENSE)
[![Continuous integration](https://github.com/EricLBuehler/candle-lora-macro/actions/workflows/ci.yml/badge.svg)](https://github.com/EricLBuehler/candle-lora-macro/actions/workflows/ci.yml)

This library makes using [`candle-lora`](https://github.com/EricLBuehler/candle-lora) as simple as adding 2 macros to your model structs and calling a method! It is inspired by the simplicity of the `peft` library's `get_peft_model` method. Like `candle-lora`, the supported concrete layer types are `Linear`, `Conv1d`, `Conv2d`, and `Embedding`.

`candle-lora-macro` exports 2 macros: `AutoLoraConvert` and `replace_layer_fields`.

The `AutoLoraConvert` derive macro automatically creates a method `get_lora_model`, when called which selects and swaps all supported layers for their LoRA counterparts. This method is the equivalent of `peft`'s `get_peft_model` method, and modifies the model in place. It expects all
layers of the supported types to be a `dyn` type, that is `Box<dyn ...LayerLike>`.

To further automate the process of using `candle-lora`, `candle-lora-macro` also provides an attribute macro called `replace_layer_fields`.
`replace_layer_fields` swaps out the concrete types for `dyn` types. If this macro is not added to the model structs, be sure to change the member types to `Box<dyn ...LayerLike>`.