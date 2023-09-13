use std::fmt::Debug;

use candle_core::{Device, DType};
use candle_lora::{LinearLayerLike, LoraLinearConfig};
use candle_lora_macro::SelectLoraLayers;
use candle_nn::{VarMap, init, Linear};

trait MyTrait: Debug { }

impl MyTrait for i32 { }

#[allow(dead_code)]
#[derive(SelectLoraLayers)]
struct Model {
    a: Box<dyn LinearLayerLike>,
    b: i32,
}

fn main() {
    let device = Device::Cpu;
    let dtype = DType::F32;

    let map = VarMap::new();
    let layer_weight = map.get(
        (10, 10),
        "layer.weight",
        init::DEFAULT_KAIMING_NORMAL,
        dtype,
        &device,
    ).unwrap();

    let mut m = Model {
        a: Box::new(Linear::new(layer_weight.clone(), None)),
        b: 1,
    };

    m.select_layers(Some(LoraLinearConfig::new(10, 10)), None, None, None);

    println!("{:?}", m.a);
}