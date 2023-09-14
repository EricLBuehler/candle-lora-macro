use candle_core::{DType, Device};
use candle_lora::{LoraConfig, LoraLinearConfig, LinearLayerLike};
use candle_lora_macro::{AutoLora, replace_layer_fields};
use candle_nn::{init, Linear, VarMap};

#[allow(dead_code)]
#[replace_layer_fields]
#[derive(AutoLora)]
struct Model {
    a: Linear,
    b: i32,
}

#[test]
fn test() {
    let device = Device::Cpu;
    let dtype = DType::F32;

    let map = VarMap::new();
    let layer_weight = map
        .get(
            (10, 10),
            "layer.weight",
            init::DEFAULT_KAIMING_NORMAL,
            dtype,
            &device,
        )
        .unwrap();

    let mut m = Model {
        a: Box::new(Linear::new(layer_weight.clone(), None)),
        b: 1,
    };

    let loraconfig = LoraConfig::new(1, 1., None, &device, dtype);
    m.get_lora_model(
        loraconfig,
        Some(LoraLinearConfig::new(10, 10)),
        None,
        None,
        None,
    );

    println!("{:?}", m.a);
}
