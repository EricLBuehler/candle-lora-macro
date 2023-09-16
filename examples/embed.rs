use candle_core::{DType, Device, Module, Result, Tensor};
use candle_lora::{EmbeddingLayerLike, LoraConfig, LoraEmbeddingConfig};
use candle_lora_macro::{replace_layer_fields, AutoLoraConvert};
use candle_nn::{init, VarMap, Embedding};

#[replace_layer_fields]
#[derive(AutoLoraConvert, Debug)]
struct Model {
    a: Embedding,
    b: i32,
}

impl Module for Model {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.a.forward(input)
    }
}

fn main() {
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

    let mut model = Model {
        a: Box::new(Embedding::new(layer_weight.clone(), 10)),
        b: 1,
    };

    let loraconfig = LoraConfig::new(1, 1., None, &device, dtype);
    model.get_lora_model(
        loraconfig,
        None,
        None,
        None,
        Some(LoraEmbeddingConfig::new(10, 10)),
    );

    let dummy_image = Tensor::zeros((10, 10), DType::U32, &device).unwrap();

    //Test the model
    let digit = model.forward(&dummy_image).unwrap();
    println!("Output: {digit:?}");

    println!("{:?}", model.a);
    println!("{:?}", model.b);
}
