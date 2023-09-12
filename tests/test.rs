use candle_lora_macro::lorafy;

#[test]
fn basic_test() {
    trait T {}

    #[lorafy]
    struct _S {
        x: Box<dyn T>
    }
}