use proc_macro::TokenStream as TokenStream1;
use proc_macro2::TokenStream;
use syn::{parse_macro_input, Data, DeriveInput, GenericArgument, Ident, Type, TypeParamBound};

fn is_ident(ident: &Ident, name: &str) -> bool {
    *ident == name
}

#[proc_macro_derive(SelectLoraLayers)]
pub fn select(tokens: TokenStream1) -> TokenStream1 {
    let ast = parse_macro_input!(tokens as DeriveInput);
    let mut linear_fields = Vec::new();
    let mut conv1d_fields = Vec::new();
    let mut conv2d_fields = Vec::new();
    let mut embed_fields = Vec::new();
    let st_name = &ast.ident;

    match ast.data {
        Data::Struct(st) => {
            for field in st.fields {
                match field.ty {
                    Type::Path(path) => {
                        let segments = path.path.segments.into_iter().collect::<Vec<_>>();
                        if segments.len() != 1 {
                            continue;
                        }
                        if !is_ident(&segments[0].ident, "Box") {
                            continue;
                        }
                        if let syn::PathArguments::AngleBracketed(bracketed) =
                            &segments.get(0).as_ref().unwrap().arguments
                        {
                            if bracketed.args.len() != 1 {
                                continue;
                            }
                            match &bracketed.args[0] {
                                GenericArgument::Type(Type::TraitObject(trobj)) => {
                                    let bounds = &trobj.bounds;
                                    if bounds.len() != 1 {
                                        continue;
                                    }
                                    match bounds.first().unwrap() {
                                        TypeParamBound::Trait(bound) => {
                                            if bound.path.segments.len() != 1 {
                                                continue;
                                            }
                                            let trt = &bound.path.segments.first().unwrap().ident;
                                            let value = (
                                                field.ident.clone().unwrap(),
                                                trt.clone(),
                                                st_name.to_string()
                                                    + &field.ident.as_ref().unwrap().to_string(),
                                            );
                                            if is_ident(trt, "LinearLayerLike") {
                                                linear_fields.push(value);
                                            } else if is_ident(trt, "Conv1dLayerLike") {
                                                conv1d_fields.push(value);
                                            } else if is_ident(trt, "Conv2dLayerLike") {
                                                conv2d_fields.push(value);
                                            } else if is_ident(trt, "EmbeddingLayerLike") {
                                                embed_fields.push(value);
                                            }
                                        }
                                        _ => continue,
                                    }
                                }
                                _ => continue,
                            }
                        } else {
                            continue;
                        }
                    }
                    _ => continue,
                }
            }
        }
        _ => {
            todo!()
        }
    }

    let mut linear_stream = TokenStream::new();
    if !linear_fields.is_empty() {
        quote_into::quote_into!(linear_stream += [#{
            for (_,_,name) in linear_fields {
                quote_into::quote_into!(linear_stream += (linear.insert(#name.to_string(), &*self.a)),)
            }
        }];);
    }

    let mut conv1d_stream = TokenStream::new();
    if !conv1d_fields.is_empty() {
        quote_into::quote_into!(conv1d_stream += [#{
            for (_,_,name) in conv1d_fields {
                quote_into::quote_into!(conv1d_stream += (linear.insert(#name.to_string(), &*self.a)),)
            }
        }];);
    }

    let mut conv2d_stream = TokenStream::new();
    if !conv2d_fields.is_empty() {
        quote_into::quote_into!(conv2d_stream += [#{
            for (_,_,name) in conv2d_fields {
                quote_into::quote_into!(conv2d_stream += (linear.insert(#name.to_string(), &*self.a)),)
            }
        }];);
    }

    let mut embed_stream = TokenStream::new();
    if !embed_fields.is_empty() {
        quote_into::quote_into!(embed_stream += [#{
            for (_,_,name) in embed_fields {
                quote_into::quote_into!(embed_stream += (linear.insert(#name.to_string(), &*self.a)),)
            }
        }];);
    }

    let result = quote::quote! {
        impl #st_name {
            fn get_lora_model<'a>(&'a mut self, lora_config: candle_lora::LoraConfig, linear_config: Option<candle_lora::LoraLinearConfig>, conv1d_config: Option<candle_lora::LoraConv1dConfig>, conv2d_config: Option<candle_lora::LoraConv2dConfig>, embed_config: Option<candle_lora::LoraEmbeddingConfig>) -> candle_lora::SelectedLayers<'a, String> {
                let mut linear: ::std::collections::HashMap<String, &dyn candle_lora::LinearLayerLike> = ::std::collections::HashMap::new();
                let mut conv1d: ::std::collections::HashMap<String, &dyn candle_lora::Conv1dLayerLike> = ::std::collections::HashMap::new();
                let mut conv2d: ::std::collections::HashMap<String, &dyn candle_lora::Conv2dLayerLike> = ::std::collections::HashMap::new();
                let mut embed: ::std::collections::HashMap<String, &dyn candle_lora::EmbeddingLayerLike> = ::std::collections::HashMap::new();
                #linear_stream
                #conv1d_stream
                #conv2d_stream
                #embed_stream

                let mut builder = candle_lora::SelectedLayersBuilder::new();
                if linear_config.is_some() {
                    builder = builder.add_linear_layers(linear, linear_config.unwrap());
                }
                if conv1d_config.is_some() {
                    builder = builder.add_conv1d_layers(conv1d, conv1d_config.unwrap());
                }
                if conv2d_config.is_some() {
                    builder = builder.add_conv2d_layers(conv2d, conv2d_config.unwrap());
                }
                if embed_config.is_some() {
                    builder = builder.add_embed_layers(embed, embed_config.unwrap());
                }
                builder.build()
            }
        }
    };

    result.into()
}
