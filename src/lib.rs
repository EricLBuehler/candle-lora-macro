use proc_macro::TokenStream as TokenStream1;
use proc_macro2::TokenStream;
use syn::{
    parse::Parser, parse_macro_input, Data, DeriveInput, Fields, GenericArgument, Ident, Type,
    TypeParamBound, Visibility,
};

#[proc_macro_attribute]
pub fn replace_layer_fields(_args: TokenStream1, input: TokenStream1) -> TokenStream1 {
    let mut ast = parse_macro_input!(input as DeriveInput);
    match &mut ast.data {
        Data::Struct(ref mut struct_data) => match &mut struct_data.fields {
            Fields::Named(fields) => {
                for field in fields.named.iter_mut() {
                    let mut f = None;
                    let ident = field.ident.clone().unwrap();
                    let ty = field.ty.clone();
                    match ty {
                        Type::Path(path) => {
                            if path.path.segments.len() == 1 {
                                match path
                                    .path
                                    .segments
                                    .first()
                                    .unwrap()
                                    .ident
                                    .to_string()
                                    .as_str()
                                {
                                    "Linear" => {
                                        if let Visibility::Public(_) = field.vis {
                                            f = Some(syn::Field::parse_named.parse2(quote::quote!(pub #ident: Box<dyn LinearLayerLike>)).unwrap());
                                        } else {
                                            f = Some(syn::Field::parse_named.parse2(quote::quote!(#ident: Box<dyn LinearLayerLike>)).unwrap());
                                        }
                                    }
                                    "Conv1d" => {
                                        if let Visibility::Public(_) = field.vis {
                                            f = Some(syn::Field::parse_named.parse2(quote::quote!(pub #ident: Box<dyn Conv1dLayerLike>)).unwrap());
                                        } else {
                                            f = Some(syn::Field::parse_named.parse2(quote::quote!(#ident: Box<dyn Conv1dLayerLike>)).unwrap());
                                        }
                                    }
                                    "Conv2d" => {
                                        if let Visibility::Public(_) = field.vis {
                                            f = Some(syn::Field::parse_named.parse2(quote::quote!(pub #ident: Box<dyn Conv2dLayerLike>)).unwrap());
                                        } else {
                                            f = Some(syn::Field::parse_named.parse2(quote::quote!(#ident: Box<dyn Conv2dLayerLike>)).unwrap());
                                        }
                                    }
                                    "Embedding" => {
                                        if let Visibility::Public(_) = field.vis {
                                            f = Some(syn::Field::parse_named.parse2(quote::quote!(pub #ident: Box<dyn EmbeddingLayerLike>)).unwrap());
                                        } else {
                                            f = Some(syn::Field::parse_named.parse2(quote::quote!(#ident: Box<dyn EmbeddingLayerLike>)).unwrap());
                                        }
                                    }
                                    _ => {}
                                }
                            } else {
                                panic!("Expected single type")
                            }
                        }
                        _ => {
                            panic!("Expected syn::Type::Path");
                        }
                    }
                    if let Some(f) = f {
                        *field = f;
                    }
                }
            }
            _ => {
                panic!("Named fields are required.")
            }
        },
        _ => {
            panic!("Cannot swap fields of non struct!");
        }
    }

    quote::quote!(#ast).into()
}

fn is_ident(ident: &Ident, name: &str) -> bool {
    *ident == name
}

#[proc_macro_derive(AutoLora)]
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
                                                field.ident.as_ref().unwrap().to_string(),
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

    //let st_name = Ident::new("Model", Span::mixed_site());

    let mut linear_stream = TokenStream::new();
    if !linear_fields.is_empty() {
        quote_into::quote_into!(linear_stream += [#{
            for (_,name) in linear_fields.iter() {
                quote_into::quote_into!(linear_stream += (linear.insert(#name.to_string(), &*self.a)),)
            }
        }];);
    }

    let mut conv1d_stream = TokenStream::new();
    if !conv1d_fields.is_empty() {
        quote_into::quote_into!(conv1d_stream += [#{
            for (_,name) in conv1d_fields.iter() {
                quote_into::quote_into!(conv1d_stream += (linear.insert(#name.to_string(), &*self.a)),)
            }
        }];);
    }

    let mut conv2d_stream = TokenStream::new();
    if !conv2d_fields.is_empty() {
        quote_into::quote_into!(conv2d_stream += [#{
            for (_,name) in conv2d_fields.iter() {
                quote_into::quote_into!(conv2d_stream += (linear.insert(#name.to_string(), &*self.a)),)
            }
        }];);
    }

    let mut embed_stream = TokenStream::new();
    if !embed_fields.is_empty() {
        quote_into::quote_into!(embed_stream += [#{
            for (_,name) in embed_fields.iter() {
                quote_into::quote_into!(embed_stream += (linear.insert(#name.to_string(), &*self.a)),)
            }
        }];);
    }

    let mut linear_stream_assign = TokenStream::new();
    if !linear_fields.is_empty() {
        quote_into::quote_into!(linear_stream_assign += [#{
            for (name, n) in linear_fields {
                linear_stream_assign.extend(quote::quote!((self.#name = ::std::boxed::Box::new(new_layers.linear.get(#n).unwrap().clone())),))
            }
        }];);
    }

    let mut conv1d_stream_assign = TokenStream::new();
    if !conv1d_fields.is_empty() {
        quote_into::quote_into!(conv1d_stream_assign += [#{
            for (name, n) in conv1d_fields {
                conv1d_stream_assign.extend(quote::quote!((self.#name = ::std::boxed::Box::new(new_layers.linear.get(#n).unwrap().clone())),))
            }
        }];);
    }

    let mut conv2d_stream_assign = TokenStream::new();
    if !conv2d_fields.is_empty() {
        quote_into::quote_into!(conv2d_stream_assign += [#{
            for (name, n) in conv2d_fields {
                conv2d_stream_assign.extend(quote::quote!((self.#name = ::std::boxed::Box::new(new_layers.linear.get(#n).unwrap().clone())),))
            }
        }];);
    }

    let mut embed_stream_assign = TokenStream::new();
    if !embed_fields.is_empty() {
        quote_into::quote_into!(embed_stream_assign += [#{
            for (name, n) in embed_fields {
                embed_stream_assign.extend(quote::quote!((self.#name = ::std::boxed::Box::new(new_layers.linear.get(#n).unwrap().clone())),))
            }
        }];);
    }

    let mut stream = TokenStream::new();
    quote_into::quote_into! { stream +=
        impl #st_name {
            fn get_lora_model<'a>(&'a mut self, lora_config: candle_lora::LoraConfig, linear_config: Option<candle_lora::LoraLinearConfig>, conv1d_config: Option<candle_lora::LoraConv1dConfig>, conv2d_config: Option<candle_lora::LoraConv2dConfig>, embed_config: Option<candle_lora::LoraEmbeddingConfig>) {
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
                let selection = builder.build();

                let new_layers = candle_lora::Lora::convert_model(selection, lora_config);

                #linear_stream_assign
                #conv1d_stream_assign
                #conv2d_stream_assign
                #embed_stream_assign
            }
        }
    }

    stream.into()
}
