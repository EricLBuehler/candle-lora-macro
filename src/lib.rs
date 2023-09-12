use proc_macro::{TokenStream, TokenTree::{Group, Ident, Punct, self}, Group as GroupStruct, Punct as PunctStruct};

#[derive(Default)]
struct Member {
    name: Option<String>,
    tp: Option<String>,
}

fn parse_member(group: &GroupStruct, toks: &mut Vec<TokenTree>) -> Vec<TokenTree> {
    let mut stream = group.stream().into_iter();
    let mut next = stream.next();
    while let Some(ref tok) = next {
        let mut member = Member::default();
        if let Ident(ident) = tok {
            member.name = Some(ident.to_string());
            println!("{:?}", member.name);
            
            while let Some(ref tok) = next {
                if let Punct(punct) = tok {
                    if punct.as_char() != ':' {
                        return toks.to_vec();
                    }
                }
            }
        }
        else {
            return toks.to_vec();
        }
    }
    toks.to_vec()
}

#[proc_macro_attribute]
pub fn lorafy(attr: TokenStream, item: TokenStream) -> TokenStream {
    assert!(attr.is_empty());
    let mut toks = item.into_iter().collect::<Vec<_>>();

    match &toks.clone()[..] {
        &[Ident(_), Ident (_), Group(ref group)] => {
            parse_member(group, &mut toks);
        }
        _ => {}
    }
    
    TokenStream::from_iter(toks)
}