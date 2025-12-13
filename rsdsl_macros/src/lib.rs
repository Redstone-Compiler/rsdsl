use proc_macro::TokenStream;
use proc_macro2::{Delimiter, Group, Span, TokenStream as TokenStream2, TokenTree};
use quote::quote;
use std::collections::HashMap;
use syn::{parse::Parse, parse::ParseStream, Result as SynResult};

#[proc_macro]
pub fn rsdsl(input: TokenStream) -> TokenStream {
    let top = syn::parse_macro_input!(input as Top);
    match build(top.tokens) {
        Ok(ts) => ts.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

struct Top {
    tokens: TokenStream2,
}
impl Parse for Top {
    fn parse(input: ParseStream) -> SynResult<Self> {
        Ok(Self {
            tokens: input.parse()?,
        })
    }
}

#[derive(Debug)]
struct DeclEnv {
    vars: HashMap<String, usize>,
    fns: HashMap<String, usize>,
}

impl DeclEnv {
    fn new() -> Self {
        Self {
            vars: HashMap::new(),
            fns: HashMap::new(),
        }
    }
}

#[derive(Debug)]
struct ModelAst {
    name: String,
    body: TokenStream2,
}

// -----------------------------
// Build
// -----------------------------

fn build(tokens: TokenStream2) -> SynResult<TokenStream2> {
    let model = parse_model(tokens)?;
    let (decls, env, rules, objective) = parse_model_body(model.body)?;

    let mname = model.name;

    let decl_exprs: Vec<TokenStream2> = decls.into_iter().map(decl_to_expr).collect();
    let rule_exprs: Vec<TokenStream2> = rules.into_iter().map(rule_to_expr).collect();
    let obj_expr = match objective {
        Some(o) => objective_to_expr(o),
        None => quote! { None },
    };

    let _ = env; // used during parsing/validation

    Ok(quote! {
        rsdsl::ModelSpec {
            name: #mname.to_string(),
            decls: vec![ #(#decl_exprs),* ],
            rules: vec![ #(#rule_exprs),* ],
            objective: #obj_expr,
        }
    })
}

// -----------------------------
// Serializable token conversion (Tok)
// -----------------------------

fn tokdelim_to_expr(d: TokDelimAst) -> TokenStream2 {
    match d {
        TokDelimAst::Paren => quote! { rsdsl::TokDelim::Paren },
        TokDelimAst::Bracket => quote! { rsdsl::TokDelim::Bracket },
        TokDelimAst::Brace => quote! { rsdsl::TokDelim::Brace },
        TokDelimAst::None => quote! { rsdsl::TokDelim::None },
    }
}

#[derive(Debug, Clone, Copy)]
enum TokDelimAst {
    Paren,
    Bracket,
    Brace,
    None,
}

fn tt_to_tok(tt: &TokenTree) -> TokAst {
    match tt {
        TokenTree::Ident(id) => TokAst::Ident(id.to_string()),
        TokenTree::Literal(l) => TokAst::Lit(l.to_string()),
        TokenTree::Punct(p) => TokAst::Punct(p.as_char()),
        TokenTree::Group(g) => {
            let delim = match g.delimiter() {
                Delimiter::Parenthesis => TokDelimAst::Paren,
                Delimiter::Bracket => TokDelimAst::Bracket,
                Delimiter::Brace => TokDelimAst::Brace,
                Delimiter::None => TokDelimAst::None,
            };
            let inner = g.stream().into_iter().map(|t| tt_to_tok(&t)).collect();
            TokAst::Group { delim, inner }
        }
    }
}

#[derive(Debug, Clone)]
enum TokAst {
    Ident(String),
    Lit(String),
    Punct(char),
    Group {
        delim: TokDelimAst,
        inner: Vec<TokAst>,
    },
}

fn tok_to_expr(t: TokAst) -> TokenStream2 {
    match t {
        TokAst::Ident(s) => quote! { rsdsl::Tok::Ident(#s.to_string()) },
        TokAst::Lit(s) => quote! { rsdsl::Tok::Lit(#s.to_string()) },
        TokAst::Punct(c) => quote! { rsdsl::Tok::Punct(#c) },
        TokAst::Group { delim, inner } => {
            let d = tokdelim_to_expr(delim);
            let inner_exprs: Vec<TokenStream2> = inner.into_iter().map(tok_to_expr).collect();
            quote! { rsdsl::Tok::Group { delim: #d, inner: vec![ #(#inner_exprs),* ] } }
        }
    }
}

fn tokens_to_toks(ts: TokenStream2) -> Vec<TokAst> {
    ts.into_iter().map(|t| tt_to_tok(&t)).collect()
}

// -----------------------------
// AST for decls/rules/stmts
// -----------------------------

#[derive(Debug)]
enum DeclAst {
    Index {
        name: String,
        rhs_toks: Vec<TokAst>,
    },
    Enum {
        name: String,
        variants: Vec<(String, Option<String>)>,
    },
    Scenario {
        name: String,
        values: Vec<String>,
    },
    Pin {
        name: String,
        ty: String,
    },
    Fn {
        name: String,
        args: Vec<String>,
        ret: String,
        ret_opt: bool,
    },
    Var {
        kind: VarKindAst,
        name: String,
        indices: Vec<String>,
        ty: Option<String>,
        domain: Option<String>,
    },
}
#[derive(Debug, Clone, Copy)]
enum VarKindAst {
    Place,
    State,
    Shape,
    Sources,
}

#[derive(Debug)]
struct RuleAst {
    name: String,
    body: Vec<StmtAst>,
}

#[derive(Debug)]
enum StmtAst {
    Require(ExprAst),
    Def {
        lhs: VarRefAst,
        rhs: ExprAst,
    },
    ForAll {
        binders: Vec<(String, String)>,
        body: Vec<StmtAst>,
    },
    Let {
        name: String,
        value: ExprAst,
    },
    Add {
        target: VarRefAst,
        value: ExprAst,
        cond: Option<ExprAst>,
        exclude: bool,
    },
    ForceEq {
        lhs: ExprAst,
        rhs: ExprAst,
    },
    Feature {
        name: String,
        body: Vec<StmtAst>,
    },
}

#[derive(Debug, Clone)]
struct VarRefAst {
    name: String,
    indices: Vec<ExprAst>,
}

#[derive(Debug, Clone)]
struct ObjectiveAst {
    sense: ObjSenseAst,
    body: ExprAst,
}
#[derive(Debug, Clone, Copy)]
enum ObjSenseAst {
    Minimize,
    Maximize,
}

#[derive(Debug, Clone)]
struct BinderAst {
    toks: Vec<TokAst>,
}

#[derive(Debug, Clone)]
enum ExprAst {
    Sym(String),
    Lit(String),

    Var(VarRefAst),
    Call {
        name: String,
        args: Vec<ExprAst>,
    },
    NamedArg {
        name: String,
        value: Box<ExprAst>,
    },

    Not(Box<ExprAst>),
    And(Box<ExprAst>, Box<ExprAst>),
    Or(Box<ExprAst>, Box<ExprAst>),

    Add(Box<ExprAst>, Box<ExprAst>),
    Sub(Box<ExprAst>, Box<ExprAst>),
    Mul(Box<ExprAst>, Box<ExprAst>),

    Eq(Box<ExprAst>, Box<ExprAst>),
    Le(Box<ExprAst>, Box<ExprAst>),
    Ge(Box<ExprAst>, Box<ExprAst>),
    Lt(Box<ExprAst>, Box<ExprAst>),
    Gt(Box<ExprAst>, Box<ExprAst>),

    Implies(Box<ExprAst>, Box<ExprAst>),

    OrList(Vec<ExprAst>),
    Sum {
        binder: BinderAst,
        body: Box<ExprAst>,
    },

    Paren(Box<ExprAst>),
}

// -----------------------------
// Convert AST -> runtime rsdsl AST
// -----------------------------

fn decl_to_expr(d: DeclAst) -> TokenStream2 {
    match d {
        DeclAst::Index { name, rhs_toks } => {
            let rhs_exprs: Vec<TokenStream2> = rhs_toks.into_iter().map(tok_to_expr).collect();
            quote! { rsdsl::Decl::Index(rsdsl::IndexDecl{ name: #name.to_string(), rhs_toks: vec![ #(#rhs_exprs),* ] }) }
        }
        DeclAst::Enum { name, variants } => {
            let vars: Vec<TokenStream2> = variants
                .into_iter()
                .map(|(n, v)| {
                    let vexpr = match v {
                        Some(s) => quote! { Some(#s.to_string()) },
                        None => quote! { None },
                    };
                    quote! { rsdsl::EnumVariant{ name: #n.to_string(), value: #vexpr } }
                })
                .collect();
            quote! { rsdsl::Decl::Enum(rsdsl::EnumDecl{ name: #name.to_string(), variants: vec![ #(#vars),* ] }) }
        }
        DeclAst::Scenario { name, values } => {
            let vals: Vec<TokenStream2> = values
                .into_iter()
                .map(|s| quote! { #s.to_string() })
                .collect();
            quote! { rsdsl::Decl::Scenario(rsdsl::ScenarioDecl{ name: #name.to_string(), values: vec![ #(#vals),* ] }) }
        }
        DeclAst::Pin { name, ty } => {
            quote! { rsdsl::Decl::Pin(rsdsl::PinDecl{ name: #name.to_string(), ty: #ty.to_string() }) }
        }
        DeclAst::Fn {
            name,
            args,
            ret,
            ret_opt,
        } => {
            let args_ts: Vec<TokenStream2> = args
                .into_iter()
                .map(|s| quote! { #s.to_string() })
                .collect();
            quote! { rsdsl::Decl::Fn(rsdsl::FnDecl{ name: #name.to_string(), args: vec![ #(#args_ts),* ], ret: #ret.to_string(), ret_optional: #ret_opt }) }
        }
        DeclAst::Var {
            kind,
            name,
            indices,
            ty,
            domain,
        } => {
            let kind_ts = match kind {
                VarKindAst::Place => quote! { rsdsl::VarKind::Place },
                VarKindAst::State => quote! { rsdsl::VarKind::State },
                VarKindAst::Shape => quote! { rsdsl::VarKind::Shape },
                VarKindAst::Sources => quote! { rsdsl::VarKind::Sources },
            };
            let idx_ts: Vec<TokenStream2> = indices
                .into_iter()
                .map(|s| quote! { #s.to_string() })
                .collect();
            let ty_ts = match ty {
                Some(s) => quote! { Some(#s.to_string()) },
                None => quote! { None },
            };
            let dom_ts = match domain {
                Some(s) => quote! { Some(#s.to_string()) },
                None => quote! { None },
            };
            quote! {
                rsdsl::Decl::Var(rsdsl::VarDecl{
                    kind: #kind_ts,
                    name: #name.to_string(),
                    indices: vec![ #(#idx_ts),* ],
                    ty: #ty_ts,
                    domain_leq: #dom_ts,
                })
            }
        }
    }
}

fn rule_to_expr(r: RuleAst) -> TokenStream2 {
    let name = r.name;
    let body_exprs: Vec<TokenStream2> = r.body.into_iter().map(stmt_to_expr).collect();
    quote! { rsdsl::Rule { name: #name.to_string(), body: vec![ #(#body_exprs),* ] } }
}

fn stmt_to_expr(s: StmtAst) -> TokenStream2 {
    match s {
        StmtAst::Require(e) => {
            let e = expr_to_expr(e);
            quote! { rsdsl::Stmt::Require(#e) }
        }
        StmtAst::Def { lhs, rhs } => {
            let lhs = varref_to_expr(lhs);
            let rhs = expr_to_expr(rhs);
            quote! { rsdsl::Stmt::Def{ lhs: #lhs, op: rsdsl::DefOp::Iff, rhs: #rhs } }
        }
        StmtAst::Let { name, value } => {
            let value = expr_to_expr(value);
            quote! { rsdsl::Stmt::Let{ name: #name.to_string(), value: #value } }
        }
        StmtAst::Add {
            target,
            value,
            cond,
            exclude,
        } => {
            let target = varref_to_expr(target);
            let value = expr_to_expr(value);
            let cond = match cond {
                Some(c) => {
                    let c = expr_to_expr(c);
                    quote! { Some(#c) }
                }
                None => quote! { None },
            };
            quote! { rsdsl::Stmt::Add{ target: #target, value: #value, cond: #cond, exclude: #exclude } }
        }
        StmtAst::ForceEq { lhs, rhs } => {
            let lhs = expr_to_expr(lhs);
            let rhs = expr_to_expr(rhs);
            quote! { rsdsl::Stmt::ForceEq{ lhs: #lhs, rhs: #rhs } }
        }
        StmtAst::Feature { name, body } => {
            let body_exprs: Vec<TokenStream2> = body.into_iter().map(stmt_to_expr).collect();
            quote! { rsdsl::Stmt::Feature{ name: #name.to_string(), body: vec![ #(#body_exprs),* ] } }
        }
        StmtAst::ForAll { binders, body } => {
            let b_pairs: Vec<TokenStream2> = binders
                .into_iter()
                .map(|(v, d)| {
                    // String -> 토큰으로 안전하게 만들기 (식별자 아님! 그냥 "문자열 값")
                    let v_lit = syn::LitStr::new(&v, proc_macro2::Span::call_site());
                    let d_lit = syn::LitStr::new(&d, proc_macro2::Span::call_site());
                    quote! { (#v_lit.to_string(), #d_lit.to_string()) }
                })
                .collect();

            let body_exprs: Vec<TokenStream2> = body.into_iter().map(stmt_to_expr).collect();

            quote! {
                rsdsl::Stmt::ForAll {
                    binders: vec![ #(#b_pairs),* ],
                    body: vec![ #(#body_exprs),* ],
                }
            }
        }
    }
}

fn varref_to_expr(v: VarRefAst) -> TokenStream2 {
    let name = v.name;
    let idx: Vec<TokenStream2> = v.indices.into_iter().map(expr_to_expr).collect();
    quote! { rsdsl::VarRef{ name: #name.to_string(), indices: vec![ #(#idx),* ] } }
}

fn binder_to_expr(b: BinderAst) -> TokenStream2 {
    let toks_vec: Vec<TokenStream2> = b.toks.into_iter().map(tok_to_expr).collect();
    quote! { rsdsl::Binder{ toks: vec![ #(#toks_vec),* ] } }
}

fn expr_to_expr(e: ExprAst) -> TokenStream2 {
    match e {
        ExprAst::Sym(s) => quote! { rsdsl::Expr::Sym(#s.to_string()) },
        ExprAst::Lit(s) => quote! { rsdsl::Expr::Lit(#s.to_string()) },

        ExprAst::Var(v) => {
            let v = varref_to_expr(v);
            quote! { rsdsl::Expr::Var(#v) }
        }
        ExprAst::Call { name, args } => {
            let args_ts: Vec<TokenStream2> = args.into_iter().map(expr_to_expr).collect();
            quote! { rsdsl::Expr::Call{ name: #name.to_string(), args: vec![ #(#args_ts),* ] } }
        }
        ExprAst::NamedArg { name, value } => {
            let v = expr_to_expr(*value);
            quote! { rsdsl::Expr::NamedArg{ name: #name.to_string(), value: Box::new(#v) } }
        }

        ExprAst::Not(x) => {
            let x = expr_to_expr(*x);
            quote! { rsdsl::Expr::Not(Box::new(#x)) }
        }
        ExprAst::And(a, b) => {
            let a = expr_to_expr(*a);
            let b = expr_to_expr(*b);
            quote! { rsdsl::Expr::And(Box::new(#a),Box::new(#b)) }
        }
        ExprAst::Or(a, b) => {
            let a = expr_to_expr(*a);
            let b = expr_to_expr(*b);
            quote! { rsdsl::Expr::Or(Box::new(#a),Box::new(#b)) }
        }

        ExprAst::Add(a, b) => {
            let a = expr_to_expr(*a);
            let b = expr_to_expr(*b);
            quote! { rsdsl::Expr::Add(Box::new(#a),Box::new(#b)) }
        }
        ExprAst::Sub(a, b) => {
            let a = expr_to_expr(*a);
            let b = expr_to_expr(*b);
            quote! { rsdsl::Expr::Sub(Box::new(#a),Box::new(#b)) }
        }
        ExprAst::Mul(a, b) => {
            let a = expr_to_expr(*a);
            let b = expr_to_expr(*b);
            quote! { rsdsl::Expr::Mul(Box::new(#a),Box::new(#b)) }
        }

        ExprAst::Eq(a, b) => {
            let a = expr_to_expr(*a);
            let b = expr_to_expr(*b);
            quote! { rsdsl::Expr::Eq(Box::new(#a),Box::new(#b)) }
        }
        ExprAst::Le(a, b) => {
            let a = expr_to_expr(*a);
            let b = expr_to_expr(*b);
            quote! { rsdsl::Expr::Le(Box::new(#a),Box::new(#b)) }
        }
        ExprAst::Ge(a, b) => {
            let a = expr_to_expr(*a);
            let b = expr_to_expr(*b);
            quote! { rsdsl::Expr::Ge(Box::new(#a),Box::new(#b)) }
        }
        ExprAst::Lt(a, b) => {
            let a = expr_to_expr(*a);
            let b = expr_to_expr(*b);
            quote! { rsdsl::Expr::Lt(Box::new(#a),Box::new(#b)) }
        }
        ExprAst::Gt(a, b) => {
            let a = expr_to_expr(*a);
            let b = expr_to_expr(*b);
            quote! { rsdsl::Expr::Gt(Box::new(#a),Box::new(#b)) }
        }

        ExprAst::Implies(a, b) => {
            let a = expr_to_expr(*a);
            let b = expr_to_expr(*b);
            quote! { rsdsl::Expr::Implies(Box::new(#a),Box::new(#b)) }
        }

        ExprAst::OrList(xs) => {
            let xs: Vec<TokenStream2> = xs.into_iter().map(expr_to_expr).collect();
            quote! { rsdsl::Expr::OrList(vec![ #(#xs),* ]) }
        }
        ExprAst::Sum { binder, body } => {
            let b = binder_to_expr(binder);
            let body = expr_to_expr(*body);
            quote! { rsdsl::Expr::Sum{ binder: #b, body: Box::new(#body) } }
        }

        ExprAst::Paren(x) => {
            let x = expr_to_expr(*x);
            quote! { rsdsl::Expr::Paren(Box::new(#x)) }
        }
    }
}

fn objective_to_expr(o: ObjectiveAst) -> TokenStream2 {
    let sense = match o.sense {
        ObjSenseAst::Minimize => quote! { rsdsl::ObjSense::Minimize },
        ObjSenseAst::Maximize => quote! { rsdsl::ObjSense::Maximize },
    };
    let body = expr_to_expr(o.body);
    quote! { Some(rsdsl::Objective{ sense: #sense, body: #body }) }
}

// -----------------------------
// Parsing
// -----------------------------

fn parse_model(tokens: TokenStream2) -> SynResult<ModelAst> {
    let mut it = tokens.into_iter().peekable();

    let tt = it
        .next()
        .ok_or_else(|| syn::Error::new(Span::call_site(), "expected `model Name { ... }`"))?;
    match tt {
        TokenTree::Ident(id) if id == "model" => {}
        _ => return Err(syn::Error::new(tt.span(), "expected `model`")),
    }

    let name = match it.next() {
        Some(TokenTree::Ident(id)) => id.to_string(),
        Some(o) => return Err(syn::Error::new(o.span(), "expected model name ident")),
        None => return Err(syn::Error::new(Span::call_site(), "expected model name")),
    };

    let body = match it.next() {
        Some(TokenTree::Group(g)) if g.delimiter() == Delimiter::Brace => g.stream(),
        Some(o) => {
            return Err(syn::Error::new(
                o.span(),
                "expected `{...}` after model name",
            ))
        }
        None => {
            return Err(syn::Error::new(
                Span::call_site(),
                "expected `{...}` after model name",
            ))
        }
    };

    Ok(ModelAst { name, body })
}

fn parse_model_body(
    body: TokenStream2,
) -> SynResult<(Vec<DeclAst>, DeclEnv, Vec<RuleAst>, Option<ObjectiveAst>)> {
    let mut decls = vec![];
    let mut env = DeclEnv::new();
    let mut rules = vec![];
    let mut objective = None;

    let mut it = body.into_iter().peekable();
    while let Some(tt) = it.peek() {
        let kw = match tt {
            TokenTree::Ident(id) => id.to_string(),
            _ => {
                it.next();
                continue;
            }
        };

        match kw.as_str() {
            "index" => {
                it.next(); // index
                let name = expect_ident_peek(&mut it, "index name")?;
                expect_punct(&mut it, '=')?;
                let rhs = collect_until_semi_vec(&mut it)?;
                decls.push(DeclAst::Index {
                    name,
                    rhs_toks: rhs.into_iter().map(|t| tt_to_tok(&t)).collect(),
                });
            }
            "enum" => {
                it.next();
                let name = expect_ident_peek(&mut it, "enum name")?;
                let grp = expect_group(&mut it, Delimiter::Brace, "enum body { ... }")?;
                let variants = parse_enum_variants(grp.stream())?;
                decls.push(DeclAst::Enum { name, variants });
            }
            "scenario" => {
                it.next();
                let name = expect_ident_peek(&mut it, "scenario name")?;
                // allow optional `in`
                if matches!(it.peek(), Some(TokenTree::Ident(id)) if id=="in") {
                    it.next();
                }
                let grp = expect_group(&mut it, Delimiter::Brace, "scenario values { ... }")?;
                let values = parse_simple_csv_id_lit(grp.stream())?;
                expect_semi(&mut it)?;
                decls.push(DeclAst::Scenario { name, values });
            }
            "pin" => {
                it.next();
                let name = expect_ident_peek(&mut it, "pin name")?;
                expect_punct(&mut it, ':')?;
                let ty = expect_ident_peek(&mut it, "pin type")?;
                expect_semi(&mut it)?;
                decls.push(DeclAst::Pin { name, ty });
            }
            "fn" => {
                let item = collect_until_semi_ts(&mut it)?;
                let d = parse_fn_decl(item, &mut env)?;
                decls.push(d);
            }
            "place" | "state" | "shape" | "sources" => {
                let item = collect_until_semi_ts(&mut it)?;
                let d = parse_var_decl(item, &mut env)?;
                decls.push(d);
            }
            "rule" => {
                it.next();
                let name = expect_ident_peek(&mut it, "rule name")?;
                let grp = expect_group(&mut it, Delimiter::Brace, "rule body { ... }")?;
                let body = parse_rule_body(grp.stream(), &env)?;
                rules.push(RuleAst { name, body });
            }
            "objective" => {
                it.next();
                let sense = match it.next() {
                    Some(TokenTree::Ident(id)) if id == "minimize" => ObjSenseAst::Minimize,
                    Some(TokenTree::Ident(id)) if id == "maximize" => ObjSenseAst::Maximize,
                    Some(o) => {
                        return Err(syn::Error::new(
                            o.span(),
                            "expected `minimize` or `maximize`",
                        ))
                    }
                    None => {
                        return Err(syn::Error::new(
                            Span::call_site(),
                            "expected `minimize` or `maximize`",
                        ))
                    }
                };
                let grp = expect_group(&mut it, Delimiter::Brace, "objective body { ... }")?;
                let stmt_tokens: Vec<TokenTree> = grp.stream().into_iter().collect();
                let mut cur = Cursor::new(&stmt_tokens, &env);
                let body_expr = parse_expr(&mut cur, 0)?;
                cur.expect_eof()?;
                objective = Some(ObjectiveAst {
                    sense,
                    body: body_expr,
                });
            }
            _ => {
                return Err(syn::Error::new(
                    tt.span(),
                    format!("unsupported top-level keyword `{kw}` (no fallback)"),
                ));
            }
        }
    }

    Ok((decls, env, rules, objective))
}

fn parse_enum_variants(ts: TokenStream2) -> SynResult<Vec<(String, Option<String>)>> {
    // Very small parser: Variant or Variant=lit, comma-separated.
    let toks: Vec<TokenTree> = ts.into_iter().collect();
    let parts = split_top_level_commas(&toks);
    let mut out = vec![];
    for p in parts {
        if p.is_empty() {
            continue;
        }
        let mut it = p.into_iter();
        let name = match it.next() {
            Some(TokenTree::Ident(id)) => id.to_string(),
            Some(o) => return Err(syn::Error::new(o.span(), "expected enum variant ident")),
            None => continue,
        };
        let mut value: Option<String> = None;
        if let Some(TokenTree::Punct(eq)) = it.next() {
            if eq.as_char() == '=' {
                let v = it
                    .next()
                    .ok_or_else(|| syn::Error::new(eq.span(), "expected value after `=`"))?;
                value = Some(v.to_string());
            }
        }
        out.push((name, value));
    }
    Ok(out)
}

fn parse_simple_csv_id_lit(ts: TokenStream2) -> SynResult<Vec<String>> {
    let toks: Vec<TokenTree> = ts.into_iter().collect();
    let parts = split_top_level_commas(&toks);
    let mut out = vec![];
    for p in parts {
        if p.is_empty() {
            continue;
        }
        if p.len() != 1 {
            return Err(syn::Error::new(
                p[0].span(),
                "scenario values must be simple id/lit",
            ));
        }
        out.push(p[0].to_string());
    }
    Ok(out)
}

// --------------
// Decl parsers
// --------------

fn parse_fn_decl(item: TokenStream2, env: &mut DeclEnv) -> SynResult<DeclAst> {
    // fn name(args) -> Ret?;
    let toks: Vec<TokenTree> = item.into_iter().collect();
    let mut cur = SimpleCursor::new(&toks);

    cur.expect_ident("fn")?;
    let name = cur.expect_ident("fn name")?;

    let args_grp = cur.expect_group(Delimiter::Parenthesis, "fn args (...)")?;
    let args = parse_ident_list(args_grp.stream())?;

    cur.expect_punct('-')?;
    cur.expect_punct('>')?;

    let ret_tt = match cur.next() {
        Some(tt) => tt,
        None => return Err(syn::Error::new(cur.span_here(), "expected return type")),
    };
    let mut ret = ret_tt.to_string();
    let mut ret_opt = false;

    // optional '?'
    if matches!(cur.peek(), Some(TokenTree::Punct(p)) if p.as_char()=='?') {
        cur.next();
        ret_opt = true;
    }

    if !cur.is_eof() {
        return Err(syn::Error::new(
            cur.span_here(),
            "unexpected tokens in fn decl",
        ));
    }

    env.fns.insert(name.clone(), args.len());
    Ok(DeclAst::Fn {
        name,
        args,
        ret,
        ret_opt,
    })
}

fn parse_var_decl(item: TokenStream2, env: &mut DeclEnv) -> SynResult<DeclAst> {
    // <kind> Name[Idx,...] : Ty <= Domain
    let toks: Vec<TokenTree> = item.into_iter().collect();
    let mut cur = SimpleCursor::new(&toks);

    let kind_id = cur.expect_ident("var kind")?;
    let kind = match kind_id.as_str() {
        "place" => VarKindAst::Place,
        "state" => VarKindAst::State,
        "shape" => VarKindAst::Shape,
        "sources" => VarKindAst::Sources,
        _ => return Err(syn::Error::new(cur.span_here(), "unknown var kind")),
    };

    let name = cur.expect_ident("var name")?;
    let idx_grp = cur.expect_group(Delimiter::Bracket, "indices [..]")?;
    let indices = parse_ident_list(idx_grp.stream())?;
    env.vars.insert(name.clone(), indices.len());

    let mut ty: Option<String> = None;
    let mut domain: Option<String> = None;

    if matches!(cur.peek(), Some(TokenTree::Punct(p)) if p.as_char()==':') {
        cur.next();
        ty = Some(cur.expect_ident("type name")?);
    }

    // optional <= DomainIdent
    if let Some((TokenTree::Punct(a), TokenTree::Punct(b))) = cur.peek2() {
        if a.as_char() == '<' && b.as_char() == '=' {
            cur.next();
            cur.next();
            domain = Some(cur.expect_ident("domain name")?);
        }
    }

    if !cur.is_eof() {
        return Err(syn::Error::new(
            cur.span_here(),
            "unexpected tokens in var decl",
        ));
    }

    Ok(DeclAst::Var {
        kind,
        name,
        indices,
        ty,
        domain,
    })
}

// --------------
// Rule parsing
// --------------

fn parse_rule_body(ts: TokenStream2, env: &DeclEnv) -> SynResult<Vec<StmtAst>> {
    let mut out = vec![];
    let mut it = ts.into_iter().peekable();

    while it.peek().is_some() {
        match it.peek() {
            Some(TokenTree::Ident(id)) if id == "forall" => {
                it.next(); // consume "forall"
                let bind_grp =
                    expect_group(&mut it, Delimiter::Parenthesis, "forall binders (...)")?;
                let binders = parse_forall_binders(bind_grp.stream())?;

                let body_grp = expect_group(&mut it, Delimiter::Brace, "forall body { ... }")?;
                let body = parse_rule_body(body_grp.stream(), env)?; // 재귀

                out.push(StmtAst::ForAll { binders, body });

                // optional trailing ';'
                if matches!(it.peek(), Some(TokenTree::Punct(p)) if p.as_char()==';') {
                    it.next();
                }
            }
            Some(TokenTree::Ident(id)) if id == "feature" => {
                it.next();
                let name = expect_ident_peek(&mut it, "feature name")?;
                let grp = expect_group(&mut it, Delimiter::Brace, "feature body { ... }")?;
                let body = parse_rule_body(grp.stream(), env)?;
                out.push(StmtAst::Feature { name, body });
                // optional trailing semicolon
                if matches!(it.peek(), Some(TokenTree::Punct(p)) if p.as_char()==';') {
                    it.next();
                }
            }
            Some(_) => {
                // statement until ';'
                let stmt = collect_until_semi_ts_from_peekable(&mut it)?;
                let toks: Vec<TokenTree> = stmt.into_iter().collect();
                if toks.is_empty() {
                    continue;
                }
                let mut cur = Cursor::new(&toks, env);

                let kw = match cur.peek() {
                    Some(TokenTree::Ident(id)) => id.to_string(),
                    Some(o) => return Err(syn::Error::new(o.span(), "expected statement keyword")),
                    None => continue,
                };

                match kw.as_str() {
                    "require" => {
                        cur.next();
                        let e = parse_expr(&mut cur, 0)?;
                        cur.expect_eof()?;
                        out.push(StmtAst::Require(e));
                    }
                    "def" => {
                        cur.next();
                        let lhs = parse_varref(&mut cur)?;
                        // <-> operator
                        if !cur.consume_iff() {
                            return Err(syn::Error::new(
                                cur.span_here(),
                                "expected `<->` in `def`",
                            ));
                        }
                        let rhs = parse_expr(&mut cur, 0)?;
                        cur.expect_eof()?;
                        out.push(StmtAst::Def { lhs, rhs });
                    }
                    "let" => {
                        cur.next();
                        let name = cur.expect_ident("let name")?;
                        cur.expect_punct('=')?;
                        let value = parse_expr(&mut cur, 0)?;
                        cur.expect_eof()?;
                        out.push(StmtAst::Let { name, value });
                    }
                    "add" | "exclude" => {
                        let exclude = kw == "exclude";
                        cur.next();
                        let target = parse_varref(&mut cur)?;
                        cur.consume_pluseq()?; // +=
                        let value = parse_expr(&mut cur, 0)?;
                        let mut cond = None;
                        if matches!(cur.peek(), Some(TokenTree::Ident(id)) if id=="if") {
                            cur.next();
                            cond = Some(parse_expr(&mut cur, 0)?);
                        }
                        cur.expect_eof()?;
                        out.push(StmtAst::Add {
                            target,
                            value,
                            cond,
                            exclude,
                        });
                    }
                    "force" => {
                        cur.next();
                        let lhs = parse_expr_until_eqeq(&mut cur)?;
                        // ==
                        if !cur.consume_eqeq() {
                            return Err(syn::Error::new(cur.span_here(), "expected `==` in force"));
                        }
                        let rhs = parse_expr(&mut cur, 0)?;
                        cur.expect_eof()?;
                        out.push(StmtAst::ForceEq { lhs, rhs });
                    }
                    _ => {
                        return Err(syn::Error::new(
                            cur.span_here(),
                            format!("unsupported rule statement `{kw}` (no fallback)"),
                        ));
                    }
                }
            }
            None => break,
        }
    }
    Ok(out)
}
fn parse_forall_binders(ts: TokenStream2) -> SynResult<Vec<(String, String)>> {
    let toks: Vec<TokenTree> = ts.into_iter().collect();
    let mut i = 0usize;

    fn take_ident(toks: &[TokenTree], i: &mut usize) -> SynResult<String> {
        match toks.get(*i) {
            Some(TokenTree::Ident(id)) => {
                *i += 1;
                Ok(id.to_string())
            }
            Some(t) => Err(syn::Error::new(t.span(), "expected ident")),
            None => Err(syn::Error::new(Span::call_site(), "unexpected eof")),
        }
    }
    fn take_punct(toks: &[TokenTree], i: &mut usize, ch: char) -> bool {
        matches!(toks.get(*i), Some(TokenTree::Punct(p)) if p.as_char()==ch)
            .then(|| {
                *i += 1;
                ()
            })
            .is_some()
    }

    let mut out = vec![];

    loop {
        // vars: ident OR (a,b,...)
        let vars: Vec<String> = match toks.get(i) {
            Some(TokenTree::Ident(_)) => vec![take_ident(&toks, &mut i)?],
            Some(TokenTree::Group(g)) if g.delimiter() == Delimiter::Parenthesis => {
                i += 1;
                let inner: Vec<TokenTree> = g.stream().into_iter().collect();
                let mut j = 0usize;
                let mut vs = vec![take_ident(&inner, &mut j)?];
                while take_punct(&inner, &mut j, ',') {
                    vs.push(take_ident(&inner, &mut j)?);
                }
                if j != inner.len() {
                    return Err(syn::Error::new(
                        g.span(),
                        "unexpected token in binder tuple",
                    ));
                }
                vs
            }
            Some(t) => {
                return Err(syn::Error::new(
                    t.span(),
                    "expected binder var or (a,b,...)",
                ))
            }
            None => return Err(syn::Error::new(Span::call_site(), "unexpected eof")),
        };

        // expect `in`
        match toks.get(i) {
            Some(TokenTree::Ident(id)) if id == "in" => i += 1,
            Some(t) => return Err(syn::Error::new(t.span(), "expected `in`")),
            None => return Err(syn::Error::new(Span::call_site(), "unexpected eof")),
        }

        // domains: A or A*B*...
        let mut doms = vec![take_ident(&toks, &mut i)?];
        while take_punct(&toks, &mut i, '*') {
            doms.push(take_ident(&toks, &mut i)?);
        }

        if vars.len() != doms.len() {
            return Err(syn::Error::new(
                Span::call_site(),
                "binder var count must match domain count",
            ));
        }
        out.extend(vars.into_iter().zip(doms.into_iter()));

        // comma to continue
        if take_punct(&toks, &mut i, ',') {
            continue;
        }
        break;
    }

    if i != toks.len() {
        return Err(syn::Error::new(
            Span::call_site(),
            "unexpected token in forall binders",
        ));
    }
    Ok(out)
}

// -----------------------------
// Expression parsing (Pratt)
// -----------------------------

#[derive(Clone, Copy)]
enum Infix {
    Mul,
    Add,
    Sub,
    And,
    Or,
    Eq,
    Le,
    Ge,
    Lt,
    Gt,
    Implies,
}

fn infix_prec(op: Infix) -> u8 {
    match op {
        Infix::Mul => 50,
        Infix::Add | Infix::Sub => 40,
        Infix::And => 30,
        Infix::Or => 25,
        Infix::Eq | Infix::Le | Infix::Ge | Infix::Lt | Infix::Gt => 20,
        Infix::Implies => 10,
    }
}

fn parse_expr(cur: &mut Cursor, min_prec: u8) -> SynResult<ExprAst> {
    let mut lhs = parse_prefix(cur)?;

    loop {
        let Some((op, prec)) = cur.peek_infix() else {
            break;
        };
        if prec < min_prec {
            break;
        }
        cur.next_op(op); // consume operator tokens
        let rhs = parse_expr(cur, prec + 1)?;
        lhs = match op {
            Infix::Mul => ExprAst::Mul(Box::new(lhs), Box::new(rhs)),
            Infix::Add => ExprAst::Add(Box::new(lhs), Box::new(rhs)),
            Infix::Sub => ExprAst::Sub(Box::new(lhs), Box::new(rhs)),
            Infix::And => ExprAst::And(Box::new(lhs), Box::new(rhs)),
            Infix::Or => ExprAst::Or(Box::new(lhs), Box::new(rhs)),
            Infix::Eq => ExprAst::Eq(Box::new(lhs), Box::new(rhs)),
            Infix::Le => ExprAst::Le(Box::new(lhs), Box::new(rhs)),
            Infix::Ge => ExprAst::Ge(Box::new(lhs), Box::new(rhs)),
            Infix::Lt => ExprAst::Lt(Box::new(lhs), Box::new(rhs)),
            Infix::Gt => ExprAst::Gt(Box::new(lhs), Box::new(rhs)),
            Infix::Implies => ExprAst::Implies(Box::new(lhs), Box::new(rhs)),
        };
    }

    Ok(lhs)
}

fn parse_expr_until_eqeq(cur: &mut Cursor) -> SynResult<ExprAst> {
    let mut lhs = parse_prefix(cur)?;
    loop {
        if cur.peek_eqeq() {
            break;
        }
        let Some((op, prec)) = cur.peek_infix() else {
            break;
        };
        if matches!(op, Infix::Eq) {
            break;
        }
        cur.next_op(op);
        let rhs = parse_expr(cur, prec + 1)?;
        lhs = match op {
            Infix::Mul => ExprAst::Mul(Box::new(lhs), Box::new(rhs)),
            Infix::Add => ExprAst::Add(Box::new(lhs), Box::new(rhs)),
            Infix::Sub => ExprAst::Sub(Box::new(lhs), Box::new(rhs)),
            Infix::And => ExprAst::And(Box::new(lhs), Box::new(rhs)),
            Infix::Or => ExprAst::Or(Box::new(lhs), Box::new(rhs)),
            Infix::Le => ExprAst::Le(Box::new(lhs), Box::new(rhs)),
            Infix::Ge => ExprAst::Ge(Box::new(lhs), Box::new(rhs)),
            Infix::Lt => ExprAst::Lt(Box::new(lhs), Box::new(rhs)),
            Infix::Gt => ExprAst::Gt(Box::new(lhs), Box::new(rhs)),
            Infix::Implies => ExprAst::Implies(Box::new(lhs), Box::new(rhs)),
            Infix::Eq => unreachable!("stopped before =="),
        };
    }
    Ok(lhs)
}

fn parse_prefix(cur: &mut Cursor) -> SynResult<ExprAst> {
    // keyword `not` or punct `!`
    if cur.consume_bang_or_not() {
        let x = parse_prefix(cur)?;
        return Ok(ExprAst::Not(Box::new(x)));
    }

    parse_primary(cur)
}

fn parse_primary(cur: &mut Cursor) -> SynResult<ExprAst> {
    let tt = cur
        .next()
        .ok_or_else(|| syn::Error::new(cur.span_here(), "unexpected end of expr"))?;

    match tt {
        TokenTree::Group(g) if g.delimiter() == Delimiter::Parenthesis => {
            let inner: Vec<TokenTree> = g.stream().into_iter().collect();
            let mut c2 = Cursor::new(&inner, cur.env);
            let e = parse_expr(&mut c2, 0)?;
            c2.expect_eof()?;
            Ok(ExprAst::Paren(Box::new(e)))
        }
        TokenTree::Ident(id) => {
            let name = id.to_string();

            // Special OR{...}
            if name == "OR" {
                if let Some(TokenTree::Group(g)) = cur.peek() {
                    if g.delimiter() == Delimiter::Brace {
                        let g = match cur.next().unwrap() {
                            TokenTree::Group(g) => g,
                            _ => unreachable!(),
                        };
                        let xs = parse_expr_list(g.stream(), cur.env)?;
                        return Ok(ExprAst::OrList(xs));
                    }
                }
            }

            // Special sum(binder) expr
            if name == "sum" {
                let g = cur.expect_group(Delimiter::Parenthesis, "sum(binder)")?;
                let binder = BinderAst {
                    toks: g.stream().into_iter().map(|t| tt_to_tok(&t)).collect(),
                };
                // Body: parse as prefix so `sum(..) X[...]` works; then allow higher ops by returning to caller.
                let body = parse_prefix(cur)?;
                return Ok(ExprAst::Sum {
                    binder,
                    body: Box::new(body),
                });
            }

            // Call or VarRef or Symbol
            if let Some(TokenTree::Group(g)) = cur.peek() {
                if g.delimiter() == Delimiter::Parenthesis {
                    let g = match cur.next().unwrap() {
                        TokenTree::Group(g) => g,
                        _ => unreachable!(),
                    };
                    let args = parse_call_args(g.stream(), cur.env)?;
                    // arity check if declared fn
                    if let Some(&decl) = cur.env.fns.get(&name) {
                        if args.len() != decl {
                            return Err(syn::Error::new(
                                id.span(),
                                format!(
                                    "argcount mismatch for `{name}`: declared {decl}, used {}",
                                    args.len()
                                ),
                            ));
                        }
                    }
                    return Ok(ExprAst::Call { name, args });
                }
                if g.delimiter() == Delimiter::Bracket {
                    let g = match cur.next().unwrap() {
                        TokenTree::Group(g) => g,
                        _ => unreachable!(),
                    };
                    let indices = parse_index_exprs(g.stream(), cur.env)?;
                    // arity check for vars
                    if let Some(&decl) = cur.env.vars.get(&name) {
                        if indices.len() != decl {
                            return Err(syn::Error::new(
                                id.span(),
                                format!(
                                    "arity mismatch for `{name}`: declared {decl}, used {}",
                                    indices.len()
                                ),
                            ));
                        }
                    }
                    return Ok(ExprAst::Var(VarRefAst { name, indices }));
                }
            }

            // keyword operators as identifiers handled in infix peek; here treat as symbol
            Ok(ExprAst::Sym(name))
        }
        TokenTree::Literal(l) => Ok(ExprAst::Lit(l.to_string())),
        TokenTree::Punct(p) if p.as_char() == '-' => {
            // unary minus: treat as `0 - expr`
            let rhs = parse_prefix(cur)?;
            Ok(ExprAst::Sub(
                Box::new(ExprAst::Lit("0".to_string())),
                Box::new(rhs),
            ))
        }
        other => Err(syn::Error::new(other.span(), "unexpected token in expr")),
    }
}

fn parse_index_exprs(ts: TokenStream2, env: &DeclEnv) -> SynResult<Vec<ExprAst>> {
    let toks: Vec<TokenTree> = ts.into_iter().collect();
    if toks.is_empty() {
        return Ok(vec![]);
    }
    let parts = split_top_level_commas(&toks);
    let mut out = vec![];
    for p in parts {
        let mut c = Cursor::new(&p, env);
        let e = parse_expr(&mut c, 0)?;
        c.expect_eof()?;
        out.push(e);
    }
    Ok(out)
}

fn parse_expr_list(ts: TokenStream2, env: &DeclEnv) -> SynResult<Vec<ExprAst>> {
    parse_index_exprs(ts, env)
}

fn parse_call_args(ts: TokenStream2, env: &DeclEnv) -> SynResult<Vec<ExprAst>> {
    let toks: Vec<TokenTree> = ts.into_iter().collect();
    if toks.is_empty() {
        return Ok(vec![]);
    }
    let parts = split_top_level_commas(&toks);
    let mut out = vec![];
    for p in parts {
        // Named arg: ident '=' expr
        if p.len() >= 3 {
            if let (TokenTree::Ident(id), TokenTree::Punct(eq)) = (&p[0], &p[1]) {
                if eq.as_char() == '=' {
                    let mut c = Cursor::new(&p[2..], env);
                    let v = parse_expr(&mut c, 0)?;
                    c.expect_eof()?;
                    out.push(ExprAst::NamedArg {
                        name: id.to_string(),
                        value: Box::new(v),
                    });
                    continue;
                }
            }
        }
        let mut c = Cursor::new(&p, env);
        let e = parse_expr(&mut c, 0)?;
        c.expect_eof()?;
        out.push(e);
    }
    Ok(out)
}

fn parse_varref(cur: &mut Cursor) -> SynResult<VarRefAst> {
    let name = cur.expect_ident("var name")?;
    let g = cur.expect_group(Delimiter::Bracket, "var indices [...]")?;
    let indices = parse_index_exprs(g.stream(), cur.env)?;
    if let Some(&decl) = cur.env.vars.get(&name) {
        if indices.len() != decl {
            return Err(syn::Error::new(
                cur.last_span,
                format!(
                    "arity mismatch for `{name}`: declared {decl}, used {}",
                    indices.len()
                ),
            ));
        }
    }
    Ok(VarRefAst { name, indices })
}

// -----------------------------
// Cursor utilities
// -----------------------------

struct Cursor<'a> {
    toks: &'a [TokenTree],
    pos: usize,
    env: &'a DeclEnv,
    last_span: Span,
}
impl<'a> Cursor<'a> {
    fn new(toks: &'a [TokenTree], env: &'a DeclEnv) -> Self {
        Self {
            toks,
            pos: 0,
            env,
            last_span: Span::call_site(),
        }
    }
    fn peek(&self) -> Option<&TokenTree> {
        self.toks.get(self.pos)
    }
    fn next(&mut self) -> Option<TokenTree> {
        let t = self.toks.get(self.pos).cloned();
        if let Some(ref tt) = t {
            self.last_span = tt.span();
        }
        if t.is_some() {
            self.pos += 1;
        }
        t
    }
    fn span_here(&self) -> Span {
        self.peek().map(|t| t.span()).unwrap_or(Span::call_site())
    }
    fn expect_eof(&self) -> SynResult<()> {
        if self.pos < self.toks.len() {
            Err(syn::Error::new(
                self.span_here(),
                "unexpected trailing tokens",
            ))
        } else {
            Ok(())
        }
    }
    fn expect_ident(&mut self, what: &str) -> SynResult<String> {
        match self.next() {
            Some(TokenTree::Ident(id)) => Ok(id.to_string()),
            Some(o) => Err(syn::Error::new(o.span(), format!("expected {what} ident"))),
            None => Err(syn::Error::new(
                Span::call_site(),
                format!("expected {what} ident"),
            )),
        }
    }
    fn expect_punct(&mut self, ch: char) -> SynResult<()> {
        match self.next() {
            Some(TokenTree::Punct(p)) if p.as_char() == ch => Ok(()),
            Some(o) => Err(syn::Error::new(o.span(), format!("expected `{ch}`"))),
            None => Err(syn::Error::new(
                Span::call_site(),
                format!("expected `{ch}`"),
            )),
        }
    }
    fn expect_group(&mut self, delim: Delimiter, what: &str) -> SynResult<Group> {
        match self.next() {
            Some(TokenTree::Group(g)) if g.delimiter() == delim => Ok(g),
            Some(o) => Err(syn::Error::new(o.span(), format!("expected {what}"))),
            None => Err(syn::Error::new(
                Span::call_site(),
                format!("expected {what}"),
            )),
        }
    }
    fn consume_iff(&mut self) -> bool {
        if self.pos + 2 >= self.toks.len() {
            return false;
        }
        match (
            &self.toks[self.pos],
            &self.toks[self.pos + 1],
            &self.toks[self.pos + 2],
        ) {
            (TokenTree::Punct(a), TokenTree::Punct(b), TokenTree::Punct(c))
                if a.as_char() == '<' && b.as_char() == '-' && c.as_char() == '>' =>
            {
                self.pos += 3;
                true
            }
            _ => false,
        }
    }
    fn consume_pluseq(&mut self) -> SynResult<()> {
        // + =
        match (self.next(), self.next()) {
            (Some(TokenTree::Punct(a)), Some(TokenTree::Punct(b)))
                if a.as_char() == '+' && b.as_char() == '=' =>
            {
                Ok(())
            }
            (Some(o), _) => Err(syn::Error::new(o.span(), "expected `+=`")),
            _ => Err(syn::Error::new(Span::call_site(), "expected `+=`")),
        }
    }
    fn consume_eqeq(&mut self) -> bool {
        if self.pos + 1 >= self.toks.len() {
            return false;
        }
        match (&self.toks[self.pos], &self.toks[self.pos + 1]) {
            (TokenTree::Punct(a), TokenTree::Punct(b))
                if a.as_char() == '=' && b.as_char() == '=' =>
            {
                self.pos += 2;
                true
            }

            _ => false,
        }
    }

    fn peek_eqeq(&self) -> bool {
        if self.pos + 1 >= self.toks.len() {
            return false;
        }
        matches!(
            (&self.toks[self.pos], &self.toks[self.pos+1]),
            (TokenTree::Punct(a), TokenTree::Punct(b)) if a.as_char()=='=' && b.as_char()=='='
        )
    }

    fn consume_bang_or_not(&mut self) -> bool {
        match self.peek() {
            Some(TokenTree::Punct(p)) if p.as_char() == '!' => {
                self.next();
                true
            }
            Some(TokenTree::Ident(id)) if id == "not" => {
                self.next();
                true
            }
            _ => false,
        }
    }

    fn peek_infix(&self) -> Option<(Infix, u8)> {
        // allow identifier keywords `and` / `or`
        match self.peek()? {
            TokenTree::Punct(p) if p.as_char() == '*' => Some((Infix::Mul, infix_prec(Infix::Mul))),
            TokenTree::Punct(p) if p.as_char() == '+' => Some((Infix::Add, infix_prec(Infix::Add))),
            TokenTree::Punct(p) if p.as_char() == '&' => Some((Infix::And, infix_prec(Infix::And))),
            TokenTree::Punct(p) if p.as_char() == '|' => Some((Infix::Or, infix_prec(Infix::Or))),
            TokenTree::Ident(id) if id == "and" => Some((Infix::And, infix_prec(Infix::And))),
            TokenTree::Ident(id) if id == "or" => Some((Infix::Or, infix_prec(Infix::Or))),
            TokenTree::Punct(p) if p.as_char() == '<' => {
                // <= or <
                if self.pos + 1 < self.toks.len() {
                    if let TokenTree::Punct(n) = &self.toks[self.pos + 1] {
                        if n.as_char() == '=' {
                            return Some((Infix::Le, infix_prec(Infix::Le)));
                        }
                    }
                }
                Some((Infix::Lt, infix_prec(Infix::Lt)))
            }
            TokenTree::Punct(p) if p.as_char() == '>' => {
                if self.pos + 1 < self.toks.len() {
                    if let TokenTree::Punct(n) = &self.toks[self.pos + 1] {
                        if n.as_char() == '=' {
                            return Some((Infix::Ge, infix_prec(Infix::Ge)));
                        }
                    }
                }
                Some((Infix::Gt, infix_prec(Infix::Gt)))
            }
            TokenTree::Punct(p) if p.as_char() == '=' => {
                if self.pos + 1 < self.toks.len() {
                    if let TokenTree::Punct(n) = &self.toks[self.pos + 1] {
                        if n.as_char() == '=' {
                            return Some((Infix::Eq, infix_prec(Infix::Eq)));
                        }
                    }
                }
                None
            }
            TokenTree::Punct(p) if p.as_char() == '-' => {
                // `->` implies, otherwise subtraction
                if self.pos + 1 < self.toks.len() {
                    if let TokenTree::Punct(n) = &self.toks[self.pos + 1] {
                        if n.as_char() == '>' {
                            return Some((Infix::Implies, infix_prec(Infix::Implies)));
                        }
                    }
                }
                Some((Infix::Sub, infix_prec(Infix::Sub)))
            }
            _ => None,
        }
    }

    fn next_op(&mut self, op: Infix) {
        match op {
            Infix::Mul | Infix::Add | Infix::Sub => {
                self.next();
            }
            Infix::And | Infix::Or => {
                // could be ident keyword
                match self.peek() {
                    Some(TokenTree::Ident(_)) => {
                        self.next();
                    }
                    _ => {
                        self.next();
                    }
                }
            }
            Infix::Eq => {
                self.pos += 2;
            }
            Infix::Le | Infix::Ge => {
                self.pos += 2;
            }
            Infix::Lt | Infix::Gt => {
                self.next();
            }
            Infix::Implies => {
                self.pos += 2;
            }
        }
    }
}

// A tiny cursor for simple decl parsing
struct SimpleCursor<'a> {
    toks: &'a [TokenTree],
    pos: usize,
}
impl<'a> SimpleCursor<'a> {
    fn new(toks: &'a [TokenTree]) -> Self {
        Self { toks, pos: 0 }
    }
    fn peek(&self) -> Option<&TokenTree> {
        self.toks.get(self.pos)
    }
    fn peek2(&self) -> Option<(&TokenTree, &TokenTree)> {
        Some((self.toks.get(self.pos)?, self.toks.get(self.pos + 1)?))
    }
    fn next(&mut self) -> Option<&TokenTree> {
        let t = self.toks.get(self.pos);
        if t.is_some() {
            self.pos += 1;
        }
        t
    }
    fn span_here(&self) -> Span {
        self.peek().map(|t| t.span()).unwrap_or(Span::call_site())
    }
    fn is_eof(&self) -> bool {
        self.pos >= self.toks.len()
    }
    fn expect_ident(&mut self, what: &str) -> SynResult<String> {
        match self.next() {
            Some(TokenTree::Ident(id)) => Ok(id.to_string()),
            Some(o) => Err(syn::Error::new(o.span(), format!("expected {what} ident"))),
            None => Err(syn::Error::new(
                Span::call_site(),
                format!("expected {what} ident"),
            )),
        }
    }
    fn expect_punct(&mut self, ch: char) -> SynResult<()> {
        match self.next() {
            Some(TokenTree::Punct(p)) if p.as_char() == ch => Ok(()),
            Some(o) => Err(syn::Error::new(o.span(), format!("expected `{ch}`"))),
            None => Err(syn::Error::new(
                Span::call_site(),
                format!("expected `{ch}`"),
            )),
        }
    }
    fn expect_group(&mut self, delim: Delimiter, what: &str) -> SynResult<Group> {
        match self.next() {
            Some(TokenTree::Group(g)) if g.delimiter() == delim => Ok(g.clone()),
            Some(o) => Err(syn::Error::new(o.span(), format!("expected {what}"))),
            None => Err(syn::Error::new(
                Span::call_site(),
                format!("expected {what}"),
            )),
        }
    }
}

// -----------------------------
// Model-body helper token consumers
// -----------------------------

fn expect_ident_peek(
    it: &mut std::iter::Peekable<proc_macro2::token_stream::IntoIter>,
    what: &str,
) -> SynResult<String> {
    match it.next() {
        Some(TokenTree::Ident(id)) => Ok(id.to_string()),
        Some(o) => Err(syn::Error::new(o.span(), format!("expected {what} ident"))),
        None => Err(syn::Error::new(
            Span::call_site(),
            format!("expected {what} ident"),
        )),
    }
}

fn expect_punct(
    it: &mut std::iter::Peekable<proc_macro2::token_stream::IntoIter>,
    ch: char,
) -> SynResult<()> {
    match it.next() {
        Some(TokenTree::Punct(p)) if p.as_char() == ch => Ok(()),
        Some(o) => Err(syn::Error::new(o.span(), format!("expected `{ch}`"))),
        None => Err(syn::Error::new(
            Span::call_site(),
            format!("expected `{ch}`"),
        )),
    }
}
fn expect_semi(it: &mut std::iter::Peekable<proc_macro2::token_stream::IntoIter>) -> SynResult<()> {
    expect_punct(it, ';')
}
fn expect_group(
    it: &mut std::iter::Peekable<proc_macro2::token_stream::IntoIter>,
    delim: Delimiter,
    what: &str,
) -> SynResult<Group> {
    match it.next() {
        Some(TokenTree::Group(g)) if g.delimiter() == delim => Ok(g),
        Some(o) => Err(syn::Error::new(o.span(), format!("expected {what}"))),
        None => Err(syn::Error::new(
            Span::call_site(),
            format!("expected {what}"),
        )),
    }
}

fn collect_until_semi_ts(
    it: &mut std::iter::Peekable<proc_macro2::token_stream::IntoIter>,
) -> SynResult<TokenStream2> {
    let mut out = TokenStream2::new();
    // take first kw token too
    while let Some(tt) = it.next() {
        if matches!(&tt, TokenTree::Punct(p) if p.as_char()==';') {
            break;
        }
        out.extend([tt]);
    }
    Ok(out)
}

fn collect_until_semi_vec(
    it: &mut std::iter::Peekable<proc_macro2::token_stream::IntoIter>,
) -> SynResult<Vec<TokenTree>> {
    let mut out = vec![];
    while let Some(tt) = it.next() {
        if matches!(&tt, TokenTree::Punct(p) if p.as_char()==';') {
            break;
        }
        out.push(tt);
    }
    Ok(out)
}

fn collect_until_semi_ts_from_peekable(
    it: &mut std::iter::Peekable<proc_macro2::token_stream::IntoIter>,
) -> SynResult<TokenStream2> {
    let mut out = TokenStream2::new();
    while let Some(tt) = it.next() {
        if matches!(&tt, TokenTree::Punct(p) if p.as_char()==';') {
            break;
        }
        out.extend([tt]);
    }
    Ok(out)
}

fn split_top_level_commas(tokens: &[TokenTree]) -> Vec<Vec<TokenTree>> {
    let mut out = vec![];
    let mut cur = vec![];
    for tt in tokens {
        match tt {
            TokenTree::Punct(p) if p.as_char() == ',' => {
                out.push(cur);
                cur = vec![];
            }
            _ => cur.push(tt.clone()),
        }
    }
    out.push(cur);
    out
}

fn parse_ident_list(ts: TokenStream2) -> SynResult<Vec<String>> {
    let toks: Vec<TokenTree> = ts.into_iter().collect();
    if toks.is_empty() {
        return Ok(vec![]);
    }
    let parts = split_top_level_commas(&toks);
    let mut out = vec![];
    for p in parts {
        if p.len() != 1 {
            return Err(syn::Error::new(p[0].span(), "expected identifier"));
        }
        match &p[0] {
            TokenTree::Ident(id) => out.push(id.to_string()),
            o => return Err(syn::Error::new(o.span(), "expected identifier")),
        }
    }
    Ok(out)
}
