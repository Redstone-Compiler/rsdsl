use proc_macro::TokenStream;
use proc_macro2::{Delimiter, TokenTree};
use quote::quote;

#[proc_macro]
pub fn rsdsl(input: TokenStream) -> TokenStream {
    let ts: proc_macro2::TokenStream = input.into();
    let mut p = Parser::new(ts.into_iter().collect());
    let spec = match p.parse_model() {
        Ok(s) => s,
        Err(e) => return e.to_compile_error().into(),
    };
    spec.into_token_stream().into()
}

#[derive(Debug)]
struct ParseError {
    msg: String,
}
impl ParseError {
    fn new(msg: impl Into<String>) -> Self {
        Self { msg: msg.into() }
    }
    fn to_compile_error(&self) -> proc_macro2::TokenStream {
        let m = &self.msg;
        quote! { compile_error!(#m); }
    }
}

type PResult<T> = Result<T, ParseError>;

#[derive(Clone, Debug)]
struct ModelSpecAst {
    name: String,
    decls: Vec<DeclAst>,
    rules: Vec<RuleAst>,
    objective: Option<ObjectiveAst>,
}
#[derive(Clone, Debug)]
enum DeclAst {
    Index {
        name: String,
        domain: String,
    },
    Enum {
        name: String,
        variants: Vec<String>,
    },
    Scenario {
        name: String,
        values: Vec<i32>,
    },
    Pin {
        name: String,
        ty: String,
    },
    Fn {
        name: String,
        args: Vec<(String, String)>,
        ret: String,
    },
    Var {
        kind: String,
        name: String,
        indices: Vec<String>,
        ty: String,
    },
}
#[derive(Clone, Debug)]
struct RuleAst {
    name: String,
    body: Vec<StmtAst>,
}

#[derive(Clone, Debug)]
enum StmtAst {
    ForAll {
        binders: Vec<BinderAst>,
        body: Vec<StmtAst>,
    },
    Feature {
        name: String,
        body: Vec<StmtAst>,
    },
    Let {
        name: String,
        expr: ExprAst,
    },
    Require(ExprAst),
    Def {
        lhs: VarRefAst,
        rhs: ExprAst,
    },
    Add {
        exclude: bool,
        target: VarRefAst,
        value: ExprAst,
        cond: Option<ExprAst>,
    },
    ForceEq {
        lhs: ExprAst,
        rhs: ExprAst,
    },
}

#[derive(Clone, Debug)]
struct BinderAst {
    vars: Vec<String>,
    domains: Vec<String>,
}

#[derive(Clone, Debug)]
struct VarRefAst {
    name: String,
    indices: Vec<ExprAst>,
}

#[derive(Clone, Debug)]
enum ExprAst {
    Sym(String),
    Lit(f64),
    Var(VarRefAst),
    NamedArg {
        name: String,
        value: Box<ExprAst>,
    },
    Not(Box<ExprAst>),
    Neg(Box<ExprAst>),
    And(Box<ExprAst>, Box<ExprAst>),
    Or(Box<ExprAst>, Box<ExprAst>),
    Implies(Box<ExprAst>, Box<ExprAst>),
    Iff(Box<ExprAst>, Box<ExprAst>),
    Add(Box<ExprAst>, Box<ExprAst>),
    Sub(Box<ExprAst>, Box<ExprAst>),
    Mul(Box<ExprAst>, Box<ExprAst>),
    Eq(Box<ExprAst>, Box<ExprAst>),
    Le(Box<ExprAst>, Box<ExprAst>),
    Ge(Box<ExprAst>, Box<ExprAst>),
    Lt(Box<ExprAst>, Box<ExprAst>),
    Gt(Box<ExprAst>, Box<ExprAst>),
    OrList(Vec<ExprAst>),
    Call {
        name: String,
        args: Vec<ExprAst>,
    },
    Sum {
        binders: Vec<BinderAst>,
        body: Box<ExprAst>,
    },
    Paren(Box<ExprAst>),
}

#[derive(Clone, Debug)]
struct ObjectiveAst {
    sense: String,
    expr: ExprAst,
}

impl ModelSpecAst {
    fn into_token_stream(self) -> proc_macro2::TokenStream {
        let name = self.name;
        let decls = self
            .decls
            .into_iter()
            .map(|d| d.into_ts())
            .collect::<Vec<_>>();
        let rules = self
            .rules
            .into_iter()
            .map(|r| r.into_ts())
            .collect::<Vec<_>>();
        let objective = match self.objective {
            Some(o) => {
                let o = o.into_ts();
                quote! { Some(#o) }
            }
            None => quote! { None },
        };

        quote! {
            rsdsl::ModelSpec {
                name: #name.to_string(),
                decls: vec![ #(#decls),* ],
                rules: vec![ #(#rules),* ],
                objective: #objective,
            }
        }
    }
}

impl DeclAst {
    fn into_ts(self) -> proc_macro2::TokenStream {
        match self {
            DeclAst::Index { name, domain } => {
                quote! { rsdsl::Decl::Index{ name: #name.to_string(), domain: #domain.to_string() } }
            }
            DeclAst::Enum { name, variants } => {
                let vs = variants
                    .into_iter()
                    .map(|v| quote! { #v.to_string() })
                    .collect::<Vec<_>>();
                quote! { rsdsl::Decl::Enum{ name: #name.to_string(), variants: vec![#(#vs),*] } }
            }
            DeclAst::Scenario { name, values } => {
                let vals = values
                    .into_iter()
                    .map(|v| quote! { #v })
                    .collect::<Vec<_>>();
                quote! { rsdsl::Decl::Scenario{ name: #name.to_string(), values: vec![#(#vals),*] } }
            }
            DeclAst::Pin { name, ty } => {
                quote! { rsdsl::Decl::Pin{ name: #name.to_string(), ty: #ty.to_string() } }
            }
            DeclAst::Fn { name, args, ret } => {
                let args_ts = args
                    .into_iter()
                    .map(|(a, t)| quote! { (#a.to_string(), #t.to_string()) })
                    .collect::<Vec<_>>();
                quote! { rsdsl::Decl::Fn{ name: #name.to_string(), args: vec![#(#args_ts),*], ret: #ret.to_string() } }
            }
            DeclAst::Var {
                kind,
                name,
                indices,
                ty,
            } => {
                let k = match kind.as_str() {
                    "place" => quote! { rsdsl::VarKind::Place },
                    "state" => quote! { rsdsl::VarKind::State },
                    "shape" => quote! { rsdsl::VarKind::Shape },
                    "sources" => quote! { rsdsl::VarKind::Sources },
                    _ => quote! { rsdsl::VarKind::Place },
                };
                let idx_ts = indices
                    .into_iter()
                    .map(|x| quote! { #x.to_string() })
                    .collect::<Vec<_>>();
                quote! { rsdsl::Decl::Var{ kind: #k, name: #name.to_string(), indices: vec![#(#idx_ts),*], ty: #ty.to_string() } }
            }
        }
    }
}

impl RuleAst {
    fn into_ts(self) -> proc_macro2::TokenStream {
        let name = self.name;
        let body = self
            .body
            .into_iter()
            .map(|s| s.into_ts())
            .collect::<Vec<_>>();
        quote! { rsdsl::Rule{ name: #name.to_string(), body: vec![#(#body),*] } }
    }
}

impl BinderAst {
    fn into_ts(self) -> proc_macro2::TokenStream {
        let vars = self
            .vars
            .into_iter()
            .map(|v| quote! { #v.to_string() })
            .collect::<Vec<_>>();
        let doms = self
            .domains
            .into_iter()
            .map(|d| quote! { #d.to_string() })
            .collect::<Vec<_>>();
        quote! { rsdsl::Binder{ vars: vec![#(#vars),*], domains: vec![#(#doms),*] } }
    }
}

impl VarRefAst {
    fn into_ts(self) -> proc_macro2::TokenStream {
        let name = self.name;
        let idx = self
            .indices
            .into_iter()
            .map(|e| e.into_ts())
            .collect::<Vec<_>>();
        quote! { rsdsl::VarRef{ name: #name.to_string(), indices: vec![#(#idx),*] } }
    }
}

impl StmtAst {
    fn into_ts(self) -> proc_macro2::TokenStream {
        match self {
            StmtAst::ForAll { binders, body } => {
                let b = binders.into_iter().map(|x| x.into_ts()).collect::<Vec<_>>();
                let s = body.into_iter().map(|x| x.into_ts()).collect::<Vec<_>>();
                quote! { rsdsl::Stmt::ForAll{ binders: vec![#(#b),*], body: vec![#(#s),*] } }
            }
            StmtAst::Feature { name, body } => {
                let s = body.into_iter().map(|x| x.into_ts()).collect::<Vec<_>>();
                quote! { rsdsl::Stmt::Feature{ name: #name.to_string(), body: vec![#(#s),*] } }
            }
            StmtAst::Let { name, expr } => {
                let e = expr.into_ts();
                quote! { rsdsl::Stmt::Let{ name: #name.to_string(), expr: #e } }
            }
            StmtAst::Require(e) => {
                let e = e.into_ts();
                quote! { rsdsl::Stmt::Require(#e) }
            }
            StmtAst::Def { lhs, rhs } => {
                let lhs = lhs.into_ts();
                let rhs = rhs.into_ts();
                quote! { rsdsl::Stmt::Def{ lhs: #lhs, rhs: #rhs } }
            }
            StmtAst::Add {
                exclude,
                target,
                value,
                cond,
            } => {
                let t = target.into_ts();
                let v = value.into_ts();
                let c = match cond {
                    Some(e) => {
                        let e = e.into_ts();
                        quote! { Some(#e) }
                    }
                    None => quote! { None },
                };
                quote! { rsdsl::Stmt::Add{ exclude: #exclude, target: #t, value: #v, cond: #c } }
            }
            StmtAst::ForceEq { lhs, rhs } => {
                let l = lhs.into_ts();
                let r = rhs.into_ts();
                quote! { rsdsl::Stmt::ForceEq{ lhs: #l, rhs: #r } }
            }
        }
    }
}

impl ExprAst {
    fn into_ts(self) -> proc_macro2::TokenStream {
        match self {
            ExprAst::Sym(s) => quote! { rsdsl::Expr::Sym(#s.to_string()) },
            ExprAst::Lit(v) => quote! { rsdsl::Expr::Lit(#v) },
            ExprAst::Var(vr) => {
                let vr = vr.into_ts();
                quote! { rsdsl::Expr::Var(#vr) }
            }
            ExprAst::NamedArg { name, value } => {
                let v = (*value).into_ts();
                quote! { rsdsl::Expr::NamedArg{ name: #name.to_string(), value: Box::new(#v) } }
            }
            ExprAst::Not(x) => {
                let x = (*x).into_ts();
                quote! { rsdsl::Expr::Not(Box::new(#x)) }
            }
            ExprAst::Neg(x) => {
                let x = (*x).into_ts();
                quote! { rsdsl::Expr::Neg(Box::new(#x)) }
            }
            ExprAst::And(a, b) => {
                let a = (*a).into_ts();
                let b = (*b).into_ts();
                quote! { rsdsl::Expr::And(Box::new(#a), Box::new(#b)) }
            }
            ExprAst::Or(a, b) => {
                let a = (*a).into_ts();
                let b = (*b).into_ts();
                quote! { rsdsl::Expr::Or(Box::new(#a), Box::new(#b)) }
            }
            ExprAst::Implies(a, b) => {
                let a = (*a).into_ts();
                let b = (*b).into_ts();
                quote! { rsdsl::Expr::Implies(Box::new(#a), Box::new(#b)) }
            }
            ExprAst::Iff(a, b) => {
                let a = (*a).into_ts();
                let b = (*b).into_ts();
                quote! { rsdsl::Expr::Iff(Box::new(#a), Box::new(#b)) }
            }
            ExprAst::Add(a, b) => {
                let a = (*a).into_ts();
                let b = (*b).into_ts();
                quote! { rsdsl::Expr::Add(Box::new(#a), Box::new(#b)) }
            }
            ExprAst::Sub(a, b) => {
                let a = (*a).into_ts();
                let b = (*b).into_ts();
                quote! { rsdsl::Expr::Sub(Box::new(#a), Box::new(#b)) }
            }
            ExprAst::Mul(a, b) => {
                let a = (*a).into_ts();
                let b = (*b).into_ts();
                quote! { rsdsl::Expr::Mul(Box::new(#a), Box::new(#b)) }
            }
            ExprAst::Eq(a, b) => {
                let a = (*a).into_ts();
                let b = (*b).into_ts();
                quote! { rsdsl::Expr::Eq(Box::new(#a), Box::new(#b)) }
            }
            ExprAst::Le(a, b) => {
                let a = (*a).into_ts();
                let b = (*b).into_ts();
                quote! { rsdsl::Expr::Le(Box::new(#a), Box::new(#b)) }
            }
            ExprAst::Ge(a, b) => {
                let a = (*a).into_ts();
                let b = (*b).into_ts();
                quote! { rsdsl::Expr::Ge(Box::new(#a), Box::new(#b)) }
            }
            ExprAst::Lt(a, b) => {
                let a = (*a).into_ts();
                let b = (*b).into_ts();
                quote! { rsdsl::Expr::Lt(Box::new(#a), Box::new(#b)) }
            }
            ExprAst::Gt(a, b) => {
                let a = (*a).into_ts();
                let b = (*b).into_ts();
                quote! { rsdsl::Expr::Gt(Box::new(#a), Box::new(#b)) }
            }
            ExprAst::OrList(xs) => {
                let xs = xs.into_iter().map(|x| x.into_ts()).collect::<Vec<_>>();
                quote! { rsdsl::Expr::OrList(vec![#(#xs),*]) }
            }
            ExprAst::Call { name, args } => {
                let args = args.into_iter().map(|x| x.into_ts()).collect::<Vec<_>>();
                quote! { rsdsl::Expr::Call{ name: #name.to_string(), args: vec![#(#args),*] } }
            }
            ExprAst::Sum { binders, body } => {
                let b = binders.into_iter().map(|x| x.into_ts()).collect::<Vec<_>>();
                let body = (*body).into_ts();
                quote! { rsdsl::Expr::Sum{ binders: vec![#(#b),*], body: Box::new(#body) } }
            }
            ExprAst::Paren(x) => {
                let x = (*x).into_ts();
                quote! { rsdsl::Expr::Paren(Box::new(#x)) }
            }
        }
    }
}

impl ObjectiveAst {
    fn into_ts(self) -> proc_macro2::TokenStream {
        let sense = match self.sense.as_str() {
            "minimize" => quote! { rsdsl::ObjSense::Minimize },
            "maximize" => quote! { rsdsl::ObjSense::Maximize },
            _ => quote! { rsdsl::ObjSense::Minimize },
        };
        let expr = self.expr.into_ts();
        quote! { rsdsl::Objective{ sense: #sense, expr: #expr } }
    }
}

struct Parser {
    toks: Vec<TokenTree>,
    i: usize,
}
impl Parser {
    fn new(toks: Vec<TokenTree>) -> Self {
        Self { toks, i: 0 }
    }
    fn eof(&self) -> bool {
        self.i >= self.toks.len()
    }
    fn peek(&self) -> Option<&TokenTree> {
        self.toks.get(self.i)
    }
    fn next(&mut self) -> Option<TokenTree> {
        if self.eof() {
            None
        } else {
            let t = self.toks[self.i].clone();
            self.i += 1;
            Some(t)
        }
    }
    fn expect_ident(&mut self, s: &str) -> PResult<()> {
        match self.next() {
            Some(TokenTree::Ident(id)) if id.to_string() == s => Ok(()),
            other => Err(ParseError::new(format!(
                "expected ident `{}`, got {:?}",
                s, other
            ))),
        }
    }
    fn take_ident(&mut self) -> PResult<String> {
        match self.next() {
            Some(TokenTree::Ident(id)) => Ok(id.to_string()),
            other => Err(ParseError::new(format!("expected ident, got {:?}", other))),
        }
    }
    fn take_lit(&mut self) -> PResult<f64> {
        match self.next() {
            Some(TokenTree::Literal(l)) => {
                let s = l.to_string();
                let s = s.trim_matches('"').to_string();
                s.parse::<f64>()
                    .map_err(|_| ParseError::new(format!("bad literal {}", s)))
            }
            other => Err(ParseError::new(format!(
                "expected literal, got {:?}",
                other
            ))),
        }
    }
    fn expect_punct(&mut self, ch: char) -> PResult<()> {
        match self.next() {
            Some(TokenTree::Punct(p)) if p.as_char() == ch => Ok(()),
            other => Err(ParseError::new(format!(
                "expected punct `{}`, got {:?}",
                ch, other
            ))),
        }
    }
    fn try_punct(&mut self, ch: char) -> bool {
        match self.peek() {
            Some(TokenTree::Punct(p)) if p.as_char() == ch => {
                self.i += 1;
                true
            }
            _ => false,
        }
    }
    fn try_ident(&mut self, s: &str) -> bool {
        match self.peek() {
            Some(TokenTree::Ident(id)) if id.to_string() == s => {
                self.i += 1;
                true
            }
            _ => false,
        }
    }
    fn take_group(&mut self, delim: Delimiter) -> PResult<Vec<TokenTree>> {
        match self.next() {
            Some(TokenTree::Group(g)) if g.delimiter() == delim => {
                Ok(g.stream().into_iter().collect())
            }
            other => Err(ParseError::new(format!(
                "expected group {:?}, got {:?}",
                delim, other
            ))),
        }
    }

    fn parse_model(&mut self) -> PResult<ModelSpecAst> {
        self.expect_ident("model")?;
        let name = self.take_ident()?;
        let body = self.take_group(Delimiter::Brace)?;
        let mut p = Parser::new(body);
        let mut decls = vec![];
        let mut rules = vec![];
        let mut objective: Option<ObjectiveAst> = None;

        while !p.eof() {
            if p.try_ident("index") {
                let idx_name = p.take_ident()?;
                p.expect_punct('=')?;
                // (x,z) in Grid ;  we ignore binder tuple and domain after in
                // read group paren
                let _ = p.take_group(Delimiter::Parenthesis)?;
                p.expect_ident("in")?;
                let dom = p.take_ident()?;
                p.expect_punct(';')?;
                decls.push(DeclAst::Index {
                    name: idx_name,
                    domain: dom,
                });
                continue;
            }
            if p.try_ident("enum") {
                let ename = p.take_ident()?;
                let g = p.take_group(Delimiter::Brace)?;
                let mut q = Parser::new(g);
                let mut vars = vec![];
                while !q.eof() {
                    if let Some(TokenTree::Ident(id)) = q.peek() {
                        vars.push(id.to_string());
                        q.i += 1;
                        q.try_punct(',');
                    } else {
                        break;
                    }
                }
                decls.push(DeclAst::Enum {
                    name: ename,
                    variants: vars,
                });
                continue;
            }
            if p.try_ident("scenario") {
                let sname = p.take_ident()?;
                p.expect_ident("in")?;
                let g = p.take_group(Delimiter::Brace)?;
                let mut q = Parser::new(g);
                let mut vals = vec![];
                while !q.eof() {
                    if let Some(TokenTree::Literal(_)) = q.peek() {
                        let v = q.take_lit()? as i32;
                        vals.push(v);
                        q.try_punct(',');
                    } else {
                        break;
                    }
                }
                p.expect_punct(';')?;
                decls.push(DeclAst::Scenario {
                    name: sname,
                    values: vals,
                });
                continue;
            }
            if p.try_ident("pin") {
                let pname = p.take_ident()?;
                p.expect_punct(':')?;
                let ty = p.take_ident()?;
                p.expect_punct(';')?;
                decls.push(DeclAst::Pin { name: pname, ty });
                continue;
            }
            if p.try_ident("fn") {
                let fname = p.take_ident()?;
                let args_g = p.take_group(Delimiter::Parenthesis)?;
                let mut q = Parser::new(args_g);
                let mut args = vec![];
                while !q.eof() {
                    let a = q.take_ident()?;
                    q.expect_punct(':')?;
                    let t = q.take_ident()?;
                    args.push((a, t));
                    if !q.try_punct(',') {
                        break;
                    }
                }
                p.expect_punct('-')?;
                p.expect_punct('>')?;
                let ret = p.take_ident()?;
                p.expect_punct(';')?;
                decls.push(DeclAst::Fn {
                    name: fname,
                    args,
                    ret,
                });
                continue;
            }
            // var decl
            if p.try_ident("place")
                || p.try_ident("state")
                || p.try_ident("shape")
                || p.try_ident("sources")
            {
                let kind_tok = match &p.toks[p.i - 1] {
                    TokenTree::Ident(id) => id.to_string(),
                    _ => "place".into(),
                };
                let vname = p.take_ident()?;
                // [indices]
                let idx_g = p.take_group(Delimiter::Bracket)?;
                let mut q = Parser::new(idx_g);
                let mut idx = vec![];
                while !q.eof() {
                    idx.push(q.take_ident()?);
                    if !q.try_punct(',') {
                        break;
                    }
                }
                p.expect_punct(':')?;
                let ty = p.take_ident()?;
                p.expect_punct(';')?;
                decls.push(DeclAst::Var {
                    kind: kind_tok,
                    name: vname,
                    indices: idx,
                    ty,
                });
                continue;
            }
            if p.try_ident("rule") {
                let rname = p.take_ident()?;
                let body_g = p.take_group(Delimiter::Brace)?;
                let mut q = Parser::new(body_g);
                let body = q.parse_stmt_block()?;
                rules.push(RuleAst { name: rname, body });
                continue;
            }
            if p.try_ident("objective") {
                let sense = p.take_ident()?; // minimize/maximize
                let g = p.take_group(Delimiter::Brace)?;
                let mut q = Parser::new(g);
                let expr = q.parse_expr(0)?;
                objective = Some(ObjectiveAst { sense, expr });
                continue;
            }
            return Err(ParseError::new(format!(
                "unexpected token at top-level: {:?}",
                p.peek()
            )));
        }

        Ok(ModelSpecAst {
            name,
            decls,
            rules,
            objective,
        })
    }

    fn parse_stmt_block(&mut self) -> PResult<Vec<StmtAst>> {
        let mut out = vec![];
        while !self.eof() {
            // ignore stray semicolons
            if self.try_punct(';') {
                continue;
            }
            out.push(self.parse_stmt()?);
        }
        Ok(out)
    }

    fn parse_stmt(&mut self) -> PResult<StmtAst> {
        if self.try_ident("forall") {
            let g = self.take_group(Delimiter::Parenthesis)?;
            let mut q = Parser::new(g);
            let binders = q.parse_binders()?;
            let body_g = self.take_group(Delimiter::Brace)?;
            let mut b = Parser::new(body_g);
            let body = b.parse_stmt_block()?;
            return Ok(StmtAst::ForAll { binders, body });
        }
        if self.try_ident("feature") {
            let name = self.take_ident()?;
            if self.try_punct(':') {
                let st = self.parse_stmt()?;
                return Ok(StmtAst::Feature {
                    name,
                    body: vec![st],
                });
            }
            let body_g = self.take_group(Delimiter::Brace)?;
            let mut b = Parser::new(body_g);
            let body = b.parse_stmt_block()?;
            return Ok(StmtAst::Feature { name, body });
        }
        if self.try_ident("let") {
            let name = self.take_ident()?;
            self.expect_punct('=')?;
            let expr = self.parse_expr(0)?;
            self.expect_punct(';')?;
            return Ok(StmtAst::Let { name, expr });
        }
        if self.try_ident("require") {
            let e = self.parse_expr(0)?;
            self.expect_punct(';')?;
            return Ok(StmtAst::Require(e));
        }
        if self.try_ident("def") {
            let lhs = self.parse_varref()?;
            // <-> token: < - >
            self.expect_punct('<')?;
            self.expect_punct('-')?;
            self.expect_punct('>')?;
            let rhs = self.parse_expr(0)?;
            self.expect_punct(';')?;
            return Ok(StmtAst::Def { lhs, rhs });
        }
        if self.try_ident("add") || self.try_ident("exclude") {
            let exclude = match &self.toks[self.i - 1] {
                TokenTree::Ident(id) => id.to_string() == "exclude",
                _ => false,
            };
            let target = self.parse_varref()?;
            // +=
            self.expect_punct('+')?;
            self.expect_punct('=')?;
            let value = self.parse_expr(0)?;
            let cond = if self.try_ident("where") {
                let c = self.parse_expr(0)?;
                Some(c)
            } else {
                None
            };
            self.expect_punct(';')?;
            return Ok(StmtAst::Add {
                exclude,
                target,
                value,
                cond,
            });
        }
        if self.try_ident("force") {
            // IMPORTANT: `parse_expr()` would greedily consume `==` into Expr::Eq,
            // so we must split the token stream at top-level `==` ourselves.
            let toks = self.collect_until_semi()?;
            let (lhs_toks, rhs_toks) = self.split_top_level_eqeq(&toks)?;
            let mut lp = Parser::new(lhs_toks);
            let lhs = lp.parse_expr(0)?;
            if !lp.eof() {
                return Err(ParseError::new("junk after LHS of `force ... == ...;`"));
            }
            let mut rp = Parser::new(rhs_toks);
            let rhs = rp.parse_expr(0)?;
            if !rp.eof() {
                return Err(ParseError::new("junk after RHS of `force ... == ...;`"));
            }
            return Ok(StmtAst::ForceEq { lhs, rhs });
        }

        Err(ParseError::new(format!(
            "unexpected statement token: {:?}",
            self.peek()
        )))
    }

    fn collect_until_semi(&mut self) -> PResult<Vec<TokenTree>> {
        let mut out = Vec::new();
        loop {
            let t = self.next().ok_or_else(|| ParseError::new("expected `;`"))?;
            match &t {
                TokenTree::Punct(p) if p.as_char() == ';' => break,
                _ => out.push(t),
            }
        }
        Ok(out)
    }

    fn split_top_level_eqeq(
        &self,
        toks: &[TokenTree],
    ) -> PResult<(Vec<TokenTree>, Vec<TokenTree>)> {
        for i in 0..toks.len().saturating_sub(1) {
            if matches!(&toks[i], TokenTree::Punct(p) if p.as_char() == '=')
                && matches!(&toks[i + 1], TokenTree::Punct(p) if p.as_char() == '=')
            {
                return Ok((toks[..i].to_vec(), toks[i + 2..].to_vec()));
            }
        }
        Err(ParseError::new("expected `==` in `force ... == ...;`"))
    }

    fn parse_binders(&mut self) -> PResult<Vec<BinderAst>> {
        // support: c in Cell , (c,d) in Cell * Dir , separated by ','
        let mut out = vec![];
        loop {
            let vars = if self.try_punct('(') {
                // actually '(' is punct in paren group not possible; tuples are represented as Group(Parenthesis) inside binder group
                return Err(ParseError::new("unexpected '(' in binder list"));
            } else if let Some(TokenTree::Group(g)) = self.peek() {
                if g.delimiter() == Delimiter::Parenthesis {
                    let inner = self.take_group(Delimiter::Parenthesis)?;
                    let mut q = Parser::new(inner);
                    let mut vs = vec![];
                    while !q.eof() {
                        vs.push(q.take_ident()?);
                        if !q.try_punct(',') {
                            break;
                        }
                    }
                    vs
                } else {
                    vec![self.take_ident()?]
                }
            } else {
                vec![self.take_ident()?]
            };
            self.expect_ident("in")?;
            let mut domains = vec![];
            domains.push(self.take_ident()?);
            while self.try_punct('*') {
                domains.push(self.take_ident()?);
            }
            out.push(BinderAst { vars, domains });
            if !self.try_punct(',') {
                break;
            }
        }
        Ok(out)
    }

    fn parse_varref(&mut self) -> PResult<VarRefAst> {
        let name = self.take_ident()?;
        let idx_g = self.take_group(Delimiter::Bracket)?;
        let mut q = Parser::new(idx_g);
        let mut idx = vec![];
        while !q.eof() {
            idx.push(q.parse_expr(0)?);
            if !q.try_punct(',') {
                break;
            }
        }
        Ok(VarRefAst { name, indices: idx })
    }

    // Pratt parser
    fn parse_expr(&mut self, min_bp: u8) -> PResult<ExprAst> {
        let mut lhs = self.parse_prefix()?;
        loop {
            let op = match self.peek() {
                Some(TokenTree::Ident(id)) if id.to_string() == "and" => "and",
                Some(TokenTree::Ident(id)) if id.to_string() == "or" => "or",
                Some(TokenTree::Punct(p)) if p.as_char() == '+' => "+",
                Some(TokenTree::Punct(p)) if p.as_char() == '-' => "-",
                Some(TokenTree::Punct(p)) if p.as_char() == '*' => "*",
                Some(TokenTree::Punct(p)) if p.as_char() == '<' => "<",
                Some(TokenTree::Punct(p)) if p.as_char() == '>' => ">",
                Some(TokenTree::Punct(p)) if p.as_char() == '=' => "=",
                _ => break,
            };
            // handle -> and <-> and comparisons
            if op == "-" {
                // could be -> if next is '>'
                if let Some(TokenTree::Punct(p1)) = self.peek() {
                    if p1.as_char() == '-' {
                        // check next two
                    }
                }
            }

            // lookahead sequences for -> and <-> and <= >= ==
            if self.match_seq(&['-', '>']) {
                let (l_bp, r_bp) = (1, 2);
                if l_bp < min_bp {
                    self.unconsume(2);
                    break;
                }
                let rhs = self.parse_expr(r_bp)?;
                lhs = ExprAst::Implies(Box::new(lhs), Box::new(rhs));
                continue;
            }
            if self.match_seq(&['<', '-', '>']) {
                let (l_bp, r_bp) = (0, 1);
                if l_bp < min_bp {
                    self.unconsume(3);
                    break;
                }
                let rhs = self.parse_expr(r_bp)?;
                lhs = ExprAst::Iff(Box::new(lhs), Box::new(rhs));
                continue;
            }
            if self.match_seq(&['<', '=']) {
                let (l_bp, r_bp) = (3, 4);
                if l_bp < min_bp {
                    self.unconsume(2);
                    break;
                }
                let rhs = self.parse_expr(r_bp)?;
                lhs = ExprAst::Le(Box::new(lhs), Box::new(rhs));
                continue;
            }
            if self.match_seq(&['>', '=']) {
                let (l_bp, r_bp) = (3, 4);
                if l_bp < min_bp {
                    self.unconsume(2);
                    break;
                }
                let rhs = self.parse_expr(r_bp)?;
                lhs = ExprAst::Ge(Box::new(lhs), Box::new(rhs));
                continue;
            }
            if self.match_seq(&['=', '=']) {
                let (l_bp, r_bp) = (3, 4);
                if l_bp < min_bp {
                    self.unconsume(2);
                    break;
                }
                let rhs = self.parse_expr(r_bp)?;
                lhs = ExprAst::Eq(Box::new(lhs), Box::new(rhs));
                continue;
            }

            // single-char ops
            let (l_bp, r_bp, kind) = match op {
                "or" => (2, 3, "or"),
                "and" => (4, 5, "and"),
                "+" => (6, 7, "+"),
                "-" => (6, 7, "-"),
                "*" => (8, 9, "*"),
                "<" => (3, 4, "<"),
                ">" => (3, 4, ">"),
                "=" => break, // handled by ==
                _ => break,
            };
            if l_bp < min_bp {
                break;
            }
            // consume op token
            self.next();
            let rhs = self.parse_expr(r_bp)?;
            lhs = match kind {
                "or" => ExprAst::Or(Box::new(lhs), Box::new(rhs)),
                "and" => ExprAst::And(Box::new(lhs), Box::new(rhs)),
                "+" => ExprAst::Add(Box::new(lhs), Box::new(rhs)),
                "-" => ExprAst::Sub(Box::new(lhs), Box::new(rhs)),
                "*" => ExprAst::Mul(Box::new(lhs), Box::new(rhs)),
                "<" => ExprAst::Lt(Box::new(lhs), Box::new(rhs)),
                ">" => ExprAst::Gt(Box::new(lhs), Box::new(rhs)),
                _ => lhs,
            };
        }
        Ok(lhs)
    }

    fn parse_prefix(&mut self) -> PResult<ExprAst> {
        // unary
        if self.try_punct('!') {
            let x = self.parse_expr(10)?;
            return Ok(ExprAst::Not(Box::new(x)));
        }
        if self.try_punct('-') {
            // unary minus
            let x = self.parse_expr(10)?;
            return Ok(ExprAst::Neg(Box::new(x)));
        }

        // OR{...}
        if self.try_ident("OR") {
            if let Some(TokenTree::Group(g)) = self.peek() {
                if g.delimiter() == Delimiter::Brace {
                    let g = self.take_group(Delimiter::Brace)?;
                    let mut q = Parser::new(g);
                    let mut xs = vec![];
                    while !q.eof() {
                        xs.push(q.parse_expr(0)?);
                        if !q.try_punct(',') {
                            break;
                        }
                    }
                    return Ok(ExprAst::OrList(xs));
                }
            }
            // OR(...)
            let args_g = self.take_group(Delimiter::Parenthesis)?;
            let mut q = Parser::new(args_g);
            let mut args = vec![];
            while !q.eof() {
                args.push(q.parse_call_arg()?);
                if !q.try_punct(',') {
                    break;
                }
            }
            return Ok(ExprAst::Call {
                name: "OR".into(),
                args,
            });
        }

        // sum(...)
        if self.try_ident("sum") {
            let g = self.take_group(Delimiter::Parenthesis)?;
            let mut q = Parser::new(g);
            let binders = q.parse_binders()?;
            let body = self.parse_expr(9)?; // binds tightly
            return Ok(ExprAst::Sum {
                binders,
                body: Box::new(body),
            });
        }

        match self.next() {
            Some(TokenTree::Literal(l)) => {
                let s = l.to_string().trim_matches('"').to_string();
                let v = s
                    .parse::<f64>()
                    .map_err(|_| ParseError::new(format!("bad number {}", s)))?;
                Ok(ExprAst::Lit(v))
            }
            Some(TokenTree::Ident(id)) => {
                let name = id.to_string();
                // call or varref or sym
                match self.peek() {
                    Some(TokenTree::Group(g)) if g.delimiter() == Delimiter::Parenthesis => {
                        let args_g = self.take_group(Delimiter::Parenthesis)?;
                        let mut q = Parser::new(args_g);
                        let mut args = vec![];
                        while !q.eof() {
                            args.push(q.parse_call_arg()?);
                            if !q.try_punct(',') {
                                break;
                            }
                        }
                        Ok(ExprAst::Call { name, args })
                    }
                    Some(TokenTree::Group(g)) if g.delimiter() == Delimiter::Bracket => {
                        let idx_g = self.take_group(Delimiter::Bracket)?;
                        let mut q = Parser::new(idx_g);
                        let mut idx = vec![];
                        while !q.eof() {
                            idx.push(q.parse_expr(0)?);
                            if !q.try_punct(',') {
                                break;
                            }
                        }
                        Ok(ExprAst::Var(VarRefAst { name, indices: idx }))
                    }
                    _ => Ok(ExprAst::Sym(name)),
                }
            }
            Some(TokenTree::Group(g)) if g.delimiter() == Delimiter::Parenthesis => {
                let inner = g.stream().into_iter().collect::<Vec<_>>();
                let mut q = Parser::new(inner);
                let e = q.parse_expr(0)?;
                Ok(ExprAst::Paren(Box::new(e)))
            }
            other => Err(ParseError::new(format!(
                "unexpected token in expr: {:?}",
                other
            ))),
        }
    }

    fn parse_call_arg(&mut self) -> PResult<ExprAst> {
        // named arg: ident = expr
        if let Some(TokenTree::Ident(id)) = self.peek() {
            // lookahead '='
            if let Some(TokenTree::Punct(p)) = self.toks.get(self.i + 1) {
                if p.as_char() == '=' {
                    let name = id.to_string();
                    self.i += 2; // consume ident and '='
                    let v = self.parse_expr(0)?;
                    return Ok(ExprAst::NamedArg {
                        name,
                        value: Box::new(v),
                    });
                }
            }
        }
        self.parse_expr(0)
    }

    fn match_seq(&mut self, seq: &[char]) -> bool {
        if self.i + seq.len() > self.toks.len() {
            return false;
        }
        for (k, ch) in seq.iter().enumerate() {
            match &self.toks[self.i + k] {
                TokenTree::Punct(p) if p.as_char() == *ch => {}
                _ => return false,
            }
        }
        self.i += seq.len();
        true
    }
    fn unconsume(&mut self, n: usize) {
        self.i -= n;
    }
}
