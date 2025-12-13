use std::collections::HashMap;
use std::io::Write;
use std::process::{Command, Stdio};

use rsdsl::scip_codegen::{codegen_scip_lp, Cell, ConcreteVar, Instance};
use rsdsl_macros::rsdsl;

/// NOTE:
/// - 현재 scip_codegen v0는 DustSrc/Add 같은 "배선/전파"를 아직 ILP로 안 내림.
/// - 그래서 이 예시는 "OUT = torch(IN)"처럼 **직접 등식으로 연결**해서,
///   rsdsl! -> .lp -> scip 실행 -> .sol 파싱 파이프라인만 먼저 검증하는 용도임.
fn main() {
    // ------------------------------------------------------------
    // 1) DSL spec (scip_codegen v0가 내릴 수 있는 subset만 사용)
    //    - require / def / force / forall / objective(sum)
    //    - boolean ops: ! & | -> <-> ==
    // ------------------------------------------------------------
    let spec = rsdsl! {
        model NotGateMini {
            index Cell = (x,z) in Grid;
            scenario Sc in {0,1};
            pin IN  : ObsPoint;
            pin OUT : ObsPoint;

            // placements (structure)
            place S[Cell]   : Solid;
            place D0[Cell]  : Dust;
            place D1[Cell]  : Dust;
            place T[Cell]   : Torch;

            // states (scenario)
            state BP[Sc,Cell]  : BlockPower;
            state DP0[Sc,Cell] : DustOn;
            state DP1[Sc,Cell] : DustOn;
            state TO[Sc,Cell]  : TorchOn;

            rule R_ALL {
                // top things need a block
                forall(c in Cell) {
                    require D1[c] -> S[c];
                    require T[c]  -> S[c];
                }

                // domain linking (state -> placement)
                forall(s in Sc, c in Cell) {
                    require BP[s,c]  -> S[c];
                    require DP0[s,c] -> D0[c];
                    require DP1[s,c] -> D1[c];
                    require TO[s,c]  -> T[c];
                }

                // minimal "strong" mode:
                // BP <-> DP1, torch inverter TO <-> T & !BP
                forall(s in Sc, c in Cell) {
                    def BP[s,c] <-> DP1[s,c];
                    def TO[s,c] <-> (T[c] & !BP[s,c]);
                }

                // wiring simplification (PoC):
                // OUT ground dust is directly driven by torch at IN cell.
                forall(s in Sc) {
                    def DP0[s, x1_z0] <-> TO[s, x0_z0];
                }

                // Truth table via pins:
                force Observe(IN,  s=0) == 0;
                force Observe(IN,  s=1) == 1;
                force Observe(OUT, s=0) == 1;
                force Observe(OUT, s=1) == 0;
            }

            objective minimize {
                wS * sum(c in Cell) S[c]
              + wT * sum(c in Cell) T[c]
              + wD * (sum(c in Cell) D0[c] + sum(c in Cell) D1[c])
            }
        }
    };

    // ------------------------------------------------------------
    // 2) Concrete instance for codegen
    // ------------------------------------------------------------
    let cells = vec![Cell { x: 0, z: 0 }, Cell { x: 1, z: 0 }];

    let mut params = HashMap::new();
    params.insert("wS".to_string(), 0.1);
    params.insert("wT".to_string(), 1.0);
    params.insert("wD".to_string(), 0.2);

    // Observe(IN, s)  -> DP1[s, x0_z0]
    // Observe(OUT, s) -> DP0[s, x1_z0]
    let observe = std::sync::Arc::new(|pin: &str, sc: i32| -> ConcreteVar {
        match pin {
            "IN" => ConcreteVar {
                name: "DP1".into(),
                indices: vec![sc.to_string(), "x0_z0".into()],
            },
            "OUT" => ConcreteVar {
                name: "DP0".into(),
                indices: vec![sc.to_string(), "x1_z0".into()],
            },
            other => panic!("unknown pin {other}"),
        }
    });

    let inst = Instance {
        cells: cells.clone(),
        scenarios: vec![0, 1],
        params,
        observe,
    };

    // ------------------------------------------------------------
    // 3) Generate SCIP LP file
    // ------------------------------------------------------------
    let lp = codegen_scip_lp(&spec, &inst).expect("codegen failed");
    std::fs::write("not_gate.lp", &lp).expect("write not_gate.lp");
    eprintln!("wrote not_gate.lp");

    // ------------------------------------------------------------
    // 4) Run SCIP
    //    - override binary: SCIP_BIN=/path/to/scip
    // ------------------------------------------------------------
    let scip_bin = std::env::var("SCIP_BIN").unwrap_or_else(|_| "scip".to_string());
    let sol_path = "not_gate.sol";

    let mut child = match Command::new(&scip_bin)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => {
            eprintln!("failed to spawn `{scip_bin}`: {e}");
            eprintln!("hint: install SCIP or set SCIP_BIN to the scip executable");
            return;
        }
    };

    let cmds = format!("read not_gate.lp\noptimize\nwrite solution {sol_path}\nquit\n");
    child
        .stdin
        .as_mut()
        .unwrap()
        .write_all(cmds.as_bytes())
        .expect("write scip commands");

    let out = child.wait_with_output().expect("wait scip");
    if !out.status.success() {
        eprintln!("SCIP exited with {}", out.status);
        eprintln!("--- stdout ---\n{}", String::from_utf8_lossy(&out.stdout));
        eprintln!("--- stderr ---\n{}", String::from_utf8_lossy(&out.stderr));
        return;
    }

    // ------------------------------------------------------------
    // 5) Parse SCIP .sol and pretty print key vars
    // ------------------------------------------------------------
    let sol_txt = match std::fs::read_to_string(sol_path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("SCIP ran, but failed to read solution file `{sol_path}`: {e}");
            eprintln!("--- stdout ---\n{}", String::from_utf8_lossy(&out.stdout));
            eprintln!("--- stderr ---\n{}", String::from_utf8_lossy(&out.stderr));
            return;
        }
    };

    let (obj, assigns) = parse_scip_sol(&sol_txt);

    println!("=== SCIP solution ===");
    if let Some(v) = obj {
        println!("objective = {v}");
    }
    println!();

    let get = |name: &str| assigns.get(name).copied().unwrap_or(0.0);

    // placements per cell
    for c in &cells {
        let id = format!("x{}_z{}", c.x, c.z);
        let s = get(&format!("S__{id}"));
        let d0 = get(&format!("D0__{id}"));
        let d1 = get(&format!("D1__{id}"));
        let t = get(&format!("T__{id}"));
        println!("cell {id}:  S={s}  D0={d0}  D1={d1}  T={t}");
    }
    println!();

    // scenario states on IN/OUT
    for sc in [0, 1] {
        let in_id = "x0_z0";
        let out_id = "x1_z0";
        let dp1 = get(&format!("DP1__{sc}__{in_id}"));
        let bp = get(&format!("BP__{sc}__{in_id}"));
        let to = get(&format!("TO__{sc}__{in_id}"));
        let dp0 = get(&format!("DP0__{sc}__{out_id}"));
        println!("s={sc}: IN(DP1)={dp1}  BP={bp}  TO={to}  OUT(DP0)={dp0}");
    }

    // dump non-zero (hide aux by default)
    println!("\nnon-zero vars:");
    let mut nz: Vec<_> = assigns.into_iter().filter(|(_, v)| *v != 0.0).collect();
    nz.sort_by(|a, b| a.0.cmp(&b.0));
    for (k, v) in nz {
        if k.starts_with("__aux_") {
            continue;
        }
        println!("{k} = {v}");
    }
}

/// Parse a SCIP `.sol` file.
/// - ignores comments
/// - parses objective value if present
/// - parses "<var> <value>" assignments
fn parse_scip_sol(sol: &str) -> (Option<f64>, HashMap<String, f64>) {
    let mut obj = None;
    let mut map = HashMap::new();

    for line in sol.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        if let Some(rest) = line.strip_prefix("objective value:") {
            if let Ok(v) = rest.trim().parse::<f64>() {
                obj = Some(v);
            }
            continue;
        }
        if let Some(rest) = line.strip_prefix("objective value =") {
            if let Ok(v) = rest.trim().parse::<f64>() {
                obj = Some(v);
            }
            continue;
        }

        let mut it = line.split_whitespace();
        let Some(name) = it.next() else { continue };
        let Some(val) = it.next() else { continue };
        if let Ok(v) = val.parse::<f64>() {
            map.insert(name.to_string(), v);
        }
    }

    (obj, map)
}
