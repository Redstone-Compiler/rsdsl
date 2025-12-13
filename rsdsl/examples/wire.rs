use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::Write;
use std::process::{Command, Stdio};
use std::sync::Arc;

use rsdsl::{codegen_scip_lp, Cell, ConcreteVar, Instance};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 3x1 line: x=0..2, z=0
    let cells = (0..=2).map(|x| Cell { x, z: 0 }).collect::<Vec<_>>();
    let scenarios = vec![0, 1];

    // Weights: you can tune these.
    let mut params = HashMap::new();
    params.insert("wS".into(), 1.0);
    params.insert("wD".into(), 1.0);
    params.insert("wT".into(), 5.0);
    params.insert("wR".into(), 5.0);

    let features = HashSet::new();

    // Observe(pin, scenario) maps to a concrete binary variable name.
    let observe = Arc::new(|pin: &str, s: i32| ConcreteVar::new(format!("OBS__{}__s{}", pin, s)));

    let inst = Instance::new(cells.clone(), scenarios, params, features, observe);

    // Full model spec (DSL)
    let spec = rsdsl::rsdsl! {
        // =======================================
    // Redstone2D_Line_Min_Rep15 (current DSL-compatible)
    // - Cell = (x,z) in Grid  (2D only; no scalar index like Step)
    // - 15-limit reachability is unrolled as Reach0..Reach15
    // - Repeater is a single fixed-direction device: at x7_z0, drives x8_z0 (reset BaseSrc)
    // - Torch inverter: TO[s,c] <-> (T[c] and !BP[s,c])
    // =======================================
    model Redstone2D_Line_Min_Rep15 {
        index Cell = (x,z) in Grid;
        scenario Sc in {0,1}; // s=0: IN=0, s=1: IN=1

        // ----- placement -----
        place S[Cell] : Solid;
        place D[Cell] : Dust;
        place T[Cell] : Torch;
        place R[Cell] : Repeater; // 방향/백/프론트는 DSL에서 못 다루니 고정 좌표로 의미 부여

        // ----- state -----
        state BP[Sc,Cell] : Bool;
        state DP[Sc,Cell] : Bool;
        state TO[Sc,Cell] : Bool;

        // “강한 소스”를 SrcSet/add 없이 Bool로 직접 둠
        state BaseSrc[Sc,Cell]  : Bool;
        state BlockSrc[Sc,Cell] : Bool;

        // Repeater IO (고정 위치에서만 의미있게 사용)
        state RepIn[Sc,Cell] : Bool;
        state RO[Sc,Cell]    : Bool;

        // Reach0..Reach15 (Step 인덱스가 불가능하므로 펼침)
        state Reach0[Sc,Cell]  : Bool;
        state Reach1[Sc,Cell]  : Bool;
        state Reach2[Sc,Cell]  : Bool;
        state Reach3[Sc,Cell]  : Bool;
        state Reach4[Sc,Cell]  : Bool;
        state Reach5[Sc,Cell]  : Bool;
        state Reach6[Sc,Cell]  : Bool;
        state Reach7[Sc,Cell]  : Bool;
        state Reach8[Sc,Cell]  : Bool;
        state Reach9[Sc,Cell]  : Bool;
        state Reach10[Sc,Cell] : Bool;
        state Reach11[Sc,Cell] : Bool;
        state Reach12[Sc,Cell] : Bool;
        state Reach13[Sc,Cell] : Bool;
        state Reach14[Sc,Cell] : Bool;
        state Reach15[Sc,Cell] : Bool;

        // ---------------------------------------
        // Placement constraints (minimal)
        // ---------------------------------------
        rule R_PLACEMENT {
          forall(c in Cell) {
            // 근사: dust/torch/repeater는 solid 필요
            require D[c] -> S[c];
            require T[c] -> S[c];
            require R[c] -> S[c];

            // 한 칸에 dust/torch/repeater 중복 금지(최소 패킹)
            require D[c] + T[c] + R[c] <= 1;
          }
        }

        // ---------------------------------------
        // Domain link (state -> placement)
        // ---------------------------------------
        rule R_DOMAIN {
          forall(s in Sc) {
            forall(c in Cell) {
              require BP[s,c] -> S[c];
              require DP[s,c] -> D[c];
              require TO[s,c] -> T[c];
              require RO[s,c] -> R[c];
            }
          }
        }

        // ---------------------------------------
        // Torch inverter (NOT core)
        // ---------------------------------------
        rule R_TORCH {
          forall(s in Sc) {
            forall(c in Cell) {
              def TO[s,c] <-> (T[c] and !BP[s,c]);
            }
          }
        }

        // ---------------------------------------
        // Block power = OR of BlockSrc (SrcSet 대신 Bool)
        // ---------------------------------------
        rule R_BLOCK_POWER {
          forall(s in Sc) {
            forall(c in Cell) {
              def BP[s,c] <-> BlockSrc[s,c];
            }
          }
        }

        // ---------------------------------------
        // Repeater semantics (fixed geometry)
        // - Interpret: R[x7_z0] is a repeater whose back is x6_z0 and front is x8_z0
        // ---------------------------------------
        rule R_REPEATER_FIXED {
          forall(s in Sc) {
            // input: behind cell signal (dust or block)
            def RepIn[s, x7_z0] <-> (DP[s, x6_z0] or BP[s, x6_z0]);

            // output ON if placed and input ON
            def RO[s, x7_z0] <-> (R[x7_z0] and RepIn[s, x7_z0]);

            // output creates strong source at front cell x8_z0 (distance reset)
            // BaseSrc is Bool, so we do "RO -> BaseSrc" and "BaseSrc -> (IN or RO or TorchOut...)" style constraints.
            require RO[s, x7_z0] -> BaseSrc[s, x8_z0];

            // (optional) repeater output also powers the front block (minimal)
            require RO[s, x7_z0] -> BlockSrc[s, x8_z0];
          }
        }

        // ---------------------------------------
        // Input injection (IN pin)
        // - Use BaseSrc at x0_z0 as the starting strong source
        // ---------------------------------------
        rule R_INPUT {
          // s=0: IN=0, s=1: IN=1
          force BaseSrc[0, x0_z0] == 0;
          force BaseSrc[1, x0_z0] == 1;

          // If you want the input to also power block at x0:
          force BlockSrc[0, x0_z0] == 0;
          force BlockSrc[1, x0_z0] == 1;
        }

        // ---------------------------------------
        // Torch output as strong source (fixed geometry; minimal)
        // - If torch at xk_z0 is ON, it injects BaseSrc into x(k+1)_z0 (just as an example)
        // - Because "add BaseSrc += ..." doesn't exist, we use implication constraints
        // ---------------------------------------
        rule R_TORCH_OUTPUT_FIXED {
          forall(s in Sc) {
            // Example: torch at x4 drives x5
            require TO[s, x4_z0] -> BaseSrc[s, x5_z0];
            require TO[s, x4_z0] -> BlockSrc[s, x5_z0];
          }
        }

        // ---------------------------------------
        // Reachability 15-limit (unrolled, fixed line)
        // - Reach0 at cell c is BaseSrc at c
        // - Reach(k+1) at x(i+1) comes from Reach(k) at x(i) if dust exists at destination
        // ---------------------------------------
        rule R_REACH_15_LINE {
          forall(s in Sc) {
            // t=0
            def Reach0[s, x0_z0]  <-> BaseSrc[s, x0_z0];
            def Reach0[s, x1_z0]  <-> BaseSrc[s, x1_z0];
            def Reach0[s, x2_z0]  <-> BaseSrc[s, x2_z0];
            def Reach0[s, x3_z0]  <-> BaseSrc[s, x3_z0];
            def Reach0[s, x4_z0]  <-> BaseSrc[s, x4_z0];
            def Reach0[s, x5_z0]  <-> BaseSrc[s, x5_z0];
            def Reach0[s, x6_z0]  <-> BaseSrc[s, x6_z0];
            def Reach0[s, x7_z0]  <-> BaseSrc[s, x7_z0];
            def Reach0[s, x8_z0]  <-> BaseSrc[s, x8_z0];
            def Reach0[s, x9_z0]  <-> BaseSrc[s, x9_z0];
            def Reach0[s, x10_z0] <-> BaseSrc[s, x10_z0];
            def Reach0[s, x11_z0] <-> BaseSrc[s, x11_z0];
            def Reach0[s, x12_z0] <-> BaseSrc[s, x12_z0];
            def Reach0[s, x13_z0] <-> BaseSrc[s, x13_z0];
            def Reach0[s, x14_z0] <-> BaseSrc[s, x14_z0];
            def Reach0[s, x15_z0] <-> BaseSrc[s, x15_z0];

            // Propagation along the line (x(i) -> x(i+1)), consuming 1 dust step per move.
            // We only define the "forward" edges explicitly. This is a minimal approximation.

            def Reach1[s,  x1_z0] <-> (D[x1_z0]  and Reach0[s,  x0_z0]);
            def Reach2[s,  x2_z0] <-> (D[x2_z0]  and Reach1[s,  x1_z0]);
            def Reach3[s,  x3_z0] <-> (D[x3_z0]  and Reach2[s,  x2_z0]);
            def Reach4[s,  x4_z0] <-> (D[x4_z0]  and Reach3[s,  x3_z0]);
            def Reach5[s,  x5_z0] <-> (D[x5_z0]  and Reach4[s,  x4_z0]);
            def Reach6[s,  x6_z0] <-> (D[x6_z0]  and Reach5[s,  x5_z0]);
            def Reach7[s,  x7_z0] <-> (D[x7_z0]  and Reach6[s,  x6_z0]);
            def Reach8[s,  x8_z0] <-> (D[x8_z0]  and Reach7[s,  x7_z0]);
            def Reach9[s,  x9_z0] <-> (D[x9_z0]  and Reach8[s,  x8_z0]);
            def Reach10[s, x10_z0] <-> (D[x10_z0] and Reach9[s,  x9_z0]);
            def Reach11[s, x11_z0] <-> (D[x11_z0] and Reach10[s, x10_z0]);
            def Reach12[s, x12_z0] <-> (D[x12_z0] and Reach11[s, x11_z0]);
            def Reach13[s, x13_z0] <-> (D[x13_z0] and Reach12[s, x12_z0]);
            def Reach14[s, x14_z0] <-> (D[x14_z0] and Reach13[s, x13_z0]);
            def Reach15[s, x15_z0] <-> (D[x15_z0] and Reach14[s, x14_z0]);

            // DP is ON if any ReachK is true at that cell (K=0..15), and dust exists.
            def DP[s, x0_z0] <-> (D[x0_z0] and (
              Reach0[s,x0_z0] or Reach1[s,x0_z0] or Reach2[s,x0_z0] or Reach3[s,x0_z0] or
              Reach4[s,x0_z0] or Reach5[s,x0_z0] or Reach6[s,x0_z0] or Reach7[s,x0_z0] or
              Reach8[s,x0_z0] or Reach9[s,x0_z0] or Reach10[s,x0_z0] or Reach11[s,x0_z0] or
              Reach12[s,x0_z0] or Reach13[s,x0_z0] or Reach14[s,x0_z0] or Reach15[s,x0_z0]
            ));

            def DP[s, x15_z0] <-> (D[x15_z0] and (
              Reach0[s,x15_z0] or Reach1[s,x15_z0] or Reach2[s,x15_z0] or Reach3[s,x15_z0] or
              Reach4[s,x15_z0] or Reach5[s,x15_z0] or Reach6[s,x15_z0] or Reach7[s,x15_z0] or
              Reach8[s,x15_z0] or Reach9[s,x15_z0] or Reach10[s,x15_z0] or Reach11[s,x15_z0] or
              Reach12[s,x15_z0] or Reach13[s,x15_z0] or Reach14[s,x15_z0] or Reach15[s,x15_z0]
            ));

            // (Optional) If you want DP definition for every cell x0..x15, you can expand similarly.
            // For brevity, only x0 and x15 are explicitly defined above.
          }
        }

        // ---------------------------------------
        // Objective (min parts)
        // - No `param` in current DSL, so weights are constants directly.
        // ---------------------------------------
        objective minimize {
            1 * sum(c in Cell) S[c]
          + 1 * sum(c in Cell) D[c]
          + 5 * sum(c in Cell) T[c]
          + 2 * sum(c in Cell) R[c]
        }
      }
    };

    let lp = codegen_scip_lp(&spec, &inst)?;
    println!("{}", lp);
    fs::write("wire.lp", lp)?;

    println!("wrote wire.lp");

    // ------------------------------------------------------------
    // 4) Run SCIP
    //    - override binary: SCIP_BIN=/path/to/scip
    // ------------------------------------------------------------
    let sol_path = "wire.sol";
    let (obj, assigns) = run_scip_and_parse_solution("wire.lp", sol_path)?;

    println!("=== SCIP solution ===");
    if let Some(v) = obj {
        println!("objective = {v}");
    }
    println!();

    let get = |name: &str| assigns.get(name).copied().unwrap_or(0.0);

    // placements per cell
    for c in cells.iter() {
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

    Ok(())
}

/// Run SCIP solver on an LP file and parse the solution.
/// Returns the objective value (if present) and variable assignments.
fn run_scip_and_parse_solution(
    lp_path: &str,
    sol_path: &str,
) -> Result<(Option<f64>, HashMap<String, f64>), Box<dyn std::error::Error>> {
    let scip_bin = std::env::var("SCIP_BIN").unwrap_or_else(|_| "scip".to_string());

    let mut child = match Command::new(&scip_bin)
        .stdin(Stdio::piped())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => {
            eprintln!("failed to spawn `{scip_bin}`: {e}");
            eprintln!("hint: install SCIP or set SCIP_BIN to the scip executable");
            return Err(Box::new(e));
        }
    };

    let cmds = format!(
        "read {lp_path}\nset write printzeros TRUE\noptimize\nwrite solution {sol_path}\nquit\n"
    );
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
        return Err(Box::new(std::io::Error::new(
            std::io::ErrorKind::Other,
            "SCIP failed",
        )));
    }

    let sol_txt = match std::fs::read_to_string(sol_path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("SCIP ran, but failed to read solution file `{sol_path}`: {e}");
            eprintln!("--- stdout ---\n{}", String::from_utf8_lossy(&out.stdout));
            eprintln!("--- stderr ---\n{}", String::from_utf8_lossy(&out.stderr));
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::Other,
                "SCIP failed to read solution file",
            )));
        }
    };

    Ok(parse_scip_sol(&sol_txt))
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
