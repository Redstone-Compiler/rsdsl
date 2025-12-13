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
        model NotGate {
            index Cell = (x,z) in Grid;
            enum Layer { GROUND, TOP }
            enum Dir { N, E, S, W }
            scenario s in {0,1};

            pin IN : ObsPoint;
            pin OUT : ObsPoint;

            // placement vars
            place S[Cell] : Solid;
            place D[Layer,Cell] : Dust;
            place T_stand[Cell] : TorchStand;
            place T_wall[Cell,Dir] : TorchWall;
            place R[Cell,Dir] : Repeater;

            // dust shape vars (optional, but shows the pattern)
            shape AxisNS[Layer,Cell] : Bool;
            shape AxisEW[Layer,Cell] : Bool;
            shape Cross[Layer,Cell] : Bool;
            shape Conn[Layer,Cell,Dir] : Bool;

            // state vars per scenario
            state BP[s,Cell] : Bool;
            state DP[s,Layer,Cell] : Bool;
            state TO_stand[s,Cell] : Bool;
            state TO_wall[s,Cell,Dir] : Bool;
            state RO[s,Cell,Dir] : Bool;

            // "sources sets"
            sources DustSrc[s,Layer,Cell] : Bool;
            sources BlockSrc[s,Cell] : Bool;

            rule placement {
                // Ground occupancy: each cell is either a solid block or one ground-component (dust/torch/repeater)
                forall (c in Cell) {
                    require S[c] + D[GROUND,c] + T_stand[c] + sum((d) in Dir) (T_wall[c,d] + R[c,d]) <= 1;
                }

                // If you place a wall torch, you must have a supporting block behind it.
                forall (c in Cell, d in Dir) {
                    require T_wall[c,d] -> S[supportForWallTorch(c,d)];
                }
            }

            rule dust_shape {
                // Shapes only matter if dust exists.
                forall (l in Layer, c in Cell) {
                    require AxisNS[l,c] <= D[l,c];
                    require AxisEW[l,c] <= D[l,c];
                    require Cross[l,c] <= D[l,c];

                    def Cross[l,c] <-> (AxisNS[l,c] and AxisEW[l,c]);

                    // Connection semantics: N/S controlled by AxisNS, E/W by AxisEW; Cross enables all.
                    def Conn[l,c,N] <-> (Cross[l,c] or AxisNS[l,c]);
                    def Conn[l,c,S] <-> (Cross[l,c] or AxisNS[l,c]);
                    def Conn[l,c,E] <-> (Cross[l,c] or AxisEW[l,c]);
                    def Conn[l,c,W] <-> (Cross[l,c] or AxisEW[l,c]);

                    // Consistency: can't be both NS and EW unless Cross is on.
                    require AxisNS[l,c] + AxisEW[l,c] <= 1 + Cross[l,c];
                }
            }

            rule state_logic {
                // Torch logic: output is 1 iff torch exists and its attached/support block is NOT powered.
                forall (s in s, c in Cell) {
                    def TO_stand[s,c] <-> (T_stand[c] and !BP[s,c]);
                }
                forall (s in s, c in Cell, d in Dir) {
                    def TO_wall[s,c,d] <-> (T_wall[c,d] and !BP[s,supportForWallTorch(c,d)]);
                }

                // Simple repeater: output == placement AND input-dust is powered at "back".
                forall (s in s, c in Cell, d in Dir) {
                    def RO[s,c,d] <-> (R[c,d] and DP[s,GROUND,back(c,d)]);
                }

                // Block sources (very simplified):
                // - A solid block can be powered by adjacent dust that points to it (both layers), adjacent torches, or repeaters.
                forall (s in s, c in Cell, d in Dir) {
                    // dust -> block: neighbor dust connects toward this cell
                    add BlockSrc[s,c] += (DP[s,GROUND,neigh(c,d)] and Conn[GROUND,neigh(c,d),opp(d)]);
                    add BlockSrc[s,c] += (DP[s,TOP,neigh(c,d)] and Conn[TOP,neigh(c,d),opp(d)]);

                    // adjacent torches (stand/wall) can power blocks around them (coarse approx)
                    add BlockSrc[s,c] += TO_stand[s,neigh(c,d)];
                    add BlockSrc[s,c] += TO_wall[s,neigh(c,d),opp(d)];

                    // repeater outputs in neighbor cell facing us (coarse approx)
                    add BlockSrc[s,c] += RO[s,neigh(c,d),opp(d)];
                }

                // Block is powered iff any source is on.
                forall (s in s, c in Cell) {
                    def BP[s,c] <-> OR(BlockSrc[s,c]);
                }

                // Dust sources (very simplified):
                // - Dust can be powered by its block-under being powered, or by adjacent block power through Conn,
                //   or by adjacent torches / repeaters.
                forall (s in s, l in Layer, c in Cell) {
                    // from block under (coarse: same cell)
                    add DustSrc[s,l,c] += BP[s,c];
                }
                forall (s in s, l in Layer, c in Cell, d in Dir) {
                    // adjacent block -> dust
                    add DustSrc[s,l,c] += (BP[s,neigh(c,d)] and Conn[l,c,d]);

                    // adjacent torches and repeaters
                    add DustSrc[s,l,c] += TO_stand[s,neigh(c,d)];
                    add DustSrc[s,l,c] += TO_wall[s,neigh(c,d),opp(d)];
                    add DustSrc[s,l,c] += RO[s,neigh(c,d),opp(d)];

                    // dust chain: neighbor dust can feed if it points to us
                    add DustSrc[s,l,c] += (DP[s,l,neigh(c,d)] and Conn[l,neigh(c,d),opp(d)]);
                }

                // Dust is powered iff it exists and any dust-source is on.
                forall (s in s, l in Layer, c in Cell) {
                    def DP[s,l,c] <-> (D[l,c] and OR(DustSrc[s,l,c]));
                }
            }

            rule truth_table_pins {
                // Choose a fixed 3-cell line:
                // IN  at x0_z0 (block power), torch at x1_z0 (wall torch attached to IN), OUT dust at x2_z0
                //   x0 -- x1 -- x2

                // Force observation values
                force Observe(IN, s=0) == 0;
                force Observe(IN, s=1) == 1;

                force Observe(OUT, s=0) == 1;
                force Observe(OUT, s=1) == 0;

                // Pin the input block power
                force BP[0, x0_z0] == Observe(IN, s=0);
                force BP[1, x0_z0] == Observe(IN, s=1);

                // Force output dust power at OUT cell on ground layer
                force DP[0, GROUND, x2_z0] == Observe(OUT, s=0);
                force DP[1, GROUND, x2_z0] == Observe(OUT, s=1);

                // Also force some placement to make the circuit possible:
                // a solid at IN cell, a wall torch at middle facing East (attached to IN), and dust at OUT.
                force S[x0_z0] == 1;
                force T_wall[x1_z0, E] == 1;
                force D[GROUND, x2_z0] == 1;

                // no extra stuff:
                force D[GROUND, x1_z0] == 0;
                force T_stand[x1_z0] == 0;
            }

            objective minimize {
                wS * sum((c) in Cell) S[c]
              + wD * sum((l,c) in Layer * Cell) D[l,c]
              + wT * (sum((c) in Cell) T_stand[c] + sum((c,d) in Cell * Dir) T_wall[c,d])
              + wR * sum((c,d) in Cell * Dir) R[c,d]
            }
        }
    };

    let lp = codegen_scip_lp(&spec, &inst)?;
    println!("{}", lp);
    fs::write("not_gate.lp", lp)?;

    println!("wrote not_gate.lp");

    // ------------------------------------------------------------
    // 4) Run SCIP
    //    - override binary: SCIP_BIN=/path/to/scip
    // ------------------------------------------------------------
    let sol_path = "not_gate.sol";
    let (obj, assigns) = run_scip_and_parse_solution("not_gate.lp", sol_path)?;

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

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{HashMap, HashSet};
    use std::fs;
    use std::path::PathBuf;
    use std::sync::Arc;

    // #[test]
    // fn run_not_gate_and_validate_solution() -> Result<(), Box<dyn std::error::Error>> {
    //     // 1) DSL 모델 작성
    //     let spec = rsdsl::rsdsl! {
    //         model NotGateMini {
    //             index Cell = (x,z) in Grid;
    //             scenario Sc in {0,1};

    //             place S[Cell] : Solid;  // solid
    //             place D0[Cell] : GroundDust; // ground dust
    //             place D1[Cell] : TopDust; // top dust
    //             place T[Cell] : TorchStand;  // standing torch

    //             state BP[Sc,Cell] : Bool;
    //             state DP0[Sc,Cell] : Bool;
    //             state DP1[Sc,Cell] : Bool;
    //             state TO[Sc,Cell] : Bool;

    //             rule wiring {
    //                 // Simple NOT pattern: top dust powers block, torch inverts, ground dust outputs
    //                 def BP[s,c] <-> (D1[c]);
    //                 def TO[s,c] <-> (T[c] and !BP[s,c]);
    //                 def DP0[s,c] <-> (TO[s,c]);
    //             }

    //             rule truth_table {
    //                 // truth table constraints
    //                 force DP1[0, x0_z0] == 0;
    //                 force DP1[1, x0_z0] == 1;
    //                 force DP0[0, x1_z0] == 1;
    //                 force DP0[1, x1_z0] == 0;
    //             }

    //             objective minimize { S[x0_z0] + T[x0_z0] + D1[x0_z0] + D0[x1_z0] }
    //         }
    //     };

    //     // 2) 인스턴스 (Grid, params)
    //     let inst = Instance {
    //         cells: (0..=2).map(|x| Cell { x, z: 0 }).collect::<Vec<_>>(),
    //         scenarios: vec![0, 1],
    //         params: HashMap::new(),
    //         features: HashSet::new(),
    //         observe: Arc::new(|pin: &str, s: i32| {
    //             ConcreteVar::new(format!("OBS__{}__s{}", pin, s))
    //         }),
    //     };

    //     // 3) LP 생성
    //     let lp = codegen_scip_lp(&spec, &inst)?;
    //     fs::write("not_gate.lp", &lp)?;
    //     println!("LP generated and written to not_gate.lp");

    //     // 4) SCIP 실행 및 솔루션 파싱
    //     let (obj, assigns) = run_scip_and_parse_solution("not_gate.lp", "not_gate.sol")?;

    //     assert!(obj.is_some(), "no objective value found");
    //     let get = |k: &str| assigns.get(k).copied().unwrap_or(0.0);

    //     let dp0_s0 = get("DP0__0__x1_z0");
    //     let dp0_s1 = get("DP0__1__x1_z0");
    //     assert_eq!(dp0_s0.round() as i32, 1, "expected OUT=1 when IN=0");
    //     assert_eq!(dp0_s1.round() as i32, 0, "expected OUT=0 when IN=1");

    //     println!("✅ verified NOT gate truth table");
    //     Ok(())
    // }

    fn tmp_paths(stem: &str) -> (PathBuf, PathBuf) {
        let lp = PathBuf::from(format!("{stem}.lp"));
        let sol = PathBuf::from(format!("{stem}.sol"));
        (lp, sol)
    }

    // fn solve_spec(
    //     stem: &str,
    //     spec: &rsdsl::ModelSpec,
    //     inst: &Instance,
    // ) -> Result<
    //     (
    //         PathBuf,
    //         PathBuf,
    //         Option<f64>,
    //         std::collections::HashMap<String, f64>,
    //     ),
    //     Box<dyn std::error::Error>,
    // > {
    //     let (lp_path, sol_path) = tmp_paths(stem);

    //     let lp_txt = codegen_scip_lp(spec, inst)?;
    //     fs::write(&lp_path, &lp_txt)?;

    //     let (obj, assigns) =
    //         run_scip_and_parse_solution(lp_path.to_str().unwrap(), sol_path.to_str().unwrap())?;

    //     Ok((lp_path, sol_path, obj, assigns))
    // }

    /// 미전개 인덱스(s,c 같은)가 LP에 남아있으면 바로 잡아내는 가드
    fn assert_no_free_indices(lp: &str) {
        // "A__s__c" 같은 형태를 빠르게 잡는 용도 (OBS__IN__s0 같은 건 안 잡힘)
        assert!(
        !lp.contains("__s__"),
        "LP에 미전개 인덱스 '__s__'가 남아있음. forall 전개가 빠진 가능성이 큼.\n--- LP snippet ---\n{}",
        lp.lines().take(80).collect::<Vec<_>>().join("\n"),
    );
    }

    fn mk_inst_line(x_max: i32) -> Instance {
        let cells = (0..=x_max).map(|x| Cell { x, z: 0 }).collect::<Vec<_>>();
        let scenarios = vec![0, 1];

        let params = HashMap::new();
        let features = HashSet::new();
        let observe =
            Arc::new(|pin: &str, s: i32| ConcreteVar::new(format!("OBS__{}__s{}", pin, s)));

        Instance::new(cells, scenarios, params, features, observe)
    }

    fn geti(assigns: &HashMap<String, f64>, name: &str) -> i32 {
        assigns.get(name).copied().unwrap().round() as i32
    }

    #[test]
    fn dsl_not_two_scenarios() -> Result<(), Box<dyn std::error::Error>> {
        // ✅ 핵심: def 를 forall로 전개
        let spec = rsdsl::rsdsl! {
            model Not2Sc {
                index Cell = (x,z) in Grid;
                scenario Sc in {0,1};

                state A[Sc,Cell] : Bool;
                state B[Sc,Cell] : Bool;

                rule logic {
                    forall (s in Sc, c in Cell) {
                        def B[s,c] <-> (!A[s,c]);
                    }
                }

                rule tt {
                    force A[0, x0_z0] == 0;
                    force A[1, x0_z0] == 1;
                }

                objective minimize { 0 }
            }
        };

        let inst = mk_inst_line(0);
        let lp = codegen_scip_lp(&spec, &inst)?;
        assert_no_free_indices(&lp);

        let (lp_path, sol_path) = tmp_paths("dsl_not_two_scenarios");
        fs::write(&lp_path, &lp)?;

        let (_obj, assigns) =
            run_scip_and_parse_solution(lp_path.to_str().unwrap(), sol_path.to_str().unwrap())?;

        println!("assigns: {:?}", assigns);

        assert_eq!(geti(&assigns, "B__0__x0_z0"), 1);
        assert_eq!(geti(&assigns, "B__1__x0_z0"), 0);
        Ok(())
    }

    #[test]
    fn dsl_sources_or_basic() -> Result<(), Box<dyn std::error::Error>> {
        let spec = rsdsl::rsdsl! {
            model SourcesOr {
                index Cell = (x,z) in Grid;
                scenario Sc in {0,1};

                state A[Sc,Cell] : Bool;
                state B[Sc,Cell] : Bool;
                state X[Sc,Cell] : Bool;

                sources Src[Sc,Cell] : Bool;

                rule logic {
                    forall (s in Sc, c in Cell) {
                        add Src[s,c] += A[s,c];
                        add Src[s,c] += B[s,c];
                        def X[s,c] <-> OR(Src[s,c]);
                    }
                }

                rule tt {
                    force A[0, x0_z0] == 0;
                    force B[0, x0_z0] == 1;

                    force A[1, x0_z0] == 0;
                    force B[1, x0_z0] == 0;
                }

                objective minimize { 0 }
            }
        };

        let inst = mk_inst_line(0);
        let lp = codegen_scip_lp(&spec, &inst)?;
        assert_no_free_indices(&lp);

        let (lp_path, sol_path) = tmp_paths("dsl_sources_or_basic");
        fs::write(&lp_path, &lp)?;
        let (_obj, assigns) =
            run_scip_and_parse_solution(lp_path.to_str().unwrap(), sol_path.to_str().unwrap())?;

        assert_eq!(geti(&assigns, "X__0__x0_z0"), 1);
        assert_eq!(geti(&assigns, "X__1__x0_z0"), 0);
        Ok(())
    }

    #[test]
    fn dsl_neigh_oob_is_zero() -> Result<(), Box<dyn std::error::Error>> {
        let spec = rsdsl::rsdsl! {
            model NeighOob {
                index Cell = (x,z) in Grid;
                enum Dir { N }
                scenario Sc in {0,1};

                state A[Sc,Cell] : Bool;
                state X[Sc,Cell] : Bool;

                rule logic {
                    forall (s in Sc, c in Cell) {
                        // 1x1 그리드면 neigh(c,N)은 항상 OOB -> __NONE__ -> 0으로 고정되는 게 기대
                        def X[s,c] <-> (A[s, neigh(c, N)]);
                    }
                }

                rule tt {
                    force A[0, x0_z0] == 1;
                    force A[1, x0_z0] == 1;
                }

                objective minimize { 0 }
            }
        };

        let inst = mk_inst_line(0);
        let lp = codegen_scip_lp(&spec, &inst)?;
        assert_no_free_indices(&lp);

        let (lp_path, sol_path) = tmp_paths("dsl_neigh_oob_is_zero");
        fs::write(&lp_path, &lp)?;
        let (_obj, assigns) =
            run_scip_and_parse_solution(lp_path.to_str().unwrap(), sol_path.to_str().unwrap())?;

        assert_eq!(geti(&assigns, "X__0__x0_z0"), 0);
        assert_eq!(geti(&assigns, "X__1__x0_z0"), 0);
        Ok(())
    }

    #[test]
    fn dsl_feature_block_toggles_constraints() -> Result<(), Box<dyn std::error::Error>> {
        let spec = rsdsl::rsdsl! {
            model FeatToggle {
                index Cell = (x,z) in Grid;
                scenario Sc in {0,1};

                state X[Sc,Cell] : Bool;

                rule base {
                    feature FORCE_X {
                        force X[0, x0_z0] == 1;
                    }
                }

                objective minimize { X[0, x0_z0] }
            }
        };

        // 1) feature 없음 -> X=0 이 최적 (obj=0)
        {
            let mut inst = mk_inst_line(0);
            inst.features = HashSet::new();

            let lp = codegen_scip_lp(&spec, &inst)?;
            let (lp_path, sol_path) = tmp_paths("dsl_feat_off");
            fs::write(&lp_path, &lp)?;
            let (obj, assigns) =
                run_scip_and_parse_solution(lp_path.to_str().unwrap(), sol_path.to_str().unwrap())?;

            assert_eq!(obj.unwrap().round() as i32, 0);
            assert_eq!(geti(&assigns, "X__0__x0_z0"), 0);
        }

        // 2) feature 있음 -> 강제 X=1 (obj=1)
        {
            let mut inst = mk_inst_line(0);
            let mut feats = HashSet::new();
            feats.insert("FORCE_X".to_string());
            inst.features = feats;

            let lp = codegen_scip_lp(&spec, &inst)?;
            let (lp_path, sol_path) = tmp_paths("dsl_feat_on");
            fs::write(&lp_path, &lp)?;
            let (obj, assigns) =
                run_scip_and_parse_solution(lp_path.to_str().unwrap(), sol_path.to_str().unwrap())?;

            assert_eq!(obj.unwrap().round() as i32, 1);
            assert_eq!(geti(&assigns, "X__0__x0_z0"), 1);
        }

        Ok(())
    }

    #[test]
    fn dsl_observe_pin_is_substituted_per_scenario() -> Result<(), Box<dyn std::error::Error>> {
        let spec = rsdsl::rsdsl! {
            model ObsSubst {
                index Cell = (x,z) in Grid;
                scenario Sc in {0,1};
                pin IN : ObsPoint;

                state X[Sc,Cell] : Bool;

                rule logic {
                    forall (s in Sc, c in Cell) {
                        def X[s,c] <-> Observe(IN, s=s);
                    }
                }

                rule tt {
                    force X[0, x0_z0] == 0;
                    force X[1, x0_z0] == 1;
                }

                objective minimize { 0 }
            }
        };

        let inst = mk_inst_line(0);
        let lp = codegen_scip_lp(&spec, &inst)?;
        assert_no_free_indices(&lp);

        let (lp_path, sol_path) = tmp_paths("dsl_observe_pin");
        fs::write(&lp_path, &lp)?;
        let (_obj, assigns) =
            run_scip_and_parse_solution(lp_path.to_str().unwrap(), sol_path.to_str().unwrap())?;

        // Observe(IN,s=0/1)가 실제로 scenario별 다른 concrete var로 치환되었는지 확인
        assert_eq!(geti(&assigns, "OBS__IN__s0"), 0);
        assert_eq!(geti(&assigns, "OBS__IN__s1"), 1);
        Ok(())
    }

    #[test]
    fn dsl_not_gate_minimal_requires_placements() -> Result<(), Box<dyn std::error::Error>> {
        // "force OUT"만 박아서 통과하는 가짜 NOT 방지:
        // IN(DP1) -> BP -> (Torch AND !BP) -> OUT(DP0)
        // OUT=1을 만들려면 T와 D0 배치가 강제되도록 연결
        let spec = rsdsl::rsdsl! {
            model NotGateTiny {
                index Cell = (x,z) in Grid;
                scenario Sc in {0,1};

                place T[Cell] : TorchStand;
                place D0[Cell] : GroundDust;

                state DP1[Sc,Cell] : Bool;  // input pin (state로 단순화)
                state BP[Sc,Cell] : Bool;
                state TO[Sc,Cell] : Bool;
                state DP0[Sc,Cell] : Bool;  // output pin (state로 단순화)

                rule wiring {
                    forall (s in Sc) {
                        def BP[s, x0_z0] <-> DP1[s, x0_z0];
                        def TO[s, x0_z0] <-> (T[x0_z0] and !BP[s, x0_z0]);
                        def DP0[s, x1_z0] <-> (D0[x1_z0] and TO[s, x0_z0]);
                    }
                }

                rule truth_table {
                    force DP1[0, x0_z0] == 0;
                    force DP1[1, x0_z0] == 1;

                    force DP0[0, x1_z0] == 1;
                    force DP0[1, x1_z0] == 0;
                }

                objective minimize { T[x0_z0] + D0[x1_z0] }
            }
        };

        // x=0..1  (IN at x0, OUT at x1)
        let inst = mk_inst_line(1);
        let lp = codegen_scip_lp(&spec, &inst)?;
        assert_no_free_indices(&lp);

        let (lp_path, sol_path) = tmp_paths("dsl_not_gate_minimal");
        fs::write(&lp_path, &lp)?;
        let (_obj, assigns) =
            run_scip_and_parse_solution(lp_path.to_str().unwrap(), sol_path.to_str().unwrap())?;

        // OUT=1을 만들려면 T와 D0가 강제로 1이 되어야 함
        assert_eq!(geti(&assigns, "T__x0_z0"), 1);
        assert_eq!(geti(&assigns, "D0__x1_z0"), 1);

        // 시나리오별 NOT 동작 확인
        assert_eq!(geti(&assigns, "DP0__0__x1_z0"), 1);
        assert_eq!(geti(&assigns, "DP0__1__x1_z0"), 0);

        Ok(())
    }

    #[test]
    fn dsl_wire3_propagates_signal() -> Result<(), Box<dyn std::error::Error>> {
        use std::{
            collections::{HashMap, HashSet},
            fs,
            sync::Arc,
        };

        let spec = rsdsl::rsdsl! {
            model Wire3 {
                index Cell = (x,z) in Grid;
                scenario Sc in {0,1};

                place D0[Cell] : GroundDust;
                state DP0[Sc,Cell] : Bool;

                rule wire {
                    def DP0[s, x1_z0] <-> (D0[x1_z0] and DP0[s, x0_z0]);
                    def DP0[s, x2_z0] <-> (D0[x2_z0] and DP0[s, x1_z0]);
                }

                rule pins {
                    force D0[x0_z0] == 1;
                    force D0[x1_z0] == 1;
                    force D0[x2_z0] == 1;

                    force DP0[0, x0_z0] == 0;
                    force DP0[1, x0_z0] == 1;
                }

                objective minimize { D0[x0_z0] + D0[x1_z0] + D0[x2_z0] }
            }
        };

        let inst = Instance {
            cells: (0..=2).map(|x| Cell { x, z: 0 }).collect(),
            scenarios: vec![0, 1],
            params: HashMap::new(),
            features: HashSet::new(),
            observe: Arc::new(|pin: &str, s: i32| {
                ConcreteVar::new(format!("OBS__{}__s{}", pin, s))
            }),
        };

        let lp = codegen_scip_lp(&spec, &inst)?;
        fs::write("wire3.lp", &lp)?;

        let (_obj, assigns) = run_scip_and_parse_solution("wire3.lp", "wire3.sol")?;

        // SCIP sol에 0값 변수는 생략될 수 있으니 unwrap_or(0.0)로 읽기
        let geti = |k: &str| assigns.get(k).copied().unwrap_or(0.0).round() as i32;

        // OUT = DP0[s, x2]
        assert_eq!(geti("DP0__0__x2_z0"), 0);
        assert_eq!(geti("DP0__1__x2_z0"), 1);

        Ok(())
    }

    #[test]
    fn scip_codegen_and_truth_table() -> Result<(), Box<dyn std::error::Error>> {
        use std::{
            collections::{HashMap, HashSet},
            sync::Arc,
        };

        // Sc=0..3을 (A,B) = (0,0),(0,1),(1,0),(1,1) 로 매핑
        let spec = rsdsl::rsdsl! {
            model AndTT {
                scenario Sc in {0,1,2,3};

                state A[Sc] : Bool;
                state B[Sc] : Bool;
                state C[Sc] : Bool;

                rule logic {
                    def C[s] <-> (A[s] and B[s]);
                }

                rule pins {
                    // (A,B) truth table 강제
                    force A[0] == 0; force B[0] == 0;
                    force A[1] == 0; force B[1] == 1;
                    force A[2] == 1; force B[2] == 0;
                    force A[3] == 1; force B[3] == 1;
                }

                objective minimize { 0 }
            }
        };

        let inst = Instance {
            cells: vec![], // 안 쓰면 빈 배열 가능
            scenarios: vec![0, 1, 2, 3],
            params: HashMap::new(),
            features: HashSet::new(),
            observe: Arc::new(|pin: &str, s: i32| {
                ConcreteVar::new(format!("OBS__{}__s{}", pin, s))
            }),
        };

        let lp = codegen_scip_lp(&spec, &inst)?;
        fs::write("and_tt.lp", &lp)?;

        let (_obj, assigns) = run_scip_and_parse_solution("and_tt.lp", "and_tt.sol")?;

        let geti = |k: &str| assigns.get(k).copied().unwrap_or(0.0).round() as i32;

        // 기대값: C = A&B
        assert_eq!(geti("C__0"), 0);
        assert_eq!(geti("C__1"), 0);
        assert_eq!(geti("C__2"), 0);
        assert_eq!(geti("C__3"), 1);

        Ok(())
    }

    #[test]
    fn scip_codegen_expands_free_scenario_index() -> Result<(), Box<dyn std::error::Error>> {
        use std::{
            collections::{HashMap, HashSet},
            sync::Arc,
        };

        let spec = rsdsl::rsdsl! {
            model Wire3 {
                index Cell = (x,z) in Grid;
                scenario Sc in {0,1};

                place D0[Cell] : GroundDust;
                state DP0[Sc,Cell] : Bool;

                rule wire {
                    def DP0[s, x1_z0] <-> (D0[x1_z0] and DP0[s, x0_z0]);
                    def DP0[s, x2_z0] <-> (D0[x2_z0] and DP0[s, x1_z0]);
                }

                rule pins {
                    force D0[x0_z0] == 1;
                    force D0[x1_z0] == 1;
                    force D0[x2_z0] == 1;

                    force DP0[0, x0_z0] == 0;
                    force DP0[1, x0_z0] == 1;
                }

                objective minimize { 0 }
            }
        };

        let inst = Instance {
            cells: (0..=2).map(|x| Cell { x, z: 0 }).collect(),
            scenarios: vec![0, 1],
            params: HashMap::new(),
            features: HashSet::new(),
            observe: Arc::new(|pin: &str, s: i32| {
                ConcreteVar::new(format!("OBS__{}__s{}", pin, s))
            }),
        };

        let lp = codegen_scip_lp(&spec, &inst)?;

        // ✅ 전개되어야 함
        assert!(lp.contains("DP0__0__x2_z0"));
        assert!(lp.contains("DP0__1__x2_z0"));

        // ❌ 가짜 인덱스 재발 방지
        assert!(!lp.contains("DP0__s__x2_z0"));

        Ok(())
    }
}
