use std::collections::HashMap;

use rsdsl::scip_codegen::{codegen_scip_lp, Cell, ConcreteVar, Instance};
use rsdsl_macros::rsdsl;

fn main() {
    let spec = rsdsl! {
        model Redstone2L {
            index Cell = (x,z) in Grid;
            enum Layer { GROUND=0, TOP=1 }
            enum Dir   { N,E,S,W }
            scenario Sc in {0,1};
            pin IN  : ObsPoint;
            pin OUT : ObsPoint;

            place T_stand[Cell] : Torch;
            place T_wall[Cell,Dir] : Torch;

            state DP[Sc,Layer,Cell] : DustOn;

            // For this demo, Observe maps pins to DP at fixed locations.
            rule R_TRUTH_TABLE_NOT {
                force Observe(IN, s=0) == 0;
                force Observe(IN, s=1) == 1;
                force Observe(OUT, s=0) == 1;
                force Observe(OUT, s=1) == 0;
            }

            objective minimize {
                wT * (sum(c in Cell) T_stand[c] + sum((c,d) in Cell * Dir) T_wall[c,d])
            }
        }
    };

    // Tiny 2-cell line: (0,0)=IN , (1,0)=OUT
    let cells = vec![Cell { x: 0, z: 0 }, Cell { x: 1, z: 0 }];
    let mut params = HashMap::new();
    params.insert("wT".to_string(), 1.0);

    // Pin observe mapping: both IN/OUT observe TOP dust at their cell.
    // We return a concrete variable reference: DP[s, TOP, cell]
    let observe = std::sync::Arc::new(move |pin: &str, s: i32| -> ConcreteVar {
        let cell = match pin {
            "IN" => "x0_z0".to_string(),
            "OUT" => "x1_z0".to_string(),
            _ => "x0_z0".to_string(),
        };
        ConcreteVar {
            name: "DP".to_string(),
            indices: vec![s.to_string(), "TOP".into(), cell],
        }
    });

    let inst = Instance {
        cells,
        scenarios: vec![0, 1],
        params,
        observe,
    };

    let lp = codegen_scip_lp(&spec, &inst).expect("codegen failed");
    println!("{}", lp);

    // Write to file for SCIP: `scip -c "read not_gate.lp optimize display solution quit"`
    std::fs::write("not_gate.lp", lp).unwrap();
    eprintln!("wrote not_gate.lp");
}
