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

            fn neigh(Cell, Dir) -> Cell?;
            fn supportForWallTorch(Cell, Dir) -> Cell?;

            // placements
            place S[Cell] : Solid;
            place D[Layer,Cell] : Dust;
            place T_stand[Cell] : Torch;
            place T_wall[Cell,Dir] : Torch;

            // state
            state BP[Sc,Cell] : BlockPower <= S;
            state DP[Sc,Layer,Cell] : DustOn <= D;
            state TO1[Sc,Cell] : TorchOn <= T_stand;
            state TOW0[Sc,Cell,Dir] : TorchOn <= T_wall;

            // shapes
            shape Conn[Layer,Cell,Dir] : Conn;

            rule R_DOMAIN {
                require BP[s,c] -> S[c];
                require DP[s,l,c] -> D[l,c];
                require TO1[s,c] -> T_stand[c];
                require TOW0[s,c,d] -> T_wall[c,d];
            }

            rule R_TORCH_INVERTER {
                def TO1[s,c] <-> T_stand[c] & !BP[s,c];
                let sup = supportForWallTorch(c,d);
                require TOW0[s,c,d] -> !BP[s,sup];
                def TOW0[s,c,d] <-> T_wall[c,d] & !BP[s,sup];
            }

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

    spec.debug_print();
    println!("{}", spec.to_pretty_json());
}
