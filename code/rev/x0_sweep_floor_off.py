"""X0a: rerun of experiments/q6_sweep.py (200 cells x 10 routers x 1000 trials, master seed 61) with the hop floor OFF
(M_FLOOR = 0 in q6_common => BMR calls bmr_route(m_min=0) and evaluate() no longer applies the hop-floor test).
Patch is applied at module top level so it also takes effect in spawned workers. Output: data/results/q6_sweep_floor_off.json
No other change to the sweep: same seeds, same trial streams."""
import sys, os
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "experiments"))
import q6_common
q6_common.M_FLOOR = 0
import q6_sweep
q6_sweep.OUT_NAME = "q6_sweep_floor_off.json"
if __name__ == "__main__":
    q6_sweep.main()
