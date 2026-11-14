"""X0c: E3.1 / Table 8 pipeline with the hop floor OFF (M_FLOOR=0 in q6_common and in the e31 module). Output e31_out_floor_off/."""
import sys, os
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "experiments")); sys.path.insert(0, os.path.join(HERE, "rev"))
import q6_common
q6_common.M_FLOOR = 0
import e31_alarm_mechanism as e
e.M_FLOOR = 0
e.OUT = os.path.join(os.path.dirname(HERE), "e31_out_floor_off"); os.makedirs(e.OUT, exist_ok=True)
if __name__ == "__main__":
    e.main()
