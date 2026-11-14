import json
from qiskit_ibm_runtime import QiskitRuntimeService
svc = QiskitRuntimeService()
ids = {"native":["db642islf4us73c4p8hg","db642l4vf2bc73d0krng","db642q4lf4us73c4p8u0","db642u04qg6s73c4738g"],
       "degraded":["db6430cvf2bc73d0ks9g","db6432imb58s738b7p40","db6436klf4us73c4p9gg","db643bcvf2bc73d0ksn0"]}
out = {}
for k, L in ids.items():
    for j in L:
        try:
            job = svc.job(j)
            r = job.result()
            md = r.metadata if hasattr(r, "metadata") else {}
            qpu = None
            try:
                pub = r[0]
                qpu = str(pub.metadata)[:300]
            except Exception as e:
                qpu = "pub err " + str(e)[:100]
            out[j] = dict(group=k, backend=job.backend().name if job.backend() else None, created=str(job.creation_date), meta=qpu)
            # layout from transpiled circuit
            try:
                circ = job.inputs["pubs"][0][0]
                lay = getattr(circ, "layout", None)
                out[j]["final_layout"] = str(lay.final_index_layout() if lay else None)
                out[j]["initial_layout"] = str(lay.initial_virtual_layout() if lay else None)
                out[j]["used_qubits"] = sorted({circ.find_bit(q).index for inst in circ.data for q in inst.qubits})
            except Exception as e:
                out[j]["layout_err"] = str(e)[:200]
        except Exception as e:
            out[j] = dict(group=k, err=str(e)[:200])
        print(j, out[j], flush=True)
json.dump(out, open("/tmp/x13_meta.json", "w"), indent=2)
