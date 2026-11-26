# Contributing

Thanks for looking! kernelforge is small on purpose: every pass should stay readable.

1. `pip install -e .[dev]` (numpy, pytest, ziglang for a C compiler on Windows) and, for GPU tests,
   `npm install` (Node >= 22.18 runs the TypeScript harness directly).
2. `python -m pytest -q`: compiler, C oracle, MSL syntax check; WebGPU tests run when a GPU adapter
   is found and skip with a reason otherwise.
3. New op? Add it to `graph.py` (with its NumPy semantics), `compute.py` (its index expression) and a
   zoo graph in `zoo.py`; the correctness tests then cover it on every backend automatically.
4. New schedule knob? Add it to the template and to `space()`; `tests/test_correctness.py` samples
   random points of every space against the oracle.
5. Benchmark numbers in the README must come from `bench/run_bench.py` on real hardware, with the
   machine and date recorded. Never edit them by hand.

Please keep PRs focused and include the test command you ran.
