# JAXViz examples

Each Python file in this directory becomes a standalone example page with source
code on the left and an interactive graph on the right.

## Add an example

Create `examples/my_example.py` with:

- `model`: the callable or module to trace.
- `example_input`: one example argument, or `example_args`: a tuple of arguments.
- `views` (optional): a tuple containing `"global"`, `"per_device"`, or both.
- `trace_context` (optional): a context manager active while tracing.
- `trace_kwargs` (optional): extra keyword arguments passed to `trace_model`.
- `title` (optional): the page title; otherwise the module docstring is used.

The displayed snippet is generated from the example source. Generator metadata is
removed and the matching `trace_model` call is appended automatically.

## Generate pages

```bash
python scripts/examples_generator.py
```

Generate one example or one program view while iterating:

```bash
python scripts/examples_generator.py flax_nnx_mlp_sharded --view per_device
```

Pages are written to `docs/examples/`. The files are standalone and can be
opened directly now or published as static assets later.

## Distributed examples

The distributed tutorials intentionally emulate eight logical CPU devices so
their `2 × 4` and `8`-way meshes run without accelerator hardware. Run their
displayed snippets in a fresh Python process, before any other JAX operation.
Production programs should instead construct meshes from their real devices.

The Flax NNX distributed examples use the explicit state-sharding initialization
sequence shared by Flax 0.11 and 0.12. Their intermediate activation constraints
make the illustrated partitioning deterministic rather than compiler-dependent.
