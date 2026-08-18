# JAXViz examples

Each Python file in this directory becomes a standalone example page with source
code on the left and an interactive graph on the right.

## Add an example

Create `examples/my_example.py` with:

- `model`: the callable or module to trace.
- `example_input`: one example argument, or `example_args`: a tuple of arguments.
- `levels` (optional): a tuple containing `"high"`, `"low"`, or both.
- `trace_context` (optional): a context manager active while tracing.
- `trace_kwargs` (optional): extra keyword arguments passed to `trace_model`.
- `title` (optional): the page title; otherwise the module docstring is used.

The displayed snippet is generated from the example source. Generator metadata is
removed and the matching `trace_model` call is appended automatically.

## Generate pages

```bash
python scripts/examples_generator.py
```

Generate one example or one graph level while iterating:

```bash
python scripts/examples_generator.py flax_nnx_mlp_sharded --level low
```

Pages are written to `examples/generated/`. The files are standalone and can be
opened directly now or published as static assets later.
