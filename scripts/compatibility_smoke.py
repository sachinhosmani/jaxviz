"""Smoke-test an installed JAXViz wheel across supported dependency stacks.

Run this script outside the repository working directory so Python imports the
installed package rather than the source checkout.
"""
import argparse
import importlib.util
import importlib.metadata
import os
import sys
from pathlib import Path


os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")

import jax
import jax.numpy as jnp
import jaxviz
from jaxviz import trace_model


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def version(distribution):
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def variable_value(variable):
    get_value = getattr(variable, "get_value", None)
    if get_value is not None:
        return get_value()
    return variable.value


def report_versions():
    package_path = Path(jaxviz.__file__).resolve()
    print("Python: {}".format(sys.version.split()[0]))
    print("JAX: {}".format(jax.__version__))
    print("Flax: {}".format(version("flax")))
    print("JAXViz: {}".format(version("jaxviz")))
    print("JAXViz path: {}".format(package_path))
    require(
        REPOSITORY_ROOT not in package_path.parents,
        "Imported JAXViz from the source checkout instead of the installed wheel",
    )


def smoke_raw_jax():
    def function(value):
        return jnp.tanh(value @ jnp.ones((4, 3)))

    html = trace_model(function, jnp.ones((2, 4)), return_html=True)
    require(isinstance(html, str), "Global trace did not return HTML")
    require("tanh" in html and "dot" in html, "Global trace omitted expected ops")
    print("PASS: raw JAX global view")


def import_nnx():
    try:
        from flax import nnx
    except (ImportError, AttributeError):
        return None
    return nnx


def smoke_nnx_global(nnx):
    class MLP(nnx.Module):
        def __init__(self, rngs):
            self.linear = nnx.Linear(4, 3, rngs=rngs)

        def __call__(self, value):
            return nnx.relu(self.linear(value))

    model = MLP(nnx.Rngs(0))
    html = trace_model(model, jnp.ones((2, 4)), return_html=True)
    require(isinstance(html, str), "NNX trace did not return HTML")
    require("relu" in html and "dot" in html, "NNX trace omitted expected ops")
    print("PASS: Flax NNX global view")


def distributed_apis_available(nnx):
    required_jax = (
        hasattr(jax, "make_mesh"),
        hasattr(jax, "set_mesh"),
        hasattr(jax, "P"),
        hasattr(jax.sharding, "AxisType"),
    )
    required_nnx = all(
        hasattr(nnx, name)
        for name in ("jit", "state", "get_partition_spec", "update", "with_partitioning")
    )
    return all(required_jax) and required_nnx


def smoke_distributed(nnx):
    try:
        jax.config.update("jax_num_cpu_devices", 8)
    except (AttributeError, RuntimeError, ValueError):
        pass

    devices = jax.devices("cpu")
    require(len(devices) >= 8, "Distributed smoke test requires 8 CPU devices")

    auto = jax.sharding.AxisType.Auto
    mesh = jax.make_mesh(
        (2, 4),
        ("data", "model"),
        axis_types=(auto, auto),
        devices=devices[:8],
    )

    class ShardedMLP(nnx.Module):
        def __init__(self, rngs):
            initializer = nnx.initializers.lecun_normal()
            self.dense0 = nnx.Linear(
                32,
                64,
                use_bias=False,
                rngs=rngs,
                kernel_init=nnx.with_partitioning(initializer, (None, "model")),
            )
            self.dense1 = nnx.Linear(
                64,
                32,
                use_bias=False,
                rngs=rngs,
                kernel_init=nnx.with_partitioning(initializer, ("model", None)),
            )

        def __call__(self, value):
            value = self.dense0(value)
            value = jax.lax.with_sharding_constraint(value, jax.P("data", "model"))
            value = nnx.relu(value)
            return self.dense1(value)

    @nnx.jit
    def create_sharded_model():
        model = ShardedMLP(nnx.Rngs(0))
        state = nnx.state(model)
        partition_specs = nnx.get_partition_spec(state)
        sharded_state = jax.lax.with_sharding_constraint(state, partition_specs)
        nnx.update(model, sharded_state)
        return model

    with jax.set_mesh(mesh):
        model = create_sharded_model()
        example_input = jax.device_put(
            jnp.ones((16, 32)),
            jax.P("data", None),
        )

        expected_dense0 = jax.sharding.NamedSharding(mesh, jax.P(None, "model"))
        expected_dense1 = jax.sharding.NamedSharding(mesh, jax.P("model", None))
        require(
            variable_value(model.dense0.kernel).sharding.is_equivalent_to(
                expected_dense0, ndim=2
            ),
            "dense0 kernel was not sharded over the model axis",
        )
        require(
            variable_value(model.dense1.kernel).sharding.is_equivalent_to(
                expected_dense1, ndim=2
            ),
            "dense1 kernel was not sharded over the model axis",
        )

        global_html = trace_model(model, example_input, view="global", return_html=True)
        local_html = trace_model(model, example_input, view="per_device", return_html=True)

    require("dot" in global_html, "Distributed global view omitted dot operations")
    require('"global": [16, 64]' in local_html, "Missing logical hidden shape")
    require('"local": [8, 16]' in local_html, "Missing per-device hidden shape")
    require('"name": "data"' in local_html, "Missing data mesh-axis annotation")
    require('"name": "model"' in local_html, "Missing model mesh-axis annotation")
    require("all-reduce" in local_html, "Missing model-parallel all-reduce")
    print("PASS: distributed global and per-device views")


def smoke_distributed_examples(examples_dir):
    example_names = (
        "flax_nnx_mlp_sharded",
        "flax_nnx_mlp_2d_parallel",
        "flax_nnx_head_parallel_attention",
    )
    for example_name in example_names:
        path = examples_dir / "{}.py".format(example_name)
        spec = importlib.util.spec_from_file_location(
            "jaxviz_compat_{}".format(example_name), path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        example_args = tuple(getattr(module, "example_args", (module.example_input,)))
        trace_kwargs = getattr(module, "trace_kwargs", {})
        with module.trace_context:
            for view in module.views:
                html = trace_model(
                    module.model,
                    *example_args,
                    view=view,
                    return_html=True,
                    **trace_kwargs
                )
                require(isinstance(html, str), "{} {} returned no HTML".format(
                    example_name, view
                ))
                if view == "per_device":
                    require('"status": "verified"' in html, "{} lacks verified shapes".format(
                        example_name
                    ))
        print("PASS: example {}".format(example_name))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--require-distributed",
        action="store_true",
        help="Fail instead of skipping when distributed APIs are unavailable",
    )
    parser.add_argument(
        "--examples-dir",
        type=Path,
        help="Also trace the repository's distributed example models",
    )
    args = parser.parse_args()

    report_versions()
    smoke_raw_jax()

    nnx = import_nnx()
    if nnx is None:
        print("SKIP: Flax NNX is unavailable")
        if args.require_distributed:
            raise AssertionError("Distributed smoke test requires Flax NNX")
        return

    smoke_nnx_global(nnx)
    if distributed_apis_available(nnx):
        smoke_distributed(nnx)
        if args.examples_dir is not None:
            smoke_distributed_examples(args.examples_dir.resolve())
    elif args.require_distributed:
        raise AssertionError("This stack lacks the distributed JAX/NNX APIs")
    else:
        print("SKIP: distributed JAX/NNX APIs are unavailable")


if __name__ == "__main__":
    main()
