import asyncio
from types import SimpleNamespace

import numpy
import pytest
from scanspec.specs import Line
from tiled.catalog import in_memory
from tiled.client import Context, from_context
from tiled.server.app import build_app

from dls_tiled.visr import compute_binned_image, get_setpoints, visr_router


@pytest.fixture
def client(tmp_path):
    catalog = in_memory(writable_storage=str(tmp_path))
    app = build_app(catalog, include_routers=[visr_router])
    with Context.from_app(app) as context:
        yield from_context(context)


def write_step_scan(client, uid="scan"):
    """A 2x2 grid of step-scan points, two points per grid cell."""
    primary = client.create_container(uid).create_container("primary")
    x = numpy.array([0.0, 0.1, 1.0, 1.1, 0.0, 0.1, 1.0, 1.1])
    y = numpy.array([0.0, 0.1, 0.0, 0.1, 1.0, 1.1, 1.0, 1.1])
    primary.write_array(x, key="X")
    primary.write_array(y, key="Y")
    for channel, scale in (("RedTotal", 1), ("GreenTotal", 10), ("BlueTotal", 100)):
        primary.write_array(numpy.arange(8.0) * scale, key=channel)


def test_binned_step_scan(client):
    write_step_scan(client)

    response = client.context.http_client.get(
        "/api/v1/binned/scan", params={"width": 2, "height": 2}
    )

    assert response.status_code == 200, response.text
    body = response.json()
    # mean of the two points in each (x, y) cell
    assert body["RedTotal"] == [[0.5, 4.5], [2.5, 6.5]]
    assert body["GreenTotal"] == [[5.0, 45.0], [25.0, 65.0]]
    assert body["x_limits"] == [0.0, 0.55, 1.1]


def test_binned_missing_run_is_422(client):
    response = client.context.http_client.get("/api/v1/binned/nonexistent")

    assert response.status_code == 422


def test_compute_binned_image_empty_bins_are_zero():
    result = compute_binned_image(
        numpy.array([2.0]),
        numpy.array([0.0]),
        numpy.array([0.0]),
        bins=2,
        range=((0, 1), (0, 1)),
    )

    assert result["img"].tolist() == [[2.0, 0.0], [0.0, 0.0]]


def _setpoints(spec, **start):
    """Run get_setpoints against a fake root whose run has this start document."""
    metadata = {"start": {"spec": spec.serialize(), **start}}
    adapter = SimpleNamespace(metadata=lambda: metadata)

    async def lookup_adapter(path):
        return adapter

    return asyncio.run(
        get_setpoints(SimpleNamespace(lookup_adapter=lookup_adapter), "uid")
    )


def test_setpoints_follow_axis_names_not_dimension_order():
    # Raster scan: y is the outer dimension, x the inner one.
    spec = Line("sample_stage-y", 0, 1, 2) * Line("sample_stage-x", 0, 10, 3)

    x, y, z = _setpoints(spec)

    assert x.tolist() == [0, 5, 10, 0, 5, 10]
    assert y.tolist() == [0, 0, 0, 1, 1, 1]
    assert numpy.isnan(z).all()


def test_setpoints_of_runs_with_repr_axes_use_the_motors_list():
    spec = Line("<Motor object at 0x1>", 0, 1, 2) * Line(
        "<Motor object at 0x2>", 0, 10, 3
    )

    x, y, _ = _setpoints(spec, motors=["sample_stage-y", "sample_stage-x"])

    assert x.tolist() == [0, 5, 10, 0, 5, 10]
    assert y.tolist() == [0, 0, 0, 1, 1, 1]


def test_setpoints_keep_dimension_order_when_axes_are_not_x_y_z():
    spec = Line("theta", 0, 1, 2) * Line("energy", 0, 10, 3)

    x, y, _ = _setpoints(spec)

    assert x.tolist() == [0, 0, 0, 1, 1, 1]
    assert y.tolist() == [0, 5, 10, 0, 5, 10]
