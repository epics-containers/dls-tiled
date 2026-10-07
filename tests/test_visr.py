import asyncio
import json
from types import SimpleNamespace

import numpy
import pytest
from fastapi import HTTPException
from scanspec.specs import Line, Spiral
from tiled.adapters.utils import DataNotReadyError
from tiled.catalog import in_memory
from tiled.client import Context, from_context
from tiled.server.app import build_app

from dls_tiled import visr
from dls_tiled.visr import (
    compute_binned_image,
    get_data,
    get_setpoints,
    visr_router,
)


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
    # mean of the two points in each cell; one row per y bin, one column per x bin
    assert body["RedTotal"] == [[0.5, 2.5], [4.5, 6.5]]
    assert body["GreenTotal"] == [[5.0, 25.0], [45.0, 65.0]]
    assert body["x_limits"] == [0.0, 0.55, 1.1]
    assert body["y_limits"] == [0.0, 0.55, 1.1]
    assert body["n_points"] == 8


def test_binned_rows_are_y_and_columns_are_x(client):
    # x only takes two values and y four, so the image must be 4 rows by 2 columns
    primary = client.create_container("raster").create_container("primary")
    x = numpy.array([0.0, 1.0] * 4)
    y = numpy.repeat([0.0, 1.0, 2.0, 3.0], 2)
    primary.write_array(x, key="X")
    primary.write_array(y, key="Y")
    for channel in ("RedTotal", "GreenTotal", "BlueTotal"):
        primary.write_array(numpy.arange(8.0), key=channel)

    response = client.context.http_client.get(
        "/api/v1/binned/raster", params={"width": 2, "height": 4}
    )

    body = response.json()
    assert body["RedTotal"] == [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0]]
    assert len(body["x_limits"]) == 3 and len(body["y_limits"]) == 5


def test_binned_reports_empty_bins_as_null_and_keeps_a_measured_zero(client):
    primary = client.create_container("sparse").create_container("primary")
    primary.write_array(numpy.array([0.0, 1.0]), key="X")
    primary.write_array(numpy.array([0.0, 1.0]), key="Y")
    for channel, values in (
        ("RedTotal", [0.0, 5.0]),
        ("GreenTotal", [1.0, 2.0]),
        ("BlueTotal", [3.0, 4.0]),
    ):
        primary.write_array(numpy.array(values), key=channel)

    response = client.context.http_client.get(
        "/api/v1/binned/sparse", params={"width": 2, "height": 2}
    )

    # a measured 0.0 stays 0.0; the two cells no point landed in are null
    assert response.json()["RedTotal"] == [[0.0, None], [None, 5.0]]


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
    assert result["counts"].tolist() == [[1.0, 0.0], [0.0, 0.0]]


def test_image_rows_are_y_transposes_and_marks_empty_bins():
    img = numpy.array([[1.0, 2.0, 0.0], [3.0, 0.0, 0.0]])  # [x_bin, y_bin]
    counts = numpy.array([[1, 1, 0], [1, 0, 0]])

    assert visr.image_rows_are_y(img, counts) == [
        [1.0, 3.0],
        [2.0, None],
        [None, None],
    ]


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


class _LaggingAdapter:
    """A leaf whose file is behind the catalog for the first `lag` reads."""

    def __init__(self, lag):
        self.lag = lag
        self.reads = 0

    def read(self):
        self.reads += 1
        if self.reads <= self.lag:
            raise DataNotReadyError("advertises (25,), only (24,) available")
        return numpy.arange(25.0)


def _root_with(adapter):
    async def lookup_adapter(segments):
        return adapter

    return SimpleNamespace(lookup_adapter=lookup_adapter)


@pytest.fixture
def fast_retries(monkeypatch):
    monkeypatch.setattr(visr, "NOT_READY_RETRY_DELAY", 0.0)


def test_get_data_retries_a_file_that_is_behind_the_catalog(fast_retries):
    adapter = _LaggingAdapter(lag=2)

    data = asyncio.run(get_data(_root_with(adapter), ["uid", "primary", "RedTotal"]))

    assert isinstance(data, numpy.ndarray)
    assert data.shape == (25,)
    assert adapter.reads == 3


def test_get_data_answers_503_with_retry_after_when_the_file_stays_behind(fast_retries):
    adapter = _LaggingAdapter(lag=100)

    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(get_data(_root_with(adapter), ["uid", "primary", "RedTotal"]))

    assert excinfo.value.status_code == 503
    assert excinfo.value.headers == {"Retry-After": "1"}
    assert adapter.reads == visr.NOT_READY_RETRIES + 1


def test_binned_keeps_a_503_instead_of_turning_it_into_422(client, monkeypatch):
    write_step_scan(client)

    async def not_ready(root, segments):
        raise HTTPException(
            status_code=503, detail="not ready", headers={"Retry-After": "1"}
        )

    monkeypatch.setattr(visr, "get_data", not_ready)

    response = client.context.http_client.get("/api/v1/binned/scan")

    assert response.status_code == 503
    assert response.headers["retry-after"] == "1"


# --- the scan's own grid ------------------------------------------------------------


def _grid_or_none(spec, motors=None):
    return visr._grid_from_spec(json.dumps(spec.serialize()), json.dumps(motors))


def _grid(spec, motors=None):
    grid = _grid_or_none(spec, motors)
    assert grid is not None
    return grid


def test_grid_edges_sit_halfway_between_setpoints():
    spec = Line("sample_stage-y", 0, 1, 2) * ~Line("sample_stage-x", 0, 10, 3)

    x_edges, y_edges, n_total = _grid(spec)

    assert x_edges.tolist() == [-2.5, 2.5, 7.5, 12.5]
    assert y_edges.tolist() == [-0.5, 0.5, 1.5]
    assert n_total == 6


def test_grid_of_runs_with_repr_axes_uses_the_motors_list():
    spec = Line("<Motor object at 0x1>", 0, 1, 2) * Line(
        "<Motor object at 0x2>", 0, 10, 3
    )

    x_edges, y_edges, _ = _grid(spec, ["sample_stage-y", "sample_stage-x"])

    assert len(x_edges) == 4 and len(y_edges) == 3


def test_grid_is_none_when_the_axes_are_not_x_and_y():
    assert _grid_or_none(Line("theta", 0, 1, 2) * Line("energy", 0, 10, 3)) is None


def test_grid_is_none_when_the_points_do_not_lie_on_a_grid():
    assert (
        _grid_or_none(
            Spiral(
                x_axis="x", x_centre=0, x_diameter=5, x_step=1, y_axis="y", y_centre=0
            )
        )
        is None
    )


def test_grid_with_a_single_position_gets_a_unit_wide_cell():
    x_edges, _, _ = _grid(Line("y", 0, 1, 2) * Line("x", 3, 3, 1))

    assert x_edges.tolist() == [2.5, 3.5]


def write_grid_scan(client, uid, n_points):
    """The first `n_points` of a 2 (x) by 3 (y) raster; its spec is in the start doc."""
    spec = Line("Y", 0, 2, 3) * Line("X", 0, 1, 2)
    metadata = {"start": {"spec": spec.serialize(), "motors": ["Y", "X"]}}
    primary = client.create_container(uid, metadata=metadata).create_container(
        "primary"
    )
    x = numpy.array([0.0, 1.0] * 3)[:n_points]
    y = numpy.repeat([0.0, 1.0, 2.0], 2)[:n_points]
    primary.write_array(x, key="X")
    primary.write_array(y, key="Y")
    for channel in ("RedTotal", "GreenTotal", "BlueTotal"):
        primary.write_array(numpy.arange(6.0)[:n_points], key=channel)


def test_binned_uses_the_scans_grid_without_being_asked(client):
    write_grid_scan(client, "grid", 6)

    body = client.context.http_client.get("/api/v1/binned/grid").json()

    assert body["RedTotal"] == [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]
    assert body["x_limits"] == [-0.5, 0.5, 1.5]
    assert body["y_limits"] == [-0.5, 0.5, 1.5, 2.5]
    assert (body["n_points"], body["n_total"]) == (6, 6)


def test_binned_grid_is_the_same_shape_from_the_first_points_of_a_scan(client):
    write_grid_scan(client, "partial", 4)  # the last row hasn't been measured yet

    body = client.context.http_client.get("/api/v1/binned/partial").json()

    assert body["RedTotal"] == [[0.0, 1.0], [2.0, 3.0], [None, None]]
    assert body["y_limits"] == [-0.5, 0.5, 1.5, 2.5]
    assert (body["n_points"], body["n_total"]) == (4, 6)


def test_binned_explicit_width_and_height_override_the_grid(client):
    write_grid_scan(client, "chosen", 6)

    body = client.context.http_client.get(
        "/api/v1/binned/chosen", params={"width": 1, "height": 1}
    ).json()

    assert body["RedTotal"] == [[2.5]]


def test_binned_without_a_spec_still_reports_no_total(client):
    write_step_scan(client)

    body = client.context.http_client.get("/api/v1/binned/scan").json()

    assert body["n_total"] is None
