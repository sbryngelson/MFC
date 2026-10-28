import re


def remove_higher_dimensional_keys(case: dict, ndims: int) -> dict:
    assert 1 <= ndims <= 3

    rm_dims = [set(), set(["y", "z"]), set(["z"]), set()][ndims]
    dim_ids = {dim: i + 1 for i, dim in enumerate(["x", "y", "z"])}
    dim_mnp = {"x": "m", "y": "n", "z": "p"}

    rm_keys = set()
    for key in case.keys():
        for dim in rm_dims:
            if any([re.match(f".+_{dim}", key), re.match(f"{dim}_.+", key), re.match(f"%{dim}", key), f"%vel({dim_ids[dim]})" in key]):
                rm_keys.add(key)
                break

    new_case = {k: v for k, v in case.items() if k not in rm_keys}

    for dim in rm_dims:
        new_case[dim_mnp[dim]] = 0

    return new_case


def ib_flux_wall_issue(p: dict, auto: bool):
    """First reason ib_flux_wall cannot apply to case p, or None. auto: the unset default, which needs every body to be a fixed,
    no-slip, non-blowing inert one; otherwise an explicit T, which needs fixed, non-blowing reacting bodies and at least one wall
    (an inert body that moves, slips or blows keeps the ghost-state wall)."""
    on = lambda k: str(p.get(k, "F")) == "T"  # noqa: E731
    num = lambda k, d: p.get(k, d) or d  # noqa: E731
    try:
        config = [
            (not (on("ib") and on("chemistry")), "ib_flux_wall requires ib and chemistry"),
            (on("igr"), "ib_flux_wall is not supported with igr"),
            (str(num("model_eqns", 2)) != "2", "ib_flux_wall requires model_eqns = 2"),
            (int(num("num_fluids", 1)) != 1, "ib_flux_wall requires num_fluids = 1"),
            (int(num("riemann_solver", 2)) != 2, "ib_flux_wall requires riemann_solver = 2 (HLLC)"),
            (on("cyl_coord"), "ib_flux_wall requires Cartesian coordinates"),
            (int(num("num_particle_clouds", 0)) > 0, "ib_flux_wall is not supported with particle clouds"),
            (on("ib_cutvol"), "ib_flux_wall is not supported with ib_cutvol"),
            (on("bubbles_euler"), "ib_flux_wall is not supported with bubbles"),
        ]
        for bad, msg in config:
            if bad:
                return msg
        walls = 0
        for i in range(1, int(num("num_ibs", 0)) + 1):
            fixed = int(num(f"patch_ib({i})%moving_ibm", 0)) == 0
            blow = float(num(f"patch_ib({i})%v_blow", 0.0)) > 0 or int(num(f"patch_ib({i})%inj_species", 0)) > 0
            if int(num(f"patch_ib({i})%surface_reaction", 0)) == 1:
                if auto:
                    return f"patch_ib({i}) reacts (set ib_flux_wall = T explicitly)"
                if not fixed:
                    return f"ib_flux_wall: reacting patch_ib({i}) must be fixed (moving_ibm = 0)"
                if blow:
                    return f"ib_flux_wall: reacting patch_ib({i}) cannot also blow (v_blow, inj_species)"
                walls += 1
            elif fixed and not blow and not on(f"patch_ib({i})%slip"):
                walls += 1
            elif auto:
                return f"patch_ib({i}) moves, slips or blows"
    except (TypeError, ValueError):  # an analytic (string) value
        return "ib_flux_wall: a patch_ib value is not a number"
    return None if walls else "ib_flux_wall requires a reacting patch_ib or a fixed, no-slip, non-blowing one"
