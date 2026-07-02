from ccl_unified_benchmark import (
    apply_rules,
    encode_task_to_limb_state,
    route_rules,
    select_task_signal,
)

MIN_BASELINE_LIMB_ACTIVATION = 0.1


def test_apply_rules_compound_pipeline_matches_expected_output():
    grid = [
        [0, 1, 0],
        [2, 0, 0],
        [0, 3, 0],
    ]
    transformed = apply_rules(grid, ["flip_h", "gravity_down"])

    assert transformed == [
        [0, 0, 0],
        [0, 1, 0],
        [0, 3, 2],
    ]


def test_encode_task_to_limb_state_prioritizes_spatial_and_action_limbs():
    grid = [[0] * 8 for _ in range(8)]
    grid[0][0] = 1
    grid[1][1] = 2

    encoded = encode_task_to_limb_state(grid, ["rot_cw", "gravity_down"], level=2)

    assert len(encoded) == 8
    assert encoded[4] > MIN_BASELINE_LIMB_ACTIVATION
    assert encoded[6] > MIN_BASELINE_LIMB_ACTIVATION


def test_route_rules_and_signal_for_compound_rules():
    rules = ["rot_cw", "gravity_down", "sort_rows"]

    routes = route_rules(rules)

    assert [item["rule"] for item in routes] == rules
    assert select_task_signal(rules) == "reasoning"
    assert routes[0]["domain"] == "spatial"
    assert routes[1]["domain"] == "action"
    assert routes[2]["domain"] == "reasoning"
