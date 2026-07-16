import numpy as np

from handumi.robots.registry import (
    EMBODIMENT_NAMES,
    GripperJointRuntime,
    load_embodiment,
    load_robot_config,
)


def test_robot_names_are_discovered_from_yaml_configs():
    assert {"axol", "openarmv1", "piper"}.issubset(set(EMBODIMENT_NAMES))


def test_openarm_uses_configured_arm_joint_names_not_left_prefix():
    runtime = load_embodiment("openarmv1")

    assert runtime.arm_joint_names("left")[0] == "openarm_left_joint1"
    assert runtime.arm_joint_names("right")[0] == "openarm_right_joint1"
    assert runtime.arm_joint_indices("left") == [0, 1, 2, 3, 4, 5, 6, 14]
    assert runtime.arm_joint_indices("right") == [7, 8, 9, 10, 11, 12, 13, 15]
    assert runtime.finger_joints == {
        "left": (GripperJointRuntime(index=14, closed_value=0.0, open_value=0.044),),
        "right": (GripperJointRuntime(index=15, closed_value=0.0, open_value=0.044),),
    }


def test_openarm_arms_90_pose_bends_both_elbows_to_pi_over_two():
    runtime = load_embodiment("openarmv1")
    q = runtime.home_q("arms_90")

    np.testing.assert_allclose(q[[3, 10]], np.pi / 2, atol=1e-7)
    assert runtime.config.default_home_pose == "forward_open"


def test_openarm_default_pose_spreads_elbows_and_points_tcp_forward():
    runtime = load_embodiment("openarmv1")
    q = runtime.home_q()

    np.testing.assert_allclose(q[[1, 8]], [-np.pi / 9, np.pi / 9], atol=1e-7)
    np.testing.assert_allclose(q[[2, 9]], [np.pi / 18, -np.pi / 18], atol=1e-7)
    np.testing.assert_allclose(q[[3, 10]], np.pi / 2, atol=1e-7)


def test_openarm_gripper_mapping_matches_urdf_convention():
    runtime = load_embodiment("openarmv1")

    np.testing.assert_allclose(runtime.config.home_q[[14, 15]], [0.0, 0.0])

    q = runtime.config.home_q.copy()
    runtime.set_finger_positions(q, {"left": 0.0, "right": 1.0})
    np.testing.assert_allclose(q[[14, 15]], [0.0, 0.044])

    runtime.set_finger_positions(q, {"left": 1.0, "right": 0.0})
    np.testing.assert_allclose(q[[14, 15]], [0.044, 0.0])


def test_openarm_solver_fk_returns_two_tcp_poses():
    runtime = load_embodiment("openarmv1")
    solver = runtime.solver_cls()

    left_pose7, right_pose7 = solver.fk_pose7(runtime.config.home_q)

    assert left_pose7.shape == (7,)
    assert right_pose7.shape == (7,)
    np.testing.assert_allclose(np.linalg.norm(left_pose7[3:]), 1.0, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(right_pose7[3:]), 1.0, atol=1e-6)


def test_piper_mjcf_prefix_mapping_stays_in_robot_config():
    runtime = load_embodiment("piper")

    assert runtime.mjcf_actuator_name("left_joint1") == "izq_joint1"
    assert runtime.mjcf_actuator_name("right_joint8") == "der_joint8"


def test_legacy_ee_links_are_derived_from_arms():
    config = load_robot_config("openarmv1")

    assert config.ee_links == {
        "left": "openarm_left_hand_tcp",
        "right": "openarm_right_hand_tcp",
    }
