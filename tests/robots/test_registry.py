import numpy as np

from handumi.robots.registry import (
    EMBODIMENT_NAMES,
    GripperJointRuntime,
    load_embodiment,
    load_robot_config,
)


def test_robot_names_are_discovered_from_yaml_configs():
    assert {"axol", "openarmv1", "piper", "trlc_dk1", "yam"}.issubset(
        set(EMBODIMENT_NAMES)
    )


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


def test_openarm_default_pose_spreads_elbows_and_points_tcp_forward():
    runtime = load_embodiment("openarmv1")
    q = runtime.home_q()

    assert runtime.config.real.backend == "openarm_can"
    np.testing.assert_allclose(q[[1, 8]], [-np.pi / 9, np.pi / 9], atol=1e-7)
    np.testing.assert_allclose(q[[2, 9]], [np.pi / 18, -np.pi / 18], atol=1e-7)
    np.testing.assert_allclose(q[[3, 10]], np.pi / 2, atol=1e-7)
    assert runtime.config.replay_max_joint_delta == 0.35


def test_openarm_gripper_mapping_matches_urdf_convention():
    runtime = load_embodiment("openarmv1")

    np.testing.assert_allclose(runtime.config.home_q[[14, 15]], [0.0, 0.0])

    q = runtime.config.home_q.copy()
    runtime.set_finger_positions(q, {"left": 0.0, "right": 1.0})
    np.testing.assert_allclose(q[[14, 15]], [0.0, 0.044])

    runtime.set_finger_positions(q, {"left": 1.0, "right": 0.0})
    np.testing.assert_allclose(q[[14, 15]], [0.044, 0.0])


def test_openarm_closed_fingers_keep_the_urdf_center_clear():
    runtime = load_embodiment("openarmv1")
    urdf = runtime.load_urdf()

    for side in ("left", "right"):
        first = urdf.joint_map[f"openarm_{side}_finger_joint1"]
        second = urdf.joint_map[f"openarm_{side}_finger_joint2"]
        np.testing.assert_allclose(first.origin[:3, 3], [0.0, -0.008, 0.1025])
        np.testing.assert_allclose(second.origin[:3, 3], [0.0, 0.008, 0.1025])


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
    assert runtime.mjcf_actuator_name("right_joint7") == "der_joint7"


def test_piper_uses_one_gripper_joint_per_arm():
    runtime = load_embodiment("piper")

    assert runtime.robot.joints.num_actuated_joints == 14
    assert runtime.arm_joint_names("left") == [f"left_joint{i}" for i in range(1, 8)]
    assert runtime.arm_joint_names("right") == [
        f"right_joint{i}" for i in range(1, 8)
    ]
    assert runtime.arm_joint_indices("left") == list(range(7))
    assert runtime.arm_joint_indices("right") == list(range(7, 14))
    assert runtime.finger_joints == {
        "left": (GripperJointRuntime(index=6, closed_value=0.0, open_value=0.035),),
        "right": (GripperJointRuntime(index=13, closed_value=0.0, open_value=0.035),),
    }

    q = runtime.home_q()
    runtime.set_finger_positions(q, {"left": 0.0, "right": 1.0})
    np.testing.assert_allclose(q[[6, 13]], [0.0, 0.035])


def test_axol_bimanual_layout_and_replay_profile():
    runtime = load_embodiment("axol")

    assert runtime.arm_joint_names("left") == [
        "left_e1_0",
        "left_e2_0",
        "left_s1_0",
        "left_s2_0",
        "left_s3_0",
        "left_w1_0",
        "left_w2_0",
    ]
    assert runtime.arm_joint_names("right") == [
        "right_e1_0",
        "right_e2_0",
        "right_s1_0",
        "right_s2_0",
        "right_s3_0",
        "right_w1_0",
        "right_w2_0",
    ]
    assert runtime.arm_joint_indices("left") == list(range(7))
    assert runtime.arm_joint_indices("right") == list(range(7, 14))
    assert runtime.config.replay_max_joint_delta == 0.35
    assert runtime.finger_joints == {"left": (), "right": ()}


def test_axol_home_fk_is_symmetric_and_visual_meshes_load():
    runtime = load_embodiment("axol")
    solver = runtime.solver_cls()

    left_pose7, right_pose7 = solver.fk_pose7(runtime.config.home_q)
    np.testing.assert_allclose(left_pose7[0], -right_pose7[0], atol=1e-6)
    np.testing.assert_allclose(left_pose7[1:3], right_pose7[1:3], atol=1e-6)

    urdf = runtime.load_urdf(load_meshes=True)
    assert urdf.scene is not None
    assert len(urdf.scene.geometry) == 18


def test_trlc_dk1_bimanual_layout_and_joint_mapping():
    runtime = load_embodiment("trlc_dk1")

    assert runtime.arm_joint_names("left") == [
        f"left_joint{i}" for i in range(1, 7)
    ]
    assert runtime.arm_joint_names("right") == [
        f"right_joint{i}" for i in range(1, 7)
    ]
    assert runtime.arm_joint_indices("left") == list(range(6))
    assert runtime.arm_joint_indices("right") == list(range(8, 14))

    solver = runtime.solver_cls()
    left_pose7, right_pose7 = solver.fk_pose7(runtime.config.home_q)
    np.testing.assert_allclose(left_pose7[[0, 2]], right_pose7[[0, 2]], atol=1e-6)
    np.testing.assert_allclose(left_pose7[1] - right_pose7[1], 0.60, atol=1e-6)


def test_trlc_dk1_visual_meshes_resolve_from_asset_root():
    runtime = load_embodiment("trlc_dk1")
    urdf = runtime.load_urdf(load_meshes=True)

    assert urdf.scene is not None
    # Each GLB contains several submeshes, so the resulting scene has more
    # entries than the 18 visual elements referenced by the bimanual URDF.
    assert len(urdf.scene.geometry) >= 18


def test_trlc_dk1_gripper_mapping_avoids_visual_finger_overlap():
    runtime = load_embodiment("trlc_dk1")
    finger_indices = [6, 7, 14, 15]

    assert runtime.config.gripper_max_width_m == 0.082
    np.testing.assert_allclose(runtime.config.home_q[finger_indices], 0.001)
    q = runtime.config.home_q.copy()
    runtime.set_finger_positions(q, {"left": 0.0, "right": 1.0})
    np.testing.assert_allclose(q[finger_indices], [-0.040, -0.040, 0.001, 0.001])

    runtime.set_finger_positions(q, {"left": 1.0, "right": 0.0})
    np.testing.assert_allclose(q[finger_indices], [0.001, 0.001, -0.040, -0.040])


def test_yam_bimanual_layout_and_forward_home():
    runtime = load_embodiment("yam")

    assert runtime.arm_joint_names("left") == [
        f"left_joint{i}" for i in range(1, 7)
    ]
    assert runtime.arm_joint_names("right") == [
        f"right_joint{i}" for i in range(1, 7)
    ]
    assert runtime.arm_joint_indices("left") == list(range(6))
    assert runtime.arm_joint_indices("right") == list(range(8, 14))
    assert runtime.config.replay_max_joint_delta == 0.35
    assert runtime.config.replay_gripper_mode == "physical-width"

    left, right = runtime.solver_cls().fk_pose7(runtime.home_q())
    np.testing.assert_allclose(left[0], -0.30, atol=1e-5)
    np.testing.assert_allclose(right[0], 0.30, atol=1e-5)
    np.testing.assert_allclose(left[1:3], right[1:3], atol=1e-6)
    np.testing.assert_allclose(left[1:3], [0.5092962, 0.4375029], atol=1e-5)


def test_yam_linear_4310_gripper_mapping_and_visual_meshes():
    runtime = load_embodiment("yam")
    finger_indices = [6, 7, 14, 15]

    np.testing.assert_allclose(runtime.home_q()[finger_indices], -0.04695)
    q = runtime.home_q()
    runtime.set_finger_positions(q, {"left": 0.0, "right": 1.0})
    np.testing.assert_allclose(q[finger_indices], [0.0, 0.0, -0.04695, -0.04695])
    runtime.set_finger_positions(q, {"left": 1.0, "right": 0.0})
    np.testing.assert_allclose(q[finger_indices], [-0.04695, -0.04695, 0.0, 0.0])

    urdf = runtime.load_urdf(load_meshes=True)
    assert urdf.scene is not None
    assert len(urdf.scene.geometry) == 25


def test_legacy_ee_links_are_derived_from_arms():
    config = load_robot_config("openarmv1")

    assert config.ee_links == {
        "left": "openarm_left_hand_tcp",
        "right": "openarm_right_hand_tcp",
    }
