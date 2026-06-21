"""Tests for Object Affordance schema.

Tests the affordance.py module including affordance types,
spatial regions, and force requirements.
"""

from dvas.data.robot_schemas.affordance import (
    AffordanceAnnotation,
    AffordanceType,
    ForceRequirements,
    GraspConstraints,
    Handedness,
    ObjectAffordance,
    SingleAffordance,
    SpatialRegion,
)


class TestSpatialRegion:
    """Test spatial region representation."""

    def test_region_2d_initialization(self):
        """Test 2D region initialization."""
        region = SpatialRegion(
            bbox_2d=(0.1, 0.2, 0.3, 0.4),
            shape="rectangular",
        )
        assert region.bbox_2d == (0.1, 0.2, 0.3, 0.4)
        assert region.shape == "rectangular"

    def test_region_3d_initialization(self):
        """Test 3D region initialization."""
        region = SpatialRegion(
            bbox_3d=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
            center=(0.25, 0.35, 0.45),
        )
        assert region.bbox_3d == (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)

    def test_area_2d_calculation(self):
        """Test 2D area calculation."""
        region = SpatialRegion(bbox_2d=(0.0, 0.0, 0.5, 0.5))
        area = region.get_area_2d()
        assert area == 0.25

    def test_volume_3d_calculation(self):
        """Test 3D volume calculation."""
        region = SpatialRegion(bbox_3d=(0.0, 0.0, 0.0, 0.5, 0.5, 0.5))
        volume = region.get_volume_3d()
        assert volume == 0.125

    def test_region_serialization(self):
        """Test region round-trip."""
        original = SpatialRegion(
            bbox_2d=(0.1, 0.2, 0.3, 0.4),
            bbox_3d=(0.0, 0.0, 0.0, 1.0, 1.0, 1.0),
            center=(0.15, 0.25, 0.5),
            surface_normal=(0.0, 0.0, 1.0),
            shape="circular",
        )
        data = original.to_dict()
        restored = SpatialRegion.from_dict(data)

        assert restored.bbox_2d == original.bbox_2d
        assert restored.surface_normal == original.surface_normal

    def test_none_area(self):
        """Test area returns None when no 2D bbox."""
        region = SpatialRegion(bbox_3d=(0, 0, 0, 1, 1, 1))
        assert region.get_area_2d() is None


class TestForceRequirements:
    """Test force requirements representation."""

    def test_force_initialization(self):
        """Test force requirements initialization."""
        force = ForceRequirements(
            min_force=5.0,
            max_force=20.0,
            target_force=10.0,
        )
        assert force.min_force == 5.0
        assert force.max_force == 20.0

    def test_force_validation(self):
        """Test force validation."""
        force = ForceRequirements(min_force=5.0, max_force=20.0)

        # Valid force
        is_valid, msg = force.validate_force(10.0)
        assert is_valid is True

        # Too low
        is_valid, msg = force.validate_force(3.0)
        assert is_valid is False
        assert "below" in msg.lower()

        # Too high
        is_valid, msg = force.validate_force(25.0)
        assert is_valid is False
        assert "exceeds" in msg.lower()

    def test_torque_requirements(self):
        """Test torque in force requirements."""
        force = ForceRequirements(
            min_torque=0.5,
            max_torque=2.0,
        )
        assert force.min_torque == 0.5

    def test_force_serialization(self):
        """Test force requirements round-trip."""
        original = ForceRequirements(
            min_force=5.0,
            max_force=25.0,
            force_direction=(0.0, 0.0, 1.0),
            sustained=True,
            duration=3.0,
        )
        data = original.to_dict()
        restored = ForceRequirements.from_dict(data)

        assert restored.min_force == original.min_force
        assert restored.sustained == original.sustained
        assert restored.force_direction == original.force_direction


class TestGraspConstraints:
    """Test grasp constraints representation."""

    def test_grasp_defaults(self):
        """Test grasp constraint defaults."""
        constraints = GraspConstraints()
        assert constraints.min_fingers == 2
        assert constraints.max_fingers == 5
        assert constraints.power_grasp is True
        assert constraints.two_handed is False

    def test_grasp_specific_config(self):
        """Test specific grasp configuration."""
        constraints = GraspConstraints(
            min_fingers=3,
            max_fingers=4,
            preferred_hand_orientation="precision",
            pinch_grasp=True,
            power_grasp=False,
            two_handed=True,
            min_aperture=0.01,
            max_aperture=0.05,
        )
        assert constraints.min_fingers == 3
        assert constraints.preferred_hand_orientation == "precision"
        assert constraints.two_handed is True

    def test_grasp_serialization(self):
        """Test grasp constraints round-trip."""
        original = GraspConstraints(
            preferred_hand_orientation="hook",
            friction_coefficient=0.7,
        )
        data = original.to_dict()
        restored = GraspConstraints.from_dict(data)

        assert restored.preferred_hand_orientation == original.preferred_hand_orientation
        assert restored.friction_coefficient == original.friction_coefficient


class TestSingleAffordance:
    """Test single affordance representation."""

    def test_affordance_initialization(self):
        """Test basic affordance initialization."""
        aff = SingleAffordance(
            affordance_type=AffordanceType.GRASPABLE,
            confidence=0.95,
        )
        assert aff.affordance_type == AffordanceType.GRASPABLE
        assert aff.confidence == 0.95

    def test_affordance_types(self):
        """Test different affordance types."""
        types = [
            AffordanceType.GRASPABLE,
            AffordanceType.PUSHABLE,
            AffordanceType.LIFTABLE,
            AffordanceType.OPENABLE,
        ]
        for aff_type in types:
            aff = SingleAffordance(affordance_type=aff_type)
            assert aff.affordance_type == aff_type

    def test_affordance_with_region(self):
        """Test affordance with spatial region."""
        region = SpatialRegion(bbox_2d=(0.2, 0.3, 0.4, 0.5))
        aff = SingleAffordance(
            affordance_type=AffordanceType.GRASPABLE,
            regions=[region],
        )
        assert len(aff.regions) == 1

    def test_affordance_with_force(self):
        """Test affordance with force requirements."""
        force = ForceRequirements(min_force=5.0, max_force=15.0)
        aff = SingleAffordance(
            affordance_type=AffordanceType.PUSHABLE,
            force_requirements=force,
        )
        assert aff.force_requirements is not None

    def test_affordance_with_grasp_constraints(self):
        """Test graspable affordance with constraints."""
        grasp = GraspConstraints(
            pinch_grasp=True,
            max_aperture=0.08,
        )
        aff = SingleAffordance(
            affordance_type=AffordanceType.GRASPABLE,
            grasp_constraints=grasp,
        )
        assert aff.grasp_constraints is not None

    def test_affordance_validation(self):
        """Test affordance validation."""
        valid_aff = SingleAffordance(
            affordance_type=AffordanceType.PUSHABLE,
            confidence=0.8,
            difficulty=3,
        )
        is_valid, errors = valid_aff.validate()
        assert is_valid is True

    def test_affordance_validation_failures(self):
        """Test validation catches errors."""
        invalid_aff = SingleAffordance(
            affordance_type=AffordanceType.GRASPABLE,
            confidence=1.5,  # Invalid
            difficulty=6,  # Invalid
            # Missing grasp_constraints for GRASPABLE
        )
        is_valid, errors = invalid_aff.validate()
        assert is_valid is False
        assert len(errors) >= 2

    def test_get_primary_region(self):
        """Test getting primary region."""
        region1 = SpatialRegion(bbox_2d=(0.0, 0.0, 0.5, 0.5))  # Area = 0.25
        region2 = SpatialRegion(bbox_2d=(0.0, 0.0, 0.3, 0.3))  # Area = 0.09

        aff = SingleAffordance(
            affordance_type=AffordanceType.GRASPABLE,
            regions=[region2, region1],
        )

        primary = aff.get_primary_region()
        assert primary == region1  # Larger area

    def test_affordance_serialization(self):
        """Test affordance round-trip."""
        original = SingleAffordance(
            affordance_type=AffordanceType.CONTAINABLE,
            confidence=0.9,
            handedness=Handedness.BOTH,
            preconditions=["object_empty"],
            requires_tool=False,
            difficulty=2,
        )
        data = original.to_dict()
        restored = SingleAffordance.from_dict(data)

        assert restored.affordance_type == original.affordance_type
        assert restored.preconditions == original.preconditions
        assert restored.difficulty == original.difficulty


class TestObjectAffordance:
    """Test object affordance aggregation."""

    def test_object_affordance_initialization(self):
        """Test object affordance initialization."""
        obj_aff = ObjectAffordance(
            object_name="mug",
            category="container",
        )
        assert obj_aff.object_name == "mug"
        assert obj_aff.category == "container"

    def test_object_properties(self):
        """Test object physical properties."""
        obj_aff = ObjectAffordance(
            object_name="bowl",
            mass=0.5,
            dimensions=(0.15, 0.15, 0.08),
            material="ceramic",
            is_fragile=True,
        )
        assert obj_aff.mass == 0.5
        assert obj_aff.is_fragile is True

    def test_add_affordance(self):
        """Test adding affordance to object."""
        obj_aff = ObjectAffordance(object_name="cup")

        aff1 = SingleAffordance(affordance_type=AffordanceType.GRASPABLE)
        aff2 = SingleAffordance(affordance_type=AffordanceType.LIFTABLE)

        obj_aff.add_affordance(aff1)
        obj_aff.add_affordance(aff2)

        assert len(obj_aff.affordances) == 2

    def test_get_affordance(self):
        """Test getting affordances by type."""
        obj_aff = ObjectAffordance(object_name="bottle")
        obj_aff.add_affordance(SingleAffordance(affordance_type=AffordanceType.GRASPABLE))
        obj_aff.add_affordance(SingleAffordance(affordance_type=AffordanceType.POURABLE))
        obj_aff.add_affordance(SingleAffordance(affordance_type=AffordanceType.GRASPABLE))

        graspable = obj_aff.get_affordance(AffordanceType.GRASPABLE)
        assert len(graspable) == 2

        pourable = obj_aff.get_affordance(AffordanceType.POURABLE)
        assert len(pourable) == 1

    def test_has_affordance(self):
        """Test checking for affordance type."""
        obj_aff = ObjectAffordance(object_name="drawer")
        obj_aff.add_affordance(SingleAffordance(affordance_type=AffordanceType.OPENABLE))
        obj_aff.add_affordance(SingleAffordance(affordance_type=AffordanceType.CLOSEABLE))

        assert obj_aff.has_affordance(AffordanceType.OPENABLE) is True
        assert obj_aff.has_affordance(AffordanceType.GRASPABLE) is False

    def test_get_graspable_regions(self):
        """Test getting graspable regions."""
        region = SpatialRegion(bbox_2d=(0.2, 0.2, 0.4, 0.4))

        obj_aff = ObjectAffordance(object_name="box")
        grasp_aff = SingleAffordance(
            affordance_type=AffordanceType.GRASPABLE,
            regions=[region],
        )
        obj_aff.add_affordance(grasp_aff)

        regions = obj_aff.get_graspable_regions()
        assert len(regions) == 1

    def test_object_serialization(self):
        """Test object affordance round-trip."""
        obj_aff = ObjectAffordance(
            object_name="plate",
            category="dishware",
            mass=0.3,
            annotated_by="human",
        )
        obj_aff.add_affordance(SingleAffordance(affordance_type=AffordanceType.GRASPABLE))

        data = obj_aff.to_dict()
        restored = ObjectAffordance.from_dict(data)

        assert restored.object_name == obj_aff.object_name
        assert restored.mass == obj_aff.mass
        assert len(restored.affordances) == 1


class TestAffordanceAnnotation:
    """Test scene-level affordance annotation."""

    def test_scene_initialization(self):
        """Test scene affordance initialization."""
        scene = AffordanceAnnotation(scene_id="kitchen_001")
        assert scene.scene_id == "kitchen_001"
        assert len(scene.object_affordances) == 0

    def test_add_object_affordance(self):
        """Test adding object affordance to scene."""
        scene = AffordanceAnnotation(scene_id="scene_001")

        obj_aff = ObjectAffordance(object_name="spoon")
        obj_aff.add_affordance(SingleAffordance(affordance_type=AffordanceType.GRASPABLE))

        scene.add_object_affordance(obj_aff)

        assert len(scene.object_affordances) == 1
        assert "spoon" in scene.object_affordances

    def test_get_object_affordance(self):
        """Test getting object affordance by name."""
        scene = AffordanceAnnotation(scene_id="scene_002")

        obj_aff = ObjectAffordance(object_name="fork")
        scene.add_object_affordance(obj_aff)

        retrieved = scene.get_object_affordance("fork")
        assert retrieved is not None
        assert retrieved.object_name == "fork"

        missing = scene.get_object_affordance("knife")
        assert missing is None

    def test_find_graspable_objects(self):
        """Test finding all graspable objects."""
        scene = AffordanceAnnotation(scene_id="scene_003")

        cup = ObjectAffordance(object_name="cup")
        cup.add_affordance(SingleAffordance(affordance_type=AffordanceType.GRASPABLE))
        scene.add_object_affordance(cup)

        table = ObjectAffordance(object_name="table")
        table.add_affordance(SingleAffordance(affordance_type=AffordanceType.SUPPORTABLE))
        scene.add_object_affordance(table)

        graspable = scene.find_graspable_objects()
        assert "cup" in graspable
        assert "table" not in graspable

    def test_find_pushable_objects(self):
        """Test finding all pushable objects."""
        scene = AffordanceAnnotation(scene_id="scene_004")

        chair = ObjectAffordance(object_name="chair")
        chair.add_affordance(SingleAffordance(affordance_type=AffordanceType.PUSHABLE))
        scene.add_object_affordance(chair)

        pushable = scene.find_pushable_objects()
        assert "chair" in pushable

    def test_get_all_affordances_of_type(self):
        """Test getting all affordances of a type across objects."""
        scene = AffordanceAnnotation(scene_id="scene_005")

        obj1 = ObjectAffordance(object_name="obj1")
        obj1.add_affordance(SingleAffordance(affordance_type=AffordanceType.GRASPABLE))
        obj1.add_affordance(SingleAffordance(affordance_type=AffordanceType.LIFTABLE))
        scene.add_object_affordance(obj1)

        obj2 = ObjectAffordance(object_name="obj2")
        obj2.add_affordance(SingleAffordance(affordance_type=AffordanceType.GRASPABLE))
        scene.add_object_affordance(obj2)

        graspable = scene.get_all_affordances_of_type(AffordanceType.GRASPABLE)
        assert len(graspable) == 2

    def test_scene_serialization(self):
        """Test scene affordance round-trip."""
        scene = AffordanceAnnotation(
            scene_id="kitchen_main",
            timestamp="2024-01-15T10:30:00",
        )

        obj = ObjectAffordance(object_name="glass")
        obj.add_affordance(SingleAffordance(affordance_type=AffordanceType.GRASPABLE))
        scene.add_object_affordance(obj)

        data = scene.to_dict()
        restored = AffordanceAnnotation.from_dict(data)

        assert restored.scene_id == scene.scene_id
        assert len(restored.object_affordances) == 1
        assert "glass" in restored.object_affordances


class TestHandedness:
    """Test handedness enumeration."""

    def test_handedness_values(self):
        """Test all handedness values."""
        values = [Handedness.LEFT, Handedness.RIGHT, Handedness.BOTH, Handedness.EITHER]
        for val in values:
            aff = SingleAffordance(
                affordance_type=AffordanceType.GRASPABLE,
                handedness=val,
            )
            assert aff.handedness == val
