"""Tests for data governance module.

Tests for DataGovernance, DataAccessPolicy, DataAccessLevel,
RetentionPolicy, RetentionType, and LineageRecord.
"""

import time


from dvas.governance.data_governance import (
    DataAccessLevel,
    DataAccessPolicy,
    DataGovernance,
    LineageRecord,
    RetentionPolicy,
    RetentionType,
)


class TestDataAccessLevel:
    """Test DataAccessLevel enum."""

    def test_level_values(self):
        """Test access level values."""
        assert DataAccessLevel.PUBLIC.value == "public"
        assert DataAccessLevel.INTERNAL.value == "internal"
        assert DataAccessLevel.RESTRICTED.value == "restricted"
        assert DataAccessLevel.CONFIDENTIAL.value == "confidential"


class TestRetentionType:
    """Test RetentionType enum."""

    def test_type_values(self):
        """Test retention type values."""
        assert RetentionType.TIME_BASED.value == "time_based"
        assert RetentionType.EVENT_BASED.value == "event_based"
        assert RetentionType.INDEFINITE.value == "indefinite"


class TestDataAccessPolicy:
    """Test DataAccessPolicy dataclass."""

    def test_policy_creation(self):
        """Test creating an access policy."""
        policy = DataAccessPolicy(
            user_id="user_001",
            data_id="data_001",
            level=DataAccessLevel.RESTRICTED,
        )
        assert policy.user_id == "user_001"
        assert policy.data_id == "data_001"
        assert policy.level == DataAccessLevel.RESTRICTED

    def test_policy_expired(self):
        """Test expired policy."""
        policy = DataAccessPolicy(
            user_id="user_001",
            data_id="data_001",
            level=DataAccessLevel.RESTRICTED,
            expires_at=time.time() - 1,
        )
        assert policy.is_expired() is True

    def test_policy_not_expired(self):
        """Test non-expired policy."""
        policy = DataAccessPolicy(
            user_id="user_001",
            data_id="data_001",
            level=DataAccessLevel.RESTRICTED,
            expires_at=time.time() + 3600,
        )
        assert policy.is_expired() is False

    def test_policy_no_expiration(self):
        """Test policy with no expiration."""
        policy = DataAccessPolicy(
            user_id="user_001",
            data_id="data_001",
            level=DataAccessLevel.RESTRICTED,
        )
        assert policy.is_expired() is False

    def test_policy_to_dict(self):
        """Test converting policy to dict."""
        policy = DataAccessPolicy(
            user_id="user_001",
            data_id="data_001",
            level=DataAccessLevel.INTERNAL,
        )
        d = policy.to_dict()
        assert d["user_id"] == "user_001"
        assert d["level"] == "internal"


class TestRetentionPolicy:
    """Test RetentionPolicy dataclass."""

    def test_policy_creation(self):
        """Test creating a retention policy."""
        policy = RetentionPolicy(
            data_type="annotation",
            retention_type=RetentionType.TIME_BASED,
            duration_days=365,
            action="archive",
        )
        assert policy.data_type == "annotation"
        assert policy.retention_type == RetentionType.TIME_BASED
        assert policy.duration_days == 365

    def test_policy_to_dict(self):
        """Test converting retention policy to dict."""
        policy = RetentionPolicy(
            data_type="annotation",
            retention_type=RetentionType.TIME_BASED,
            duration_days=365,
        )
        d = policy.to_dict()
        assert d["data_type"] == "annotation"
        assert d["retention_type"] == "time_based"


class TestLineageRecord:
    """Test LineageRecord dataclass."""

    def test_record_creation(self):
        """Test creating a lineage record."""
        record = LineageRecord(
            record_id="data_002",
            source_id="data_001",
            operation="derived",
            user_id="user_001",
        )
        assert record.record_id == "data_002"
        assert record.source_id == "data_001"
        assert record.operation == "derived"

    def test_record_to_dict(self):
        """Test converting lineage record to dict."""
        record = LineageRecord(
            record_id="data_002",
            source_id="data_001",
            operation="derived",
        )
        d = record.to_dict()
        assert d["record_id"] == "data_002"
        assert d["source_id"] == "data_001"


class TestDataGovernance:
    """Test DataGovernance class."""

    def test_init(self):
        """Test initialization."""
        governance = DataGovernance()
        assert governance is not None

    def test_grant_access(self):
        """Test granting access."""
        governance = DataGovernance()
        policy = governance.grant_access(
            "user_001",
            "data_001",
            DataAccessLevel.RESTRICTED,
        )
        assert policy.user_id == "user_001"
        assert policy.data_id == "data_001"

    def test_check_access_granted(self):
        """Test checking granted access."""
        governance = DataGovernance()
        governance.grant_access("user_001", "data_001", DataAccessLevel.RESTRICTED)
        assert governance.check_access("user_001", "data_001", DataAccessLevel.RESTRICTED) is True

    def test_check_access_higher_level(self):
        """Test checking access with higher required level."""
        governance = DataGovernance()
        governance.grant_access("user_001", "data_001", DataAccessLevel.INTERNAL)
        # INTERNAL < RESTRICTED
        assert governance.check_access("user_001", "data_001", DataAccessLevel.RESTRICTED) is False

    def test_check_access_lower_level(self):
        """Test checking access with lower required level."""
        governance = DataGovernance()
        governance.grant_access("user_001", "data_001", DataAccessLevel.RESTRICTED)
        # RESTRICTED >= INTERNAL
        assert governance.check_access("user_001", "data_001", DataAccessLevel.INTERNAL) is True

    def test_check_access_not_granted(self):
        """Test checking non-granted access."""
        governance = DataGovernance()
        assert governance.check_access("user_001", "data_001", DataAccessLevel.PUBLIC) is False

    def test_check_access_expired(self):
        """Test checking expired access."""
        governance = DataGovernance()
        governance.grant_access(
            "user_001",
            "data_001",
            DataAccessLevel.RESTRICTED,
            expires_at=time.time() - 1,
        )
        assert governance.check_access("user_001", "data_001", DataAccessLevel.RESTRICTED) is False

    def test_revoke_access(self):
        """Test revoking access."""
        governance = DataGovernance()
        governance.grant_access("user_001", "data_001", DataAccessLevel.RESTRICTED)
        assert governance.revoke_access("user_001", "data_001") is True
        assert governance.check_access("user_001", "data_001", DataAccessLevel.RESTRICTED) is False

    def test_revoke_access_not_found(self):
        """Test revoking non-existent access."""
        governance = DataGovernance()
        assert governance.revoke_access("user_001", "data_001") is False

    def test_get_access_policies(self):
        """Test getting access policies."""
        governance = DataGovernance()
        governance.grant_access("user_001", "data_001", DataAccessLevel.RESTRICTED)
        policies = governance.get_access_policies("data_001")
        assert len(policies) == 1

    def test_add_retention_policy(self):
        """Test adding retention policy."""
        governance = DataGovernance()
        policy = RetentionPolicy(
            data_type="annotation",
            retention_type=RetentionType.TIME_BASED,
            duration_days=365,
        )
        governance.add_retention_policy(policy)
        retrieved = governance.get_retention_policy("annotation")
        assert retrieved is not None
        assert retrieved.duration_days == 365

    def test_remove_retention_policy(self):
        """Test removing retention policy."""
        governance = DataGovernance()
        governance.add_retention_policy(
            RetentionPolicy(
                data_type="annotation",
                retention_type=RetentionType.TIME_BASED,
                duration_days=365,
            )
        )
        assert governance.remove_retention_policy("annotation") is True
        assert governance.get_retention_policy("annotation") is None

    def test_remove_retention_policy_not_found(self):
        """Test removing non-existent retention policy."""
        governance = DataGovernance()
        assert governance.remove_retention_policy("nonexistent") is False

    def test_is_expired_time_based(self):
        """Test time-based expiration."""
        governance = DataGovernance()
        governance.add_retention_policy(
            RetentionPolicy(
                data_type="annotation",
                retention_type=RetentionType.TIME_BASED,
                duration_days=1,
            )
        )
        old_time = time.time() - 2 * 86400  # 2 days ago
        assert governance.is_expired("annotation", old_time) is True

    def test_is_expired_not_expired(self):
        """Test non-expired data."""
        governance = DataGovernance()
        governance.add_retention_policy(
            RetentionPolicy(
                data_type="annotation",
                retention_type=RetentionType.TIME_BASED,
                duration_days=365,
            )
        )
        assert governance.is_expired("annotation", time.time()) is False

    def test_is_expired_indefinite(self):
        """Test indefinite retention."""
        governance = DataGovernance()
        governance.add_retention_policy(
            RetentionPolicy(
                data_type="annotation",
                retention_type=RetentionType.INDEFINITE,
            )
        )
        assert governance.is_expired("annotation", time.time()) is False

    def test_is_expired_event_based(self):
        """Test event-based retention."""
        governance = DataGovernance()
        governance.add_retention_policy(
            RetentionPolicy(
                data_type="annotation",
                retention_type=RetentionType.EVENT_BASED,
                trigger_event="user_deleted",
            )
        )
        assert governance.is_expired("annotation", time.time(), "user_deleted") is True
        assert governance.is_expired("annotation", time.time(), "other_event") is False

    def test_is_expired_no_policy(self):
        """Test expiration with no policy."""
        governance = DataGovernance()
        assert governance.is_expired("annotation", time.time()) is False

    def test_add_quality_rule(self):
        """Test adding quality rule."""
        governance = DataGovernance()
        governance.add_quality_rule(
            "annotation",
            {
                "field": "confidence",
                "condition": "gte",
                "value": 0.8,
            },
        )

    def test_validate_quality_pass(self):
        """Test quality validation passing."""
        governance = DataGovernance()
        governance.add_quality_rule(
            "annotation",
            {
                "field": "confidence",
                "condition": "gte",
                "value": 0.8,
            },
        )
        errors = governance.validate_quality("annotation", {"confidence": 0.9})
        assert len(errors) == 0

    def test_validate_quality_fail(self):
        """Test quality validation failing."""
        governance = DataGovernance()
        governance.add_quality_rule(
            "annotation",
            {
                "field": "confidence",
                "condition": "gte",
                "value": 0.8,
            },
        )
        errors = governance.validate_quality("annotation", {"confidence": 0.5})
        assert len(errors) == 1

    def test_validate_quality_missing_field(self):
        """Test quality validation with missing field."""
        governance = DataGovernance()
        governance.add_quality_rule(
            "annotation",
            {
                "field": "confidence",
                "condition": "gte",
                "value": 0.8,
            },
        )
        errors = governance.validate_quality("annotation", {})
        assert len(errors) == 1
        assert "Missing required field" in errors[0]

    def test_track_lineage(self):
        """Test tracking data lineage."""
        governance = DataGovernance()
        record = governance.track_lineage(
            record_id="data_002",
            source_id="data_001",
            operation="derived",
            user_id="user_001",
        )
        assert record.record_id == "data_002"
        assert record.source_id == "data_001"

    def test_get_lineage(self):
        """Test getting lineage records."""
        governance = DataGovernance()
        governance.track_lineage("data_002", "data_001", "derived")
        records = governance.get_lineage("data_002")
        assert len(records) == 1
        assert records[0].operation == "derived"

    def test_get_ancestors(self):
        """Test getting ancestor records."""
        governance = DataGovernance()
        governance.track_lineage("data_002", "data_001", "derived")
        governance.track_lineage("data_003", "data_002", "transformed")
        ancestors = governance.get_ancestors("data_003")
        assert len(ancestors) >= 1

    def test_record_consent(self):
        """Test recording consent."""
        governance = DataGovernance()
        governance.record_consent("user_001", "data_processing", True)
        assert governance.check_consent("user_001", "data_processing") is True

    def test_check_consent_not_given(self):
        """Test checking consent not given."""
        governance = DataGovernance()
        assert governance.check_consent("user_001", "data_processing") is False

    def test_check_consent_revoked(self):
        """Test checking revoked consent."""
        governance = DataGovernance()
        governance.record_consent("user_001", "data_processing", True)
        governance.record_consent("user_001", "data_processing", False)
        assert governance.check_consent("user_001", "data_processing") is False

    def test_request_data_deletion(self):
        """Test requesting data deletion."""
        governance = DataGovernance()
        record = governance.request_data_deletion("user_001", ["data_001", "data_002"])
        assert record["user_id"] == "user_001"
        assert record["data_ids"] == ["data_001", "data_002"]
        assert record["status"] == "requested"

    def test_export_user_data(self):
        """Test exporting user data."""
        governance = DataGovernance()
        governance.grant_access("user_001", "data_001", DataAccessLevel.RESTRICTED)
        governance.record_consent("user_001", "processing", True)
        export = governance.export_user_data("user_001")
        assert export["user_id"] == "user_001"
        assert "access_policies" in export
        assert "consent_records" in export

    def test_get_audit_log(self):
        """Test getting audit log."""
        governance = DataGovernance()
        governance.grant_access("user_001", "data_001", DataAccessLevel.RESTRICTED)
        log = governance.get_audit_log()
        assert len(log) >= 1
        assert log[0]["event_type"] == "access_granted"

    def test_get_stats(self):
        """Test getting governance statistics."""
        governance = DataGovernance()
        governance.add_retention_policy(
            RetentionPolicy(
                data_type="annotation",
                retention_type=RetentionType.TIME_BASED,
                duration_days=365,
            )
        )
        governance.grant_access("user_001", "data_001", DataAccessLevel.RESTRICTED)
        stats = governance.get_stats()
        assert stats["retention_policies"] == 1
        assert stats["access_policies"] == 1

    def test_multiple_access_policies(self):
        """Test multiple access policies for same user/data."""
        governance = DataGovernance()
        governance.grant_access("user_001", "data_001", DataAccessLevel.INTERNAL)
        governance.grant_access("user_001", "data_001", DataAccessLevel.RESTRICTED)
        # RESTRICTED is higher, so should pass
        assert governance.check_access("user_001", "data_001", DataAccessLevel.RESTRICTED) is True

    def test_get_access_policies_no_match(self):
        """Test getting access policies with no match."""
        governance = DataGovernance()
        policies = governance.get_access_policies("nonexistent")
        assert len(policies) == 0

    def test_track_lineage_no_source(self):
        """Test tracking lineage without source."""
        governance = DataGovernance()
        record = governance.track_lineage("data_001", operation="created")
        assert record.source_id is None
        assert record.operation == "created"
