"""Tests for data retention policy module.

Tests for DataRetentionPolicy, RetentionRule, RetentionRecord,
RetentionAction, and DataType.
"""

import time


from dvas.security.retention import (
    DataRetentionPolicy,
    DataType,
    RetentionAction,
    RetentionRecord,
    RetentionRule,
)


class TestRetentionAction:
    """Test RetentionAction enum."""

    def test_action_values(self):
        """Test action values."""
        assert RetentionAction.DELETE.value == "delete"
        assert RetentionAction.ARCHIVE.value == "archive"
        assert RetentionAction.ANONYMIZE.value == "anonymize"
        assert RetentionAction.NOTIFY.value == "notify"
        assert RetentionAction.MARK_FOR_REVIEW.value == "mark_for_review"


class TestDataType:
    """Test DataType enum."""

    def test_type_values(self):
        """Test data type values."""
        assert DataType.ANNOTATION.value == "annotation"
        assert DataType.VIDEO.value == "video"
        assert DataType.EXPORT.value == "export"
        assert DataType.AUDIT_LOG.value == "audit_log"
        assert DataType.USER_DATA.value == "user_data"
        assert DataType.TEMPORARY.value == "temporary"
        assert DataType.SESSION.value == "session"
        assert DataType.BACKUP.value == "backup"


class TestRetentionRule:
    """Test RetentionRule dataclass."""

    def test_rule_creation(self):
        """Test creating a retention rule."""
        rule = RetentionRule(
            data_type=DataType.ANNOTATION,
            retention_days=365,
            action=RetentionAction.ARCHIVE,
            description="Annotations retained for 1 year",
        )
        assert rule.data_type == DataType.ANNOTATION
        assert rule.retention_days == 365
        assert rule.action == RetentionAction.ARCHIVE
        assert rule.enabled is True

    def test_rule_to_dict(self):
        """Test converting rule to dict."""
        rule = RetentionRule(
            data_type=DataType.ANNOTATION,
            retention_days=365,
            action=RetentionAction.ARCHIVE,
        )
        d = rule.to_dict()
        assert d["data_type"] == "annotation"
        assert d["retention_days"] == 365
        assert d["action"] == "archive"


class TestRetentionRecord:
    """Test RetentionRecord dataclass."""

    def test_record_creation(self):
        """Test creating a retention record."""
        record = RetentionRecord(
            record_id="ann_001",
            data_type=DataType.ANNOTATION,
            created_at=time.time(),
            last_accessed=time.time(),
            owner_id="user_001",
        )
        assert record.record_id == "ann_001"
        assert record.data_type == DataType.ANNOTATION
        assert record.exempt is False

    def test_record_to_dict(self):
        """Test converting record to dict."""
        now = time.time()
        record = RetentionRecord(
            record_id="ann_001",
            data_type=DataType.ANNOTATION,
            created_at=now,
            last_accessed=now,
            owner_id="user_001",
            metadata={"key": "value"},
        )
        d = record.to_dict()
        assert d["record_id"] == "ann_001"
        assert d["data_type"] == "annotation"
        assert d["metadata"]["key"] == "value"


class TestDataRetentionPolicy:
    """Test DataRetentionPolicy class."""

    def test_init(self):
        """Test initialization."""
        policy = DataRetentionPolicy()
        assert policy is not None

    def test_init_with_custom_rules(self):
        """Test initialization with custom rules."""
        rules = [
            RetentionRule(
                data_type=DataType.ANNOTATION,
                retention_days=30,
                action=RetentionAction.DELETE,
            ),
        ]
        policy = DataRetentionPolicy(rules=rules)
        rule = policy.get_rule(DataType.ANNOTATION)
        assert rule.retention_days == 30

    def test_add_rule(self):
        """Test adding a rule."""
        policy = DataRetentionPolicy(rules=[])
        rule = RetentionRule(
            data_type=DataType.USER_DATA,
            retention_days=7,
            action=RetentionAction.DELETE,
        )
        policy.add_rule(rule)
        assert policy.get_rule(DataType.USER_DATA) is not None

    def test_remove_rule(self):
        """Test removing a rule."""
        policy = DataRetentionPolicy()
        assert policy.remove_rule(DataType.ANNOTATION) is True
        assert policy.get_rule(DataType.ANNOTATION) is None
        assert policy.remove_rule(DataType.ANNOTATION) is False

    def test_get_rule_not_found(self):
        """Test getting non-existent rule."""
        policy = DataRetentionPolicy(rules=[])
        assert policy.get_rule(DataType.ANNOTATION) is None

    def test_track_record(self):
        """Test tracking a record."""
        policy = DataRetentionPolicy()
        record = policy.track_record(
            record_id="ann_001",
            data_type=DataType.ANNOTATION,
            owner_id="user_001",
        )
        assert record.record_id == "ann_001"
        assert record.data_type == DataType.ANNOTATION

    def test_track_record_with_metadata(self):
        """Test tracking record with metadata."""
        policy = DataRetentionPolicy()
        record = policy.track_record(
            record_id="ann_001",
            data_type=DataType.ANNOTATION,
            owner_id="user_001",
            metadata={"project": "test"},
        )
        assert record.metadata["project"] == "test"

    def test_untrack_record(self):
        """Test untracking a record."""
        policy = DataRetentionPolicy()
        policy.track_record("ann_001", DataType.ANNOTATION, "user_001")
        assert policy.untrack_record("ann_001") is True
        assert policy.untrack_record("ann_001") is False

    def test_update_access_time(self):
        """Test updating access time."""
        policy = DataRetentionPolicy()
        policy.track_record("ann_001", DataType.ANNOTATION, "user_001")
        old_time = policy._records["ann_001"].last_accessed
        time.sleep(0.01)
        assert policy.update_access_time("ann_001") is True
        assert policy._records["ann_001"].last_accessed > old_time

    def test_update_access_time_not_found(self):
        """Test updating access time for non-existent record."""
        policy = DataRetentionPolicy()
        assert policy.update_access_time("nonexistent") is False

    def test_get_expired_records(self):
        """Test getting expired records."""
        policy = DataRetentionPolicy()
        # Create a record that is already expired (created 400 days ago)
        old_time = time.time() - 400 * 86400
        policy.track_record(
            "ann_001",
            DataType.ANNOTATION,
            "user_001",
            created_at=old_time,
        )
        expired = policy.get_expired_records()
        assert len(expired) == 1
        assert expired[0].record_id == "ann_001"

    def test_get_expired_records_not_expired(self):
        """Test getting expired records when none are expired."""
        policy = DataRetentionPolicy()
        policy.track_record("ann_001", DataType.ANNOTATION, "user_001")
        expired = policy.get_expired_records()
        assert len(expired) == 0

    def test_get_expired_records_exempt(self):
        """Test that exempt records are not returned."""
        policy = DataRetentionPolicy()
        old_time = time.time() - 400 * 86400
        policy.track_record(
            "ann_001",
            DataType.ANNOTATION,
            "user_001",
            created_at=old_time,
        )
        policy.set_exemption("ann_001", True)
        expired = policy.get_expired_records()
        assert len(expired) == 0

    def test_get_expired_records_disabled_rule(self):
        """Test that disabled rules don't trigger expiration."""
        policy = DataRetentionPolicy()
        old_time = time.time() - 400 * 86400
        policy.track_record(
            "ann_001",
            DataType.ANNOTATION,
            "user_001",
            created_at=old_time,
        )
        rule = policy.get_rule(DataType.ANNOTATION)
        if rule:
            rule.enabled = False
        expired = policy.get_expired_records()
        assert len(expired) == 0

    def test_apply_retention(self):
        """Test applying retention."""
        policy = DataRetentionPolicy()
        policy.track_record("ann_001", DataType.ANNOTATION, "user_001")

        handler_called = []

        def handler(record):
            handler_called.append(record.record_id)

        policy.register_handler(RetentionAction.ARCHIVE, handler)
        assert policy.apply_retention("ann_001") is True
        assert "ann_001" in handler_called

    def test_apply_retention_not_found(self):
        """Test applying retention to non-existent record."""
        policy = DataRetentionPolicy()
        assert policy.apply_retention("nonexistent") is False

    def test_apply_retention_no_rule(self):
        """Test applying retention with no rule."""
        policy = DataRetentionPolicy(rules=[])
        policy.track_record("ann_001", DataType.ANNOTATION, "user_001")
        assert policy.apply_retention("ann_001") is False

    def test_register_handler(self):
        """Test registering a handler."""
        policy = DataRetentionPolicy()

        def handler(record):
            pass

        policy.register_handler(RetentionAction.DELETE, handler)
        assert len(policy._handlers[RetentionAction.DELETE]) == 1

    def test_set_retention_override(self):
        """Test setting retention override."""
        policy = DataRetentionPolicy()
        policy.track_record("ann_001", DataType.ANNOTATION, "user_001")
        assert policy.set_retention_override("ann_001", 7) is True
        assert policy._records["ann_001"].retention_override_days == 7

    def test_set_retention_override_not_found(self):
        """Test setting override for non-existent record."""
        policy = DataRetentionPolicy()
        assert policy.set_retention_override("nonexistent", 7) is False

    def test_set_exemption(self):
        """Test setting exemption."""
        policy = DataRetentionPolicy()
        policy.track_record("ann_001", DataType.ANNOTATION, "user_001")
        assert policy.set_exemption("ann_001", True) is True
        assert policy._records["ann_001"].exempt is True

    def test_set_exemption_not_found(self):
        """Test setting exemption for non-existent record."""
        policy = DataRetentionPolicy()
        assert policy.set_exemption("nonexistent", True) is False

    def test_get_record_age(self):
        """Test getting record age."""
        policy = DataRetentionPolicy()
        old_time = time.time() - 10 * 86400  # 10 days ago
        policy.track_record(
            "ann_001",
            DataType.ANNOTATION,
            "user_001",
            created_at=old_time,
        )
        age = policy.get_record_age("ann_001")
        assert age is not None
        assert age >= 10

    def test_get_record_age_not_found(self):
        """Test getting age of non-existent record."""
        policy = DataRetentionPolicy()
        assert policy.get_record_age("nonexistent") is None

    def test_get_records_by_type(self):
        """Test getting records by type."""
        policy = DataRetentionPolicy()
        policy.track_record("ann_001", DataType.ANNOTATION, "user_001")
        policy.track_record("vid_001", DataType.VIDEO, "user_001")
        policy.track_record("ann_002", DataType.ANNOTATION, "user_002")

        annotations = policy.get_records_by_type(DataType.ANNOTATION)
        assert len(annotations) == 2

    def test_get_stats(self):
        """Test getting statistics."""
        policy = DataRetentionPolicy()
        policy.track_record("ann_001", DataType.ANNOTATION, "user_001")
        policy.track_record("vid_001", DataType.VIDEO, "user_001")

        stats = policy.get_stats()
        assert stats["total_records"] == 2
        assert stats["records_by_type"]["annotation"] == 1
        assert stats["records_by_type"]["video"] == 1
        assert stats["expired_records"] == 0

    def test_list_rules(self):
        """Test listing rules."""
        policy = DataRetentionPolicy()
        rules = policy.list_rules()
        assert len(rules) > 0
        data_types = [r.data_type for r in rules]
        assert DataType.ANNOTATION in data_types
        assert DataType.VIDEO in data_types

    def test_handler_error_handling(self):
        """Test that handler errors don't crash the system."""
        policy = DataRetentionPolicy()
        policy.track_record("ann_001", DataType.ANNOTATION, "user_001")

        def failing_handler(record):
            raise RuntimeError("Handler error")

        policy.register_handler(RetentionAction.ARCHIVE, failing_handler)
        # Should not raise
        assert policy.apply_retention("ann_001") is True


class TestDataRetentionPolicyEdgeCases:
    """Test edge cases for DataRetentionPolicy."""

    def test_empty_policy(self):
        """Test with empty policy."""
        policy = DataRetentionPolicy(rules=[])
        assert len(policy.list_rules()) == 0

    def test_record_without_rule(self):
        """Test tracking record with no matching rule."""
        policy = DataRetentionPolicy(rules=[])
        policy.track_record("ann_001", DataType.ANNOTATION, "user_001")
        expired = policy.get_expired_records()
        assert len(expired) == 0

    def test_negative_retention_days(self):
        """Test with negative retention days."""
        policy = DataRetentionPolicy(rules=[])
        rule = RetentionRule(
            data_type=DataType.TEMPORARY,
            retention_days=-1,
            action=RetentionAction.DELETE,
        )
        policy.add_rule(rule)
        policy.track_record("tmp_001", DataType.TEMPORARY, "user_001")
        expired = policy.get_expired_records()
        # Negative retention means always expired
        assert len(expired) == 1

    def test_zero_retention_days(self):
        """Test with zero retention days."""
        policy = DataRetentionPolicy(rules=[])
        rule = RetentionRule(
            data_type=DataType.TEMPORARY,
            retention_days=0,
            action=RetentionAction.DELETE,
        )
        policy.add_rule(rule)
        policy.track_record("tmp_001", DataType.TEMPORARY, "user_001")
        expired = policy.get_expired_records()
        # Zero retention means immediately expired
        assert len(expired) == 1

    def test_override_shorter_than_rule(self):
        """Test override shorter than rule."""
        policy = DataRetentionPolicy()
        old_time = time.time() - 200 * 86400
        policy.track_record(
            "ann_001",
            DataType.ANNOTATION,
            "user_001",
            created_at=old_time,
        )
        # Override to 100 days - still expired (200 > 100)
        policy.set_retention_override("ann_001", 100)
        expired = policy.get_expired_records()
        assert len(expired) == 1

    def test_override_longer_than_rule(self):
        """Test override longer than rule."""
        policy = DataRetentionPolicy()
        old_time = time.time() - 200 * 86400
        policy.track_record(
            "ann_001",
            DataType.ANNOTATION,
            "user_001",
            created_at=old_time,
        )
        # Override to 300 days - not expired (200 < 300)
        policy.set_retention_override("ann_001", 300)
        expired = policy.get_expired_records()
        assert len(expired) == 0
