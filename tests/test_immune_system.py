"""Tests for Compactor immune system — write zones, language grid, SYSTEM stripping."""

from dcc_middleware import (
    ContextCompactorMiddleware,
    MemoryCapsule,
    CapsuleMetadata,
)


def _make_capsule(global_context="", key_decisions=None,
                  current_state="", frame=0):
    return MemoryCapsule(
        topic="test",
        global_context=global_context,
        key_decisions=key_decisions or [],
        current_state=current_state,
        metadata=CapsuleMetadata(last_updated_frame=frame),
    )


# ---------------------------------------------------------------------------
# _has_cjk — Language grid
# ---------------------------------------------------------------------------

def test_has_cjk_empty():
    """Chuỗi rỗng → False."""
    assert ContextCompactorMiddleware._has_cjk("") is False


def test_has_cjk_english():
    """ASCII thuần → False."""
    assert ContextCompactorMiddleware._has_cjk("Hello, this is English.") is False
    assert ContextCompactorMiddleware._has_cjk("JSON output with { braces }") is False


def test_has_cjk_chinese():
    """Tiếng Trung (CJK Unified Ideographs) → True."""
    assert ContextCompactorMiddleware._has_cjk("这是一个测试") is True
    assert ContextCompactorMiddleware._has_cjk("混合English") is True


def test_has_cjk_japanese_kanji():
    """Kanji (dùng CJK Unified Ideographs) → True."""
    assert ContextCompactorMiddleware._has_cjk("日本語の漢字") is True


def test_has_cjk_korean_hanja():
    """Hanja/Hangul → True (Hangul is 0xAC00–0xD7AF, not covered yet)."""
    # Most Korean hanja are CJK Unified Ideographs
    assert ContextCompactorMiddleware._has_cjk("韓國語") is True


# ---------------------------------------------------------------------------
# _strip_system_blocks — SYSTEM block sanitizer
# ---------------------------------------------------------------------------

def test_strip_system_blocks_none():
    """Không có SYSTEM block → giữ nguyên."""
    result = ContextCompactorMiddleware._strip_system_blocks(
        "Bình thường không có block"
    )
    assert result == "Bình thường không có block"


def test_strip_system_blocks_simple():
    """[SYSTEM: ...] simple → bị xoá."""
    result = ContextCompactorMiddleware._strip_system_blocks(
        'Trước [SYSTEM: GHI NHẬN QUYẾT ĐỊNH ID 9] Sau'
    )
    assert "SYSTEM:" not in result
    assert result.strip() == "Trước  Sau"


def test_strip_system_blocks_multiline():
    """[SYSTEM: ...] nhiều dòng → bị xoá."""
    result = ContextCompactorMiddleware._strip_system_blocks(
        "Đầu\n[SYSTEM: GHI NHẬN\nQUYẾT ĐỊNH\nID 9]\nCuối"
    )
    assert "SYSTEM:" not in result
    assert "Đầu" in result
    assert "Cuối" in result


def test_strip_system_blocks_multiple():
    """Nhiều SYSTEM block → tất cả bị xoá."""
    result = ContextCompactorMiddleware._strip_system_blocks(
        "[SYSTEM: một] giữa [SYSTEM: hai] cuối"
    )
    assert "SYSTEM:" not in result
    assert result.strip() == "giữa  cuối"


# ---------------------------------------------------------------------------
# _enforce_write_zones — LOCKED + GUARDED + FLUID
# ---------------------------------------------------------------------------

def test_write_zones_locked():
    """LOCKED: global_context không đổi dù compactor đưa cái khác."""
    old = _make_capsule(
        global_context="Dự án parking — KHÔNG làm thanh toán.",
        key_decisions=["PostGIS làm database"],
        current_state="Testing",
        frame=3,
    )
    incoming = _make_capsule(
        global_context="Dự án parking — đã tích hợp thanh toán.",  # bỊA
        key_decisions=["PostGIS làm database", "Đã tích hợp Stripe"],  # bỊA
        current_state="Đã deploy",
        frame=4,
    )
    result = ContextCompactorMiddleware._enforce_write_zones(old, incoming)
    # LOCKED: phải giữ nguyên của old
    assert "KHÔNG làm thanh toán" in result.global_context
    assert "Đã tích hợp thanh toán" not in result.global_context


def test_write_zones_guarded_append_only():
    """GUARDED: key_decisions cũ được giữ, mới được thêm vào."""
    old = _make_capsule(
        global_context="Dự án parking",
        key_decisions=["PostGIS làm database", "WebSockets realtime"],
        current_state="Active",
        frame=2,
    )
    incoming = _make_capsule(
        global_context="Dự án parking — bỏ WebSockets",  # sẽ bị LOCKED ghi đè
        key_decisions=["Chỉ dùng SSE, bỏ WebSockets"],  # xoá mất quyết định cũ
        current_state="Sửa kiến trúc",
        frame=3,
    )
    result = ContextCompactorMiddleware._enforce_write_zones(old, incoming)
    # GUARDED: quyết định cũ vẫn còn
    assert "WebSockets realtime" in result.key_decisions
    # Quyết định mới được thêm vào (exact match)
    assert "Chỉ dùng SSE, bỏ WebSockets" in result.key_decisions
    # Có 3 quyết định: 2 cũ + 1 mới
    assert len(result.key_decisions) == 3


def test_write_zones_guarded_no_duplicates():
    """GUARDED: không thêm duplicate decision."""
    old = _make_capsule(
        global_context="Dự án",
        key_decisions=["PostGIS làm database"],
        current_state="Active",
    )
    incoming = _make_capsule(
        global_context="Dự án khác",
        key_decisions=["PostGIS làm database"],  # duplicate
        current_state="Khác",
    )
    result = ContextCompactorMiddleware._enforce_write_zones(old, incoming)
    assert len(result.key_decisions) == 1
    assert result.key_decisions == ["PostGIS làm database"]


def test_write_zones_fluid_current_state():
    """FLUID: current_state được cập nhật tự do."""
    old = _make_capsule(
        global_context="Dự án",
        key_decisions=["Decision"],
        current_state="Testing",
        frame=1,
    )
    incoming = _make_capsule(
        global_context="Dự án — thay đổi",
        key_decisions=["Decision", "New"],
        current_state="Đã deploy lên production",
        frame=2,
    )
    result = ContextCompactorMiddleware._enforce_write_zones(old, incoming)
    # FLUID: current_state lấy từ incoming
    assert "Đã deploy lên production" in result.current_state


# ---------------------------------------------------------------------------
# _fail_safe_merge — keep old capsule on reject
# ---------------------------------------------------------------------------

def test_fail_safe_merge_preserves_decisions():
    """Khi compactor bị reject, quyết định cũ được giữ."""
    old = _make_capsule(
        global_context="Dự án parking — KHÔNG thanh toán.",
        key_decisions=["PostGIS", "WebSockets"],
        current_state="Testing Phase",
        frame=5,
    )
    result = ContextCompactorMiddleware()._fail_safe_merge(old, "Thêm payment")
    assert "PostGIS" in result.key_decisions
    assert "WebSockets" in result.key_decisions
    assert "KHÔNG thanh toán" in result.global_context
    assert "auto-note" in result.current_state
    assert result.metadata.last_updated_frame == 6
