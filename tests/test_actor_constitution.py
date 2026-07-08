"""Tests for build_actor_system_prompt — actor constitution enforcement."""
from dcc_middleware import MemoryCapsule, CapsuleMetadata
from dcc_app import build_actor_system_prompt


def _make_capsule(global_context="", key_decisions=None,
                  current_state="", frame=0):
    return MemoryCapsule(
        topic="test",
        global_context=global_context,
        key_decisions=key_decisions or [],
        current_state=current_state,
        metadata=CapsuleMetadata(last_updated_frame=frame),
    )


def test_empty_capsule_no_constitution():
    """Capsule rỗng → prompt không có HIẾN PHÁP."""
    cap = _make_capsule()  # global_context="", key_decisions=[]
    prompt = build_actor_system_prompt(cap)
    assert "HIẾN PHÁP" not in prompt
    assert "trợ lý AI thông thường" in prompt
    assert "KHẾ ƯỚC NGÔN NGỮ" in prompt


def test_empty_capsule_default_message():
    """Capsule với global_context mặc định → vẫn là empty."""
    cap = _make_capsule(global_context="New topic. No prior history.")
    prompt = build_actor_system_prompt(cap)
    assert "HIẾN PHÁP" not in prompt
    assert "trợ lý AI" in prompt


def test_active_capsule_contains_constitution():
    """Capsule có dữ liệu → prompt chứa HIẾN PHÁP."""
    cap = _make_capsule(
        global_context="Dự án realtime parking",
        key_decisions=["PostGIS làm database không gian"],
        current_state="Đang kiểm thử hiệu năng truy vấn",
        frame=3,
    )
    prompt = build_actor_system_prompt(cap)
    assert "HIẾN PHÁP" in prompt
    assert "kiến trúc sư của dự án" in prompt
    assert "BẢO VỆ các quyết định" in prompt


def test_active_capsule_contains_capsule_json():
    """Prompt chứa capsule JSON giữa thẻ <capsule>."""
    cap = _make_capsule(
        global_context="Dự án parking",
        key_decisions=["PostGIS"],
        current_state="Testing",
        frame=2,
    )
    prompt = build_actor_system_prompt(cap)
    assert "<capsule>" in prompt
    assert "</capsule>" in prompt
    assert "Dự án parking" in prompt
    assert "PostGIS" in prompt


def test_rules_placed_after_capsule():
    """Luật THI HÀNH nằm SAU capsule — model chú ý phần cuối."""
    cap = _make_capsule(
        global_context="Dự án parking",
        key_decisions=["PostGIS"],
        current_state="Testing",
    )
    prompt = build_actor_system_prompt(cap)
    capsule_pos = prompt.index("<capsule>")
    rules_pos = prompt.index("LUẬT THI HÀNH")
    assert rules_pos > capsule_pos, "Rules must be AFTER capsule"


def test_contains_anti_map_rule():
    """Prompt chứa luật ⛔1 về anti-map."""
    cap = _make_capsule(
        global_context="Dự án parking. Ngoài phạm vi: thanh toán.",
        key_decisions=["PostGIS"],
        current_state="Testing",
    )
    prompt = build_actor_system_prompt(cap)
    assert "⛔ 1." in prompt
    assert "Ngoài phạm vi" in prompt or "anti-map" in prompt


def test_contains_red_decision_rule():
    """Prompt chứa luật ⛔2 về Reversibility 🔴."""
    cap = _make_capsule(
        global_context="Dự án parking",
        key_decisions=["WebSockets (Reversibility: 🔴, Switch: khi có 1M users)"],
        current_state="Testing",
    )
    prompt = build_actor_system_prompt(cap)
    assert "⛔ 2." in prompt
    assert "🔴" in prompt
    assert "switch_trigger" in prompt


def test_contains_no_fake_memory_rule():
    """Prompt chứa luật ⛔3 cấm fake memory."""
    cap = _make_capsule(
        global_context="Dự án",
        key_decisions=["Decision"],
        current_state="Active",
    )
    prompt = build_actor_system_prompt(cap)
    assert "⛔ 3." in prompt
    assert "[SYSTEM:" in prompt or "đã ghi vào bộ nhớ" in prompt


def test_contains_no_cheerlead_rule():
    """Prompt chứa luật ⛔4 cấm khen trước khi đối chiếu."""
    cap = _make_capsule(
        global_context="Dự án",
        key_decisions=["Decision"],
        current_state="Active",
    )
    prompt = build_actor_system_prompt(cap)
    assert "⛔ 4." in prompt
    assert "hợp lý" in prompt or "rất hay" in prompt


def test_contains_language_contract():
    """Prompt chứa khế ước ngôn ngữ (cả head lẫn tail)."""
    cap = _make_capsule(
        global_context="Dự án",
        key_decisions=["Decision"],
        current_state="Active",
    )
    prompt = build_actor_system_prompt(cap)
    assert "KHẾ ƯỚC NGÔN NGỮ" in prompt
    assert "Edge Computing" in prompt  # example in template
    assert "NHẮC LẠI" in prompt  # tail marker
