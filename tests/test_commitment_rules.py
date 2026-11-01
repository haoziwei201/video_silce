"""R2 / R3：承诺卡的程序校验、状态判定、来源门禁、冲突保留、待确认分流（不调 LLM）。"""
from _common import ROOT
from src import commitment as cm


def _ev(eid, status="已读公开页面并整理摘要；未由小组或厂商确认", reviewer="", kind="doc"):
    return {"evidence_id": eid, "kind": kind, "text": "…", "status": status, "human_reviewer": reviewer}


EVS = [_ev("A"), _ev("B"), _ev("S", status="历史资料，已失效")]


def test_hallucinated_citation_is_dropped_but_valid_kept():
    r = cm.verify_and_judge({"supported_statement": "提供 API 接口", "conditions": ["需匹配接口协议"],
                             "unknowns": ["客户系统的接口文档"], "cited_evidence_ids": ["A", "GHOST"], "conflicts": []}, EVS)
    assert r["status"] == cm.STATUS_CONDITIONAL
    assert r["valid_evidence_ids"] == ["A"] and r["dropped_evidence_ids"] == ["GHOST"]
    assert r["source_review"] == cm.REVIEW_PENDING


def test_only_stale_source_means_needs_tech_and_statement_cleared():
    r = cm.verify_and_judge({"supported_statement": "两周上线", "conditions": [], "unknowns": [],
                             "cited_evidence_ids": ["S"], "conflicts": []}, EVS)
    assert r["status"] == cm.STATUS_NEEDS_TECH and r["supported_statement"] == ""
    assert r["stale_evidence_ids"] == ["S"]
    assert any("两周上线" in u for u in r["unknowns"]) and any("失效" in u for u in r["unknowns"])


def test_no_citation_means_needs_tech():
    r = cm.verify_and_judge({"supported_statement": "", "conditions": [], "unknowns": ["实施周期"],
                             "cited_evidence_ids": [], "conflicts": []}, EVS)
    assert r["status"] == cm.STATUS_NEEDS_TECH


def test_supported_without_conditions():
    r = cm.verify_and_judge({"supported_statement": "提供 SDK", "conditions": [], "unknowns": [],
                             "cited_evidence_ids": ["A"], "conflicts": []}, EVS)
    assert r["status"] == cm.STATUS_SUPPORTED


def test_conflict_with_valid_ids_is_preserved_not_resolved():
    r = cm.verify_and_judge({"supported_statement": "支持扫描件", "conditions": [], "unknowns": [],
                             "cited_evidence_ids": ["A", "B"],
                             "conflicts": [{"evidence_ids": ["A", "B"], "note": "格式支持矛盾"}]}, EVS)
    assert r["status"] == cm.STATUS_CONFLICT and r["conflicts"][0]["note"] == "格式支持矛盾"


def test_conflict_with_unknown_id_is_ignored():
    r = cm.verify_and_judge({"supported_statement": "x", "conditions": [], "unknowns": [],
                             "cited_evidence_ids": ["A"], "conflicts": [{"evidence_ids": ["A", "NOPE"], "note": ""}]}, EVS)
    assert r["status"] == cm.STATUS_SUPPORTED and r["conflicts"] == []


def test_reviewed_source_marks_verified():
    r = cm.verify_and_judge({"supported_statement": "x", "conditions": [], "unknowns": [],
                             "cited_evidence_ids": ["R"], "conflicts": []}, [_ev("R", reviewer="李姐")])
    assert r["source_review"] == cm.REVIEW_VERIFIED


def test_route_customer_vs_tech():
    ro = cm.route_unknowns(["客户系统的接口文档与数据格式", "实施周期需要评估", "该版本是否支持扫描件", "贵方期望的上线时间"])
    assert [q["item"] for q in ro["ask_customer"]] == ["客户系统的接口文档与数据格式", "贵方期望的上线时间"]
    assert [q["suggested_role"] for q in ro["ask_tech"]] == ["售前/实施", "产品"]
    assert all(q["question"].startswith("请贵方补充") for q in ro["ask_customer"])


def test_heuristic_decompose_splits_demo_question():
    cl = cm._heuristic_decompose("你们能不能直接接入我们现有的业务系统，马上上线，以后扫描件都不用人工看？")
    asks = [c["customer_ask"] for c in cl]
    assert len(asks) == 3, asks
    assert cl[0]["claim_id"] == "C1"


def test_fallback_reply_never_promises_without_evidence():
    cards = [{"customer_ask": "马上上线", "supported_statement": "", "conditions": [], "ask_customer": []}]
    text = cm._fallback_reply(cards)
    assert "不足以直接承诺" in text


def test_build_card_only_includes_valid_evidence():
    claim = {"claim_id": "C1", "customer_ask": "能接入吗", "topic": "对接", "given_conditions": [], "search_query": "接口"}
    judged = cm.verify_and_judge({"supported_statement": "有 API", "conditions": ["需匹配协议"], "unknowns": ["客户系统接口"],
                                  "cited_evidence_ids": ["A", "GHOST"], "conflicts": []}, EVS)
    card = cm.build_card(claim, judged, cm.route_unknowns(judged["unknowns"]), EVS)
    assert [e["evidence_id"] for e in card["evidence"]] == ["A"]
    assert card["candidate_evidence_ids"] == ["A", "B", "S"] and card["dropped_evidence_ids"] == ["GHOST"]
    assert card["ask_customer"] and not card["ask_tech"]


def test_llm_json_parser_handles_fences_and_noise():
    assert cm.LLMClient._parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert cm.LLMClient._parse_json('前言 {"b": [1,2]} 后记') == {"b": [1, 2]}
    assert cm.LLMClient._parse_json("not json") is None


def run():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  ✓ {name}")


if __name__ == "__main__":
    run()
