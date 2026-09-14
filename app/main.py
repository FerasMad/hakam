"""Hakam demo: vision decision -> contract -> Laws of the Game -> Arabic explanation.

    streamlit run app/main.py

The language step uses src.llm.generate.explain when it exists (Anas's module).
Until then, or if the API call fails on stage, a deterministic Arabic template
built from the contract is shown instead and labelled as such - the demo never
breaks, and it never invents anything the contract does not contain.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ABSTAIN_MESSAGE_AR, CONFIDENCE_THRESHOLD
from src.contract import HakamContract, mock_contract
from src.llm import lexicon

LABEL_AR = {
    "offence": "مخالفة", "no_offence": "لا توجد مخالفة",
    "card": "تستوجب بطاقة", "no_card": "لا تستوجب بطاقة",
    "yellow": "إنذار", "red": "طرد",
    "tackling": "تدخل", "standing tackling": "تدخل من وضع الوقوف", "high leg": "رفع القدم عالياً",
    "holding": "إعاقة", "pushing": "دفع", "elbowing": "ضرب بالمرفق", "challenge": "التحام",
    "dive": "سقوط تمثيلي", "dont know": "غير محدد",
    "upper_body": "الجزء العلوي من الجسم", "under_body": "الجزء السفلي من الجسم",
    "with_contact": "بتماس مباشر", "without_contact": "بدون تماس",
    "yes": "نعم", "no": "لا",
}

SCENARIOS = {
    "Foul, card (confident)": lambda: mock_contract(colour=None),
    "Foul, no card": lambda: mock_contract(card="no_card", colour=None),
    "No offence": lambda: mock_contract(offence="no_offence", card=None, colour=None),
    "Low confidence -> refer": lambda: mock_contract(confidence=0.45, colour=None),
}


def ar(label: str) -> str:
    return LABEL_AR.get(str(label).lower(), str(label))


def template_explanation(c: HakamContract) -> str:
    """Grounded fallback: every phrase comes from a contract field."""
    if c.should_abstain():
        return ABSTAIN_MESSAGE_AR
    weak = set(c.low_confidence_fields())

    def hedge(field: str, text: str) -> str:
        return f"على الأرجح {text}" if field in weak else text

    decision = ar(c.offence.label)
    if c.card is not None:
        decision += " — " + hedge("card", ar(c.card.label))

    details = []
    a = c.attributes
    if "action_class" in a:
        details.append(hedge("action_class", f"نوع اللعب: {ar(a['action_class'].label)}"))
    if "contact" in a:
        details.append(hedge("contact", ar(a["contact"].label)))
    if "body_part" in a:
        details.append(hedge("body_part", f"من {ar(a['body_part'].label)}"))

    return (
        f"القرار: {decision}\n"
        f"المادة: القانون 12 — الأخطاء وسوء السلوك\n"
        f"التفسير: {'، '.join(details) if details else 'لا توجد تفاصيل إضافية'}.\n"
        f"مستوى الثقة: {c.offence.confidence:.0%}"
    )


def explain(c: HakamContract) -> tuple[str, list[dict], str]:
    try:
        from src.llm.generate import explain as llm_explain

        out = llm_explain(c)
        return out.text_ar, out.articles, "LLM + retrieval"
    except Exception as exc:  # module missing, no API key, network down
        return template_explanation(c), [], f"template fallback ({type(exc).__name__})"


def faithfulness(text: str, c: HakamContract) -> tuple[list[str], list[str]]:
    allowed = {lexicon._key(v) for v in c.citable_values()}
    if c.should_abstain():
        allowed.add("refer")
    claimed = lexicon.found_labels(text)
    return sorted(claimed & allowed), sorted(claimed - allowed)


# ---------------------------------------------------------------------------

st.set_page_config(page_title="Hakam - VAR explainer", layout="wide")
st.title("حكم — Hakam")
st.caption("Vision model decides. The language model only sees the contract, never the video.")

CONTRACTS = Path(__file__).resolve().parent.parent / "artifacts" / "contracts"

with st.sidebar:
    st.header("Input")
    sources = ["Scenario", "Upload contract JSON"]
    if (CONTRACTS / "test_index.json").exists():
        sources.insert(0, "Test set (real model)")
    source = st.radio("Contract source", sources)
    if source == "Test set (real model)":
        index = {i["action_id"]: i for i in json.loads((CONTRACTS / "test_index.json").read_text())}
        aid = st.selectbox("Test action", list(index),
                           format_func=lambda a: f"{a} - {index[a]['offence']} / {index[a]['card'] or '-'}")
        contract = HakamContract.from_dict(
            json.loads((CONTRACTS / "test" / f"{aid}.json").read_text(encoding="utf-8")))
        st.caption(f"Referee label (not shown to the LLM): {index[aid]['truth']}")
    elif source == "Scenario":
        contract = SCENARIOS[st.selectbox("Scenario", list(SCENARIOS))]()
    else:
        up = st.file_uploader("contract.json", type="json")
        contract = HakamContract.from_dict(json.load(up)) if up else mock_contract(colour=None)
    clip = st.file_uploader("Clip (optional, shown locally only)", type=["mp4"])

left, right = st.columns([1, 1])

with left:
    if clip:
        st.video(clip)
    st.subheader("1. Vision decision")
    if contract.should_abstain():
        band = "refer to human"
    elif contract.card and contract.card.label == "card":
        band = "flag"
    else:
        band = "clear"
    st.metric("Offence", ar(contract.offence.label), f"{contract.offence.confidence:.0%} confidence")
    if contract.card:
        st.metric("Card", ar(contract.card.label), f"{contract.card.confidence:.0%} confidence")
    st.info(f"Action: **{band}** (threshold {CONFIDENCE_THRESHOLD:.0%})")

    st.subheader("2. Contract — the only thing the LLM sees")
    st.code(contract.to_json(), language="json")

with right:
    st.subheader("3. Arabic explanation")
    text, articles, engine = explain(contract)
    st.markdown(
        f"<div dir='rtl' style='font-size:1.15rem;white-space:pre-wrap'>{text}</div>",
        unsafe_allow_html=True,
    )
    st.caption(f"Engine: {engine}")

    if articles:
        st.subheader("4. Retrieved Laws")
        for art in articles:
            with st.expander(f"{art.get('law', '')} — {art.get('section', '')}"):
                st.markdown(f"<div dir='rtl'>{art.get('text_ar', '')}</div>", unsafe_allow_html=True)

    st.subheader("5. Faithfulness check")
    ok, bad = faithfulness(text, contract)
    st.write(f"Supported claims: {', '.join(ok) or '—'}")
    if bad:
        st.error(f"Unsupported claims: {', '.join(bad)}")
    else:
        st.success("Every label claimed is in the contract.")
