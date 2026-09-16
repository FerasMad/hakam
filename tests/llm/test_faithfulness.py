from src.contract import mock_contract
from src.llm.faithfulness import score


def test_invented_red_card_is_unsupported_when_colour_is_null():
    result = score(
        "القرار: مخالفة وبطاقة حمراء. المادة: القانون 12. التفسير: تدخل مع تلامس.",
        mock_contract(colour=None),
        articles=[{"law": "Law 12", "section": "Fouls"}],
    )

    assert "red" in result["unsupported"]
    assert result["faithfulness"] < 1.0


def test_contract_only_claims_are_fully_supported():
    result = score(
        (
            "القرار: مخالفة — تستوجب بطاقة — لون البطاقة غير محدد. "
            "المادة: القانون 12. التفسير: تدخل مع تلامس."
        ),
        mock_contract(colour=None),
        articles=[{"law": "Law 12", "section": "Fouls"}],
    )

    assert result["unsupported"] == []
    assert result["faithfulness"] == 1.0
    assert result["cites_article"] is True


def test_conditional_card_colour_is_not_counted_as_incident_claim():
    result = score(
        (
            "القرار: مخالفة — تستوجب بطاقة — لون البطاقة غير محدد. "
            "قد تكون البطاقة حمراء إذا ثبت استخدام القوة المفرطة، لكن القرار لم يحدد ذلك."
        ),
        mock_contract(colour=None),
        [],
    )

    assert "red" not in result["unsupported"]


def test_low_confidence_field_requires_hedging():
    contract = mock_contract()
    contract.attributes["action_class"].confidence = 0.42

    hedged = score("على الأرجح كان الفعل تدخلاً مع تلامس.", contract, [])
    asserted = score("كان الفعل تدخلاً مع تلامس.", contract, [])

    assert hedged["hedged_low_conf"] is True
    assert asserted["hedged_low_conf"] is False


def test_negated_offence_word_does_not_contradict_no_offence_decision():
    contract = mock_contract(offence="no_offence", card=None, colour=None)
    result = score(
        (
            "القرار: لا توجد مخالفة\n"
            "المادة: القانون 12 — المنافسة القانونية\n"
            "التفسير: اللعب سليم ولا يحقق معايير مخالفة.\n"
            "مستوى الثقة: ثقة قرار المخالفة: 88%"
        ),
        contract,
        [{"law": "Law 12", "section": "Foul criteria"}],
    )

    assert "offence" not in result["unsupported"]
    assert result["faithfulness"] == 1.0


def test_attached_text_and_generic_intervention_are_not_action_claims():
    contract = mock_contract(colour=None)
    contract.attributes["action_class"].label = "high leg"
    result = score(
        "المادة: القانون 12. يشرح النص المرفق معيار التدخل القانوني.",
        contract,
        [{"law": "Law 12", "section": "Foul criteria"}],
    )

    assert "elbowing" not in result["claimed_labels"]
    assert "tackling" not in result["claimed_labels"]


def test_generic_tackle_word_is_supported_for_standing_tackle_contract():
    contract = mock_contract(colour=None)
    contract.attributes["action_class"].label = "standing tackling"
    result = score(
        (
            "القرار: مخالفة — تستوجب بطاقة — لون البطاقة غير محدد\n"
            "لماذا تُعد مخالفة: صُنّف الفعل تدخل من وضع الوقوف، وإذا اعتُبر "
            "التدخل متهوراً فتستوجب الواقعة إنذاراً.\n"
            "المادة: القانون 12"
        ),
        contract,
        [{"law": "Law 12", "section": "Direct free kick"}],
    )

    assert result["unsupported"] == []
    assert result["faithfulness"] == 1.0


def test_law_title_does_not_break_low_confidence_hedging():
    contract = mock_contract(colour=None)
    contract.attributes["action_class"].label = "elbowing"
    contract.attributes["action_class"].confidence = 0.48
    result = score(
        (
            "القرار: مخالفة — تستوجب بطاقة — لون البطاقة غير محدد\n"
            "المادة: القانون 12 — استخدام المرفق\n"
            "التفسير: توضح المادة معيار القانون دون وصف الفعل.\n"
            "مستوى الثقة: مواضع التحفظ: تصنيف الفعل"
        ),
        contract,
        [{"law": "Law 12", "section": "Direct free kick"}],
    )

    assert result["hedged_low_conf"] is True
