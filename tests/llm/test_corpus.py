import json

from src.config import LAWS_DIR
from src.contract import Prediction, mock_contract
from src.llm.lexicon import missing_labels


REQUIRED_FIELDS = {
    "id",
    "law",
    "section",
    "title_ar",
    "text_ar",
    "text_en",
    "tags",
    "source_pages",
}


def test_corpus_is_curated_bilingual_and_has_unique_ids():
    corpus = json.loads((LAWS_DIR / "corpus.json").read_text(encoding="utf-8"))

    assert 30 <= len(corpus) <= 50
    assert len({item["id"] for item in corpus}) == len(corpus)
    for item in corpus:
        assert REQUIRED_FIELDS <= set(item)
        assert item["text_ar"].strip()
        assert item["text_en"].strip()
        assert item["source_pages"]
        assert item["tags"]


def test_lexicon_covers_every_contract_label():
    action_classes = (
        "tackling",
        "standing tackling",
        "high leg",
        "holding",
        "pushing",
        "elbowing",
        "challenge",
        "dive",
        "dont know",
    )
    contracts = [
        mock_contract(),
        mock_contract(offence="no_offence", card=None, colour=None),
        mock_contract(card="no_card", colour=None),
        mock_contract(colour=None),
    ]
    for action_class in action_classes:
        contract = mock_contract(colour=None)
        contract.attributes.update(
            {
                "action_class": Prediction(action_class, 0.80),
                "body_part": Prediction("upper_body", 0.80),
                "contact": Prediction("without_contact", 0.80),
                "try_to_play": Prediction("yes", 0.80),
                "touch_ball": Prediction("no", 0.80),
            }
        )
        contracts.append(contract)

    assert all(missing_labels(contract.citable_values()) == set() for contract in contracts)
