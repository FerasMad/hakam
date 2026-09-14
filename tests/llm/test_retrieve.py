from src.contract import HakamContract, Prediction
from src.llm.retrieve import _load_corpus, retrieve


def _contract(
    offence: str = "offence",
    card: str | None = "card",
    colour: str | None = None,
    contact: str = "with_contact",
    action_class: str = "tackling",
) -> HakamContract:
    return HakamContract(
        action_id="retrieval-test",
        offence=Prediction(offence, 0.91),
        card=None if card is None else Prediction(card, 0.84),
        card_colour=None if colour is None else Prediction(colour, 0.82),
        attributes={
            "contact": Prediction(contact, 0.90),
            "action_class": Prediction(action_class, 0.87),
            "body_part": Prediction("under_body", 0.78),
        },
    )


def test_retrieve_returns_k_items_and_law_12_in_top_three(monkeypatch):
    monkeypatch.setenv("HAKAM_RETRIEVAL_MODE", "tags")
    _load_corpus.cache_clear()

    articles = retrieve(_contract(), k=5)

    assert len(articles) == 5
    assert any(article["law"] == "Law 12" for article in articles[:3])
    assert any("tackling" in article["tags"] for article in articles[:3])


def test_no_offence_prefers_fair_challenge_article(monkeypatch):
    monkeypatch.setenv("HAKAM_RETRIEVAL_MODE", "tags")
    _load_corpus.cache_clear()

    articles = retrieve(
        _contract(
            offence="no_offence",
            card=None,
            contact="with_contact",
            action_class="challenge",
        ),
        k=3,
    )

    assert articles[0]["id"] == "law12-no-offence-fair-challenge"


def test_pushing_does_not_infer_a_goal_scoring_opportunity(monkeypatch):
    monkeypatch.setenv("HAKAM_RETRIEVAL_MODE", "tags")
    _load_corpus.cache_clear()

    articles = retrieve(_contract(action_class="pushing"), k=3)

    assert articles[0]["id"] == "law12-1-pushing"
    assert not any("dogso" in article["id"] for article in articles)


def test_dive_does_not_retrieve_thrown_object_articles(monkeypatch):
    monkeypatch.setenv("HAKAM_RETRIEVAL_MODE", "tags")
    _load_corpus.cache_clear()

    articles = retrieve(
        _contract(action_class="dive", contact="without_contact"),
        k=3,
    )

    assert articles[0]["id"] == "law12-3-simulation"
    assert not any("thrown-object" in article["id"] for article in articles)
