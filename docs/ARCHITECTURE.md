# Architecture

## 1. End-to-end pipeline

```
        SoccerNet-MVFoul
        3,901 incidents | 2+ camera views each | referee-annotated
                    |
                    v
        +-------------------------------+
        |  Frame sampling               |
        |  frames 63-87 @ 17 fps        |
        |  (the foul is ~1.5 seconds)   |
        +---------------+---------------+
                        |
                        v
        +-------------------------------+
        |  Video backbone   [FROZEN]    |
        |  MViT-v2-S  ->  VideoMAE      |
        +---------------+---------------+
                        |
                        v
        +-------------------------------+
        |  Feature cache on disk        |
        |  encode once, reuse forever   |
        +---------------+---------------+
                        |
                        v
        +-------------------------------+
        |  Multi-view pooling           |
        +---------------+---------------+
                        |
                        v
        +-------------------------------+
        |  Cascade heads + auxiliaries  |
        +---------------+---------------+
                        |
    ====================|====================  CONTRACT BOUNDARY
                        v
        +-------------------------------+
        |  HakamContract  (JSON)        |
        |  labels + confidences only    |
        +---------------+---------------+
                        |
                        v
        +-------------------------------+      +------------------------+
        |  FAISS retrieval              |<-----|  Laws of the Game (AR) |
        +---------------+---------------+      +------------------------+
                        |
                        v
        +-------------------------------+
        |  Claude -> Arabic explanation |
        +---------------+---------------+
                        |
                        v
        +-------------------------------+
        |  Streamlit prototype          |
        +-------------------------------+
```

## 2. The cascade

Three binary decisions instead of one multi-class head. This mirrors how a referee
actually decides, and each stage has a 50% chance floor rather than 12.5%.

```
                          incident
                              |
                              v
                   +----------------------+
                   |  Stage 1: offence?   |
                   +----------------------+
                        |            |
                  no_offence      offence
                        |            |
                        v            v
                  [ explain     +--------------------+
                    no offence ]|  Stage 2: card?    |
                                +--------------------+
                                    |            |
                                no_card         card
                                    |            |
                                    v            v
                            [ explain      +---------------------+
                              foul, no      |  Stage 3: colour?   |
                              card ]        +---------------------+
                                                |            |
                                              yellow        red
                                                |            |
                                                v            v
                                          [ explain     [ explain
                                            caution ]     dismissal ]
```

Any stage falling below the confidence threshold stops the cascade and the system
abstains rather than guessing.

## 3. Multi-view fusion

The backbone weights are shared across views. Only the pooling layer and the heads
are learned on top of cached embeddings.

```
   view 1  (live action)  --->  [ backbone ]  --->  emb_1  --+
                                                             |
   view 2  (replay)       --->  [ backbone ]  --->  emb_2  --+--> [ pooling ] --> [ heads ]
                                                             |
   view N  (replay)       --->  [ backbone ]  --->  emb_N  --+
                                 shared weights
```

## 4. Why the contract matters

This is the property that separates Hakam from an end-to-end multimodal model.

```
        VISION SIDE               |            LANGUAGE SIDE
                                  |
   pixels, frames, embeddings     |    labels + confidences only
                                  |
   +----------------------+       |       +----------------------+
   |   cascade heads      |       |       |   retrieval + LLM    |
   +----------+-----------+       |       +----------+-----------+
              |                   |                  ^
              |                   |                  |
              +----> HakamContract (JSON) -----------+
                                  |
                                  |
        The language model never receives a single pixel.
        It cannot assert a visual detail the classifier did not produce.
```

A multimodal model shown raw frames can describe contact it never reliably
perceived. Here that failure mode is structurally impossible rather than
discouraged by prompt wording — and the property is measurable:

```
                       claims traceable to a contract field
  faithfulness  =  ----------------------------------------
                        total claims in the explanation
```

## 5. What is built when

The language half depends only on JSON, so it can be finished before the dataset
arrives.

```
   NO DATA NEEDED                        NEEDS DATASET ACCESS
   ------------------------------        ------------------------------
   Laws corpus + chunking                Frame sampling
   Contract schema          ........>    Backbone + feature cache
   Retrieval + prompts                   Cascade training
   Mock contracts                        Evaluation on the real test split
   Faithfulness evaluation
   Streamlit shell
```
