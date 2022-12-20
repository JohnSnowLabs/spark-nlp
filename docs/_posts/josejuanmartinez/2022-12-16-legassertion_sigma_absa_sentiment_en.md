---
layout: model
title: Legal Sentiment Analysis using Assertion Status (Sigma, ABSA dataset)
author: John Snow Labs
name: legassertion_sigma_absa_sentiment
date: 2022-12-16
tags: [en, licensed]
task: Assertion Status
language: en
edition: Legal NLP 1.0.0
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This mode was trained to be benchmarked against the SigmaLaw's official Aspect-based Sentiment Analysis model, based on ABSA dataset, where several parties were tagger with their sentiments in lega texts.

For more information a bout the annotation guidelines please check their official paper https://arxiv.org/pdf/2011.06326.pdf

Macro-F1 Reported by SigmaLaw:
- TD-LSTM 0.564682
- TC-LSTM 0.543762
- AE-LSTM 0.558778
- AT-LSTM 0.559181
- ATAE-LSTM 0.580193
- IAN 0.564990
- MemNet 0.436025
- Cabasc 0.564300
- RAM 0.602201

Obtained with Legal NLP:
- Assertion Status 0.637 (+0.035 compared to RAM, +0.08 in average)

More details here: https://arxiv.org/pdf/2011.06326.pdf

## Predicted Entities

`neutral`, `positive`, `negative`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/legal/models/legassertion_sigma_absa_sentiment_en_1.0.0_3.0_1671205882337.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
document_assembler = nlp.DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

sentence_detector = nlp.SentenceDetector() \
    .setInputCols(["document"]) \
    .setOutputCol("sentence")

tokenizer = nlp.Tokenizer()\
    .setInputCols(["sentence"])\
    .setOutputCol("token")

embeddings = nlp.RoBertaEmbeddings.pretrained("roberta_embeddings_legal_roberta_base","en") \
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("embeddings")

ner = legal.NerModel.pretrained("legner_sigma_absa_people", "en", "legal/models")\
        .setInputCols(["sentence", "token", "embeddings"])\
        .setOutputCol("ner")

ner_converter = nlp.NerConverter() \
    .setInputCols(["sentence", "token", "ner"]) \
    .setOutputCol("ner_chunk")

assertion = legal.AssertionDLModel.pretrained("legassertion_sigma_absa_sentiment", "en", "legal/models")\
    .setInputCols(["sentence", "ner_chunk", "embeddings"])\
    .setOutputCol("label")

pipe = nlp.Pipeline(stages = [ document_assembler, sentence_detector, tokenizer, embeddings, ner, ner_converter, assertion])

text = "Petitioner Jae Lee moved to the United States from South Korea with his parents when he was 13. He feared that a criminal conviction may affect his status."

```

</div>

## Results

```bash
+------------------+---------+
|         ner_chunk|assertion|
+------------------+---------+
|Petitioner Jae Lee|  neutral|
|               his|  neutral|
|                he|  neutral|
|       He| negative|
|      his| negative|
+---------+---------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|legassertion_sigma_absa_sentiment|
|Compatibility:|Legal NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[document, chunk, embeddings]|
|Output Labels:|[assertion]|
|Language:|en|
|Size:|2.2 MB|

## References

https://metatext.io/datasets/sigmalaw-absa

## Benchmarking

```bash
label	 tp	 fp	 fn	 prec	 rec	 f1
neutral	 36	 25	 32	 0.59016395	 0.5294118	 0.5581395
positive	 166	 111	 84	 0.599278	 0.664	 0.629981
negative	 236	 82	 102	 0.7421384	 0.69822484	 0.7195123
Macro-average 438 218 218 0.6438601 0.63054556 0.63713324
Micro-average 438 218 218 0.66768295 0.66768295 0.66768295
```