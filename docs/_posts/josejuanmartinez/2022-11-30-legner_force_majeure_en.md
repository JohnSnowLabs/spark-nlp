---
layout: model
title: NER on Force Majeure Clauses
author: John Snow Labs
name: legner_force_majeure
date: 2022-11-30
tags: [force, majeure, en, licensed]
task: Named Entity Recognition
language: en
edition: Legal NLP 1.0.0
spark_version: 3.0
supported: true
annotator: LegalNerModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model should be run on Force Majeure clauses. Use a Text Classifier to identify those clauses in your document, then run this NER on them - it will extract keywords related to Force Majeure exemptions.

## Predicted Entities

`O`, `FORCE_MAJEURE`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/legal/models/legner_force_majeure_en_1.0.0_3.0_1669802449878.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use

documentAssembler = nlp.DocumentAssembler()\
        .setInputCol("text")\
        .setOutputCol("document")
        
sentenceDetector = nlp.SentenceDetectorDLModel.pretrained("sentence_detector_dl","xx")\
        .setInputCols(["document"])\
        .setOutputCol("sentence")

tokenizer = nlp.Tokenizer()\
        .setInputCols(["sentence"])\
        .setOutputCol("token")

embeddings = nlp.RoBertaEmbeddings.pretrained("roberta_embeddings_legal_roberta_base","en") \
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("embeddings")

ner_model = legal.NerModel.pretrained('legner_force_majeure','en','legal/models')\
        .setInputCols(["sentence", "token", "embeddings"])\
        .setOutputCol("ner")

ner_converter = nlp.NerConverter()\
        .setInputCols(["sentence","token","ner"])\
        .setOutputCol("ner_chunk")

nlpPipeline = Pipeline(stages=[
        documentAssembler,
        sentenceDetector,
        tokenizer,
        embeddings,
        ner_model,
        ner_converter])

empty_data = spark.createDataFrame([[""]]).toDF("text")

model = nlpPipeline.fit(empty_data)

text = ["""Force Majeure. In no event shall the Trustee be responsible or liable for any failure or delay in the performance of its obligations hereunder arising out of or caused by, directly or indirectly, forces beyond its control, including, without limitation, strikes, work stoppages, accidents, acts of war or terrorism, civil or military disturbances, nuclear or natural catastrophes or acts of God, and interruptions, loss or malfunctions of utilities, communications or computer (software and hardware) services; it being understood that the Trustee shall use reasonable efforts which are consistent with accepted practices in the banking industry to resume performance as soon as practicable under the circumstances."""]

res = model.transform(spark.createDataFrame([text]).toDF("text"))

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
+--------------+---------------+
|         token|      ner_label|
+--------------+---------------+
...
|             ,|              O|
|      directly|              O|
|            or|              O|
|    indirectly|              O|
|             ,|              O|
|        forces|              O|
|        beyond|              O|
|           its|              O|
|       control|              O|
|             ,|              O|
|     including|              O|
|             ,|              O|
|       without|              O|
|    limitation|              O|
|             ,|              O|
|       strikes|B-FORCE_MAJEURE|
|             ,|              O|
|          work|B-FORCE_MAJEURE|
|     stoppages|I-FORCE_MAJEURE|
|             ,|              O|
|     accidents|B-FORCE_MAJEURE|
|             ,|              O|
|          acts|B-FORCE_MAJEURE|
|            of|I-FORCE_MAJEURE|
|           war|I-FORCE_MAJEURE|
|            or|              O|
|     terrorism|B-FORCE_MAJEURE|
|             ,|              O|
|         civil|B-FORCE_MAJEURE|
|            or|              O|
|      military|B-FORCE_MAJEURE|
|  disturbances|I-FORCE_MAJEURE|
|             ,|              O|
|       nuclear|B-FORCE_MAJEURE|
|            or|              O|
|       natural|B-FORCE_MAJEURE|
|  catastrophes|I-FORCE_MAJEURE|
|            or|              O|
|          acts|B-FORCE_MAJEURE|
|            of|I-FORCE_MAJEURE|
|           God|I-FORCE_MAJEURE|
|             ,|              O|
|           and|              O|
| interruptions|B-FORCE_MAJEURE|
|             ,|              O|
|          loss|B-FORCE_MAJEURE|
|            or|              O|
|  malfunctions|B-FORCE_MAJEURE|
|            of|I-FORCE_MAJEURE|
|     utilities|I-FORCE_MAJEURE|
|             ,|              O|
|communications|B-FORCE_MAJEURE|
...
+--------------+---------------+
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|legner_force_majeure|
|Compatibility:|Legal NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|16.5 MB|

## References

In-house annotations on CUAD dataset

## Benchmarking

```bash
label	 tp	 fp	 fn	 prec	 rec	 f1
I-FORCE_MAJEURE	 91	 36	 37	 0.71653545	 0.7109375	 0.7137255
B-FORCE_MAJEURE	 140	 32	 17	 0.81395346	 0.89171976	 0.85106385
Macro-average	 231 68 54 0.7652445 0.80132866 0.782871
Micro-average	 231 68 54 0.77257526 0.8105263 0.7910959
```