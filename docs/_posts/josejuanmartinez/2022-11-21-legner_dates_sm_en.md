---
layout: model
title: Extract Effective, Renewal, Termination dates (Small)
author: John Snow Labs
name: legner_dates_sm
date: 2022-11-21
tags: [renewal, effective, termination, date, en, licensed]
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

This models extracts if a date is a Effective Date, a Renewal Date or a Termination Date, and also extracts the keywords surrounding that may be pointing about what kind of date it is. Please take into account that the keyword was not used to learn the date, all entities were training separately. But you can use the keywords to double check the date type is correct.

## Predicted Entities

`EFFDATE`, `EFFDATE_KEYWORD`, `RENDATE`, `RENDATE_KEYWORD`, `TERMINDATE`, `TERMINDATE_KEYWORD`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/legal/models/legner_dates_sm_en_1.0.0_3.0_1669028480461.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
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

ner_model = legal.NerModel.pretrained('legner_dates_sm', 'en', 'legal/models')\
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

text = ["""RENEWAL DATE. The date on which this Agreement shall renew, July 1st, pursuant to the terms and conditions contained herein."""]

res = model.transform(spark.createDataFrame([text]).toDF("text"))
```

</div>

## Results

```bash
+----------+-----------------+
|     token|        ner_label|
+----------+-----------------+
|   RENEWAL|B-RENDATE_KEYWORD|
|      DATE|I-RENDATE_KEYWORD|
|         .|                O|
|       The|                O|
|      date|                O|
|        on|                O|
|     which|                O|
|      this|                O|
| Agreement|                O|
|     shall|                O|
|     renew|                O|
|         ,|                O|
|      July|        B-RENDATE|
|       1st|        I-RENDATE|
|         ,|                O|
|  pursuant|                O|
|        to|                O|
|       the|                O|
|     terms|                O|
|       and|                O|
|conditions|                O|
| contained|                O|
|    herein|                O|
|         .|                O|
+----------+-----------------+

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|legner_dates_sm|
|Compatibility:|Legal NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|16.1 MB|

## References

In-house dataset.

## Benchmarking

```bash
label	 tp	 fp	 fn	 prec	 rec	 f1
I-RENDATE	 6	 2	 5	 0.75	 0.54545456	 0.631579
B-EFFDATE_KEYWORD	 5	 0	 0	 1.0	 1.0	 1.0
B-EFFDATE	 5	 0	 0	 1.0	 1.0	 1.0
B-TERMINDATE	 3	 1	 0	 0.75	 1.0	 0.85714287
I-TERMINDATE	 9	 4	 0	 0.6923077	 1.0	 0.8181818
I-EFFDATE	 15	 0	 0	 1.0	 1.0	 1.0
I-RENDATE_KEYWORD	 4	 0	 0	 1.0	 1.0	 1.0
I-EFFDATE_KEYWORD	 5	 0	 0	 1.0	 1.0	 1.0
I-TERMINDATE_KEYWORD	 5	 0	 0	 1.0	 1.0	 1.0
B-RENDATE	 2	 1	 2	 0.6666667	 0.5	 0.57142854
B-TERMINDATE_KEYWORD	 5	 0	 0	 1.0	 1.0	 1.0
B-RENDATE_KEYWORD	 3	 0	 1	 1.0	 0.75	 0.85714287
Macro-average 67 8 8 0.90491456 0.8996212 0.9022601
Micro-average 67 8 8 0.8933333 0.8933333 0.89333326
```
