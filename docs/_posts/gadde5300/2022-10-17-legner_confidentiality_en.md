---
layout: model
title: Legal Warranty NER
author: John Snow Labs
name: legner_confidentiality
date: 2022-10-17
tags: [legal, en, ner, licensed, confidentiality, open_source]
task: Named Entity Recognition
language: en
edition: Spark NLP 4.2.0
spark_version: 3.2
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This is a Legal Named Entity Recognition Model to identify the Subject (who), Action (web), Object(the indemnification) and Indirect Object (to whom) from Confidentiality clauses.

## Predicted Entities

`CONFIDENTIALITY`, `CONFIDENTIALITY_ACTION`, `CONFIDENTIALITY_INDIRECT_OBJECT`, `CONFIDENTIALITY_SUBJECT`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/legner_confidentiality_en_4.2.0_3.2_1665982821402.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

ner_model = legal.NerModel.pretrained('legner_confidentiality', 'en', 'legal/models')\
        .setInputCols(["sentence", "token", "embeddings"])\
        .setOutputCol("ner")

ner_converter = nlp.NerConverter()\
        .setInputCols(["sentence","token","ner"])\
        .setOutputCol("ner_chunk")

nlpPipeline = Pipeline(stages=[documentAssembler,sentenceDetector,tokenizer,embeddings,ner_model,ner_converter])

data = spark.createDataFrame([["( b ) not to use such Confidential Information and Know-How of the Disclosing Party for any purpose other than those set forth herein ."]]).toDF("text")

result = nlpPipeline.fit(data).transform(data)
```

</div>

## Results

```bash
+------------------------+-------------------------------+
|chunk                   |entity                         |
+------------------------+-------------------------------+
|not to use              |CONFIDENTIALITY_ACTION         |
|Confidential Information|CONFIDENTIALITY                |
|Disclosing Party        |CONFIDENTIALITY_INDIRECT_OBJECT|
+------------------------+-------------------------------+

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|legner_confidentiality|
|Compatibility:|Spark NLP 4.2.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|16.3 MB|

## References

In-house annotated examples from CUAD legal dataset
