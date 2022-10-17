---
layout: model
title: Legal Warranty NER
author: John Snow Labs
name: legner_warranty
date: 2022-10-17
tags: [legal, en, ner, licensed, warranty]
task: Named Entity Recognition
language: en
edition: Spark NLP for Finance 1.0.0
spark_version: 3.2
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This is a Legal Named Entity Recognition Model to identify the Subject (who), Action (web), Object(the indemnification) and Indirect Object (to whom) from Warranty clauses.

## Predicted Entities

`WARRANTY`, `WARRANTY_ACTION`, `WARRANTY_SUBJECT`, `WARRANTY_INDIRECT_OBJECT`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/legner_warranty_en_1.0.0_3.2_1666003385749.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

ner_model = legal.NerModel.pretrained('legner_warranty', 'en', 'legal/models')\
        .setInputCols(["sentence", "token", "embeddings"])\
        .setOutputCol("ner")

ner_converter = nlp.NerConverter()\
        .setInputCols(["sentence","token","ner"])\
        .setOutputCol("ner_chunk")

nlpPipeline = Pipeline(stages=[documentAssembler,sentenceDetector,tokenizer,embeddings,ner_model,ner_converter])

data = spark.createDataFrame([["It has the right , without restriction , to grant the licenses granted under this Agreement."]]).toDF("text")

result = nlpPipeline.fit(data).transform(data)
```

</div>

## Results

```bash
+----------------------------------------------------------------------------------------+--------+
|chunk                                                                                   |entity  |
+----------------------------------------------------------------------------------------+--------+
|has the right , without restriction , to grant the licenses granted under this Agreement|WARRANTY|
+----------------------------------------------------------------------------------------+--------+

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|legner_warranty|
|Compatibility:|Spark NLP for Finance 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|16.3 MB|

## References

In-house annotated examples from CUAD legal dataset