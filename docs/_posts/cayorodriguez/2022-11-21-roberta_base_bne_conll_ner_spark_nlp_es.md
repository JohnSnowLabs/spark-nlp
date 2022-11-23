---
layout: model
title: Spanish Named Entity Recognition, (RoBERTa base trained with data from the National Library of Spain (BNE) and CONLL 2003 data), by the TEMU Unit of the BSC-CNS
author: cayorodriguez
name: roberta_base_bne_conll_ner_spark_nlp
date: 2022-11-21
tags: [es, open_source]
task: Named Entity Recognition
language: es
edition: Spark NLP 4.0.0
spark_version: 3.2
supported: false
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained Named Entity Recognition model, uploaded to Hugging Face, adapted and imported into Spark NLP. roberta-base-bne-conll-ner_spark_nlp is a Spanish model orginally trained by TEMU-BSC for PlanTL-GOB-ES.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/community.johnsnowlabs.com/cayorodriguez/roberta_base_bne_conll_ner_spark_nlp_es_4.0.0_3.2_1669018824287.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use

documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

sentenceDetector = SentenceDetector()\
.setInputCols(["document"])\
.setOutputCol("sentence")

tokenizer = Tokenizer() \
    .setInputCols("sentence") \
    .setOutputCol("token")
  
ner = RoBertaForTokenClassification.pretrained("roberta_base_bne_conll_ner_spark_nlp","es") \
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("ner")
    
pipeline = Pipeline(stages=[documentAssembler, tokenizer, ner])

data = spark.createDataFrame([["El Plan Nacional para el Impulso de las Tecnologías del Lenguage es una iniciativa del Gobierno de España"]]).toDF("text")

result = pipeline.fit(data).transform(data)

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

sentenceDetector = SentenceDetector()\
.setInputCols(["document"])\
.setOutputCol("sentence")

tokenizer = Tokenizer() \
    .setInputCols("sentence") \
    .setOutputCol("token")
  
ner = RoBertaForTokenClassification.pretrained("roberta_base_bne_conll_ner_spark_nlp","es") \
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("ner")
    
pipeline = Pipeline(stages=[documentAssembler, tokenizer, ner])

data = spark.createDataFrame([["El Plan Nacional para el Impulso de las Tecnologías del Lenguage es una iniciativa del Gobierno de España"]]).toDF("text")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|roberta_base_bne_conll_ner_spark_nlp|
|Compatibility:|Spark NLP 4.0.0+|
|License:|Open Source|
|Edition:|Community|
|Input Labels:|[document, token]|
|Output Labels:|[ner]|
|Language:|es|
|Size:|447.3 MB|
|Case sensitive:|true|
|Max sentence length:|128|