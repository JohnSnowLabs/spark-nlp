---
layout: model
title: Korean RobertaForSequenceClassification Base Cased model (from jason9693)
author: John Snow Labs
name: roberta_classifier_soongsilbert_base_beep
date: 2022-09-09
tags: [ko, open_source, roberta, sequence_classification, classification]
task: Text Classification
language: ko
edition: Spark NLP 4.1.0
spark_version: 3.0
supported: true
annotator: RoBertaForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RobertaForSequenceClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `SoongsilBERT-base-beep` is a Korean model originally trained by `jason9693`.

## Predicted Entities

`hate`, `offensive`, `none`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/roberta_classifier_soongsilbert_base_beep_ko_4.1.0_3.0_1662761258077.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler() \
    .setInputCols(["text"]) \
    .setOutputCols("document")

tokenizer = Tokenizer() \
    .setInputCols("document") \
    .setOutputCol("token")

seq_classifier = RoBertaForSequenceClassification.pretrained("roberta_classifier_soongsilbert_base_beep","ko") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("class")
    
pipeline = Pipeline(stages=[documentAssembler, tokenizer, seq_classifier])

data = spark.createDataFrame([["PUT YOUR STRING HERE"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler() 
      .setInputCols(Array("text")) 
      .setOutputCols(Array("document"))
      
val tokenizer = new Tokenizer()
    .setInputCols("document")
    .setOutputCol("token")
 
val seq_classifier = RoBertaForSequenceClassification.pretrained("roberta_classifier_soongsilbert_base_beep","ko") 
    .setInputCols(Array("document", "token"))
    .setOutputCol("class")
   
val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, seq_classifier))

val data = Seq("PUT YOUR STRING HERE").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|roberta_classifier_soongsilbert_base_beep|
|Compatibility:|Spark NLP 4.1.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|ko|
|Size:|369.4 MB|
|Case sensitive:|true|
|Max sentence length:|256|

## References

- https://huggingface.co/jason9693/SoongsilBERT-base-beep
- https://github.com/e9t/nsmc
- https://github.com/naver/nlp-challenge
- https://github.com/google-research-datasets/paws
- https://github.com/kakaobrain/KorNLUDatasets
- https://github.com/songys/Question_pair
- https://korquad.github.io/category/1.0_KOR.html
- https://github.com/kocohub/korean-hate-speech
- https://github.com/monologg/KoELECTRA
- https://github.com/SKTBrain/KoBERT
- https://github.com/tbai2019/HanBert-54k-N
- https://github.com/monologg/HanBert-Transformers