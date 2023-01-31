---
layout: model
title: German BertForTokenClassification Large Cased model (from Ruth)
author: John Snow Labs
name: bert_ner_gbert_large_germaner
date: 2022-08-02
tags: [bert, ner, open_source, de]
task: Named Entity Recognition
language: de
edition: Spark NLP 4.1.0
spark_version: 3.0
supported: true
annotator: BertForTokenClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForTokenClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `gbert-large-germaner` is a German model originally trained by `Ruth`.

## Predicted Entities

`PER`, `ORG`, `LOC`, `OTH`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_ner_gbert_large_germaner_de_4.1.0_3.0_1659456331791.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_ner_gbert_large_germaner_de_4.1.0_3.0_1659456331791.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler() \
        .setInputCol("text") \
        .setOutputCol("document")

sentenceDetector = SentenceDetectorDLModel.pretrained("sentence_detector_dl", "xx")\
       .setInputCols(["document"])\
       .setOutputCol("sentence")

tokenizer = Tokenizer() \
    .setInputCols("sentence") \
    .setOutputCol("token")

tokenClassifier = BertForTokenClassification.pretrained("bert_ner_gbert_large_germaner","de") \
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("ner")

pipeline = Pipeline(stages=[documentAssembler, sentenceDetector, tokenizer, tokenClassifier])

data = spark.createDataFrame([["Ich liebe Spark NLP"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler() 
          .setInputCol("text") 
          .setOutputCol("document")

val sentenceDetector = SentenceDetectorDLModel.pretrained("sentence_detector_dl", "xx")
       .setInputCols(Array("document"))
       .setOutputCol("sentence")

val tokenizer = new Tokenizer() 
    .setInputCols(Array("sentence"))
    .setOutputCol("token")

val tokenClassifier = BertForTokenClassification.pretrained("bert_ner_gbert_large_germaner","de") 
    .setInputCols(Array("sentence", "token")) 
    .setOutputCol("ner")

val pipeline = new Pipeline().setStages(Array(documentAssembler,sentenceDetector, tokenizer, tokenClassifier))

val data = Seq("Ich liebe Spark NLP").toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_ner_gbert_large_germaner|
|Compatibility:|Spark NLP 4.1.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[ner]|
|Language:|de|
|Size:|1.3 GB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

- https://huggingface.co/Ruth/gbert-large-germaner
- https://paperswithcode.com/sota?task=Token+Classification&dataset=germaner