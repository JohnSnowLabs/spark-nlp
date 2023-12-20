---
layout: model
title: Icelandic DistilBertForTokenClassification Cased model (from m3hrdadfi)
author: John Snow Labs
name: distilbert_token_classifier_icelandic_ner
date: 2023-11-20
tags: [is, open_source, distilbert, token_classification, ner, onnx]
task: Named Entity Recognition
language: is
edition: Spark NLP 5.2.0
spark_version: 3.0
supported: true
engine: onnx
annotator: DistilBertForTokenClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained DistilBertForTokenClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `icelandic-ner-distilbert` is a Icelandic model originally trained by `m3hrdadfi`.

## Predicted Entities

`Money`, `Date`, `Time`, `Percent`, `Miscellaneous`, `Location`, `Person`, `Organization`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/distilbert_token_classifier_icelandic_ner_is_5.2.0_3.0_1700518398506.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/distilbert_token_classifier_icelandic_ner_is_5.2.0_3.0_1700518398506.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

tokenizer = Tokenizer() \
    .setInputCols("document") \
    .setOutputCol("token")

tokenClassifier = DistilBertForTokenClassification.pretrained("distilbert_token_classifier_icelandic_ner_distilbert","is") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("ner")

pipeline = Pipeline(stages=[documentAssembler, tokenizer, tokenClassifier])

data = spark.createDataFrame([["PUT YOUR STRING HERE"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")

val tokenizer = new Tokenizer()
    .setInputCols("document")
    .setOutputCol("token")

val tokenClassifier = DistilBertForTokenClassification.pretrained("distilbert_token_classifier_icelandic_ner_distilbert","is")
    .setInputCols(Array("document", "token"))
    .setOutputCol("ner")

val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, tokenClassifier))

val data = Seq("PUT YOUR STRING HERE").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|distilbert_token_classifier_icelandic_ner|
|Compatibility:|Spark NLP 5.2.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[documents, token]|
|Output Labels:|[ner]|
|Language:|is|
|Size:|505.4 MB|

## References

References

- https://huggingface.co/m3hrdadfi/icelandic-ner-distilbert
- http://hdl.handle.net/20.500.12537/42
- https://en.ru.is/
- https://github.com/m3hrdadfi/icelandic-ner/issues