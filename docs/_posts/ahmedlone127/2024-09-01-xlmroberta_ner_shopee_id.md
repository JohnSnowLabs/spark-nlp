---
layout: model
title: Indonesian XLMRobertaForTokenClassification Cased model (from vkhangpham)
author: John Snow Labs
name: xlmroberta_ner_shopee
date: 2024-09-01
tags: [id, open_source, xlm_roberta, ner, onnx]
task: Named Entity Recognition
language: id
edition: Spark NLP 5.4.2
spark_version: 3.0
supported: true
engine: onnx
annotator: XlmRoBertaForTokenClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained XLMRobertaForTokenClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `shopee-ner` is a Indonesian model originally trained by `vkhangpham`.

## Predicted Entities

`STR`, `POI`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/xlmroberta_ner_shopee_id_5.4.2_3.0_1725180461018.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/xlmroberta_ner_shopee_id_5.4.2_3.0_1725180461018.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

token_classifier = XlmRoBertaForTokenClassification.pretrained("xlmroberta_ner_shopee","id") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("ner")

ner_converter = NerConverter()\
    .setInputCols(["document", "token", "ner"])\
    .setOutputCol("ner_chunk")

pipeline = Pipeline(stages=[documentAssembler, tokenizer, token_classifier, ner_converter])

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

val token_classifier = XlmRoBertaForTokenClassification.pretrained("xlmroberta_ner_shopee","id")
    .setInputCols(Array("document", "token"))
    .setOutputCol("ner")

val ner_converter = new NerConverter()
    .setInputCols(Array("document", "token', "ner"))
    .setOutputCol("ner_chunk")

val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, token_classifier, ner_converter))

val data = Seq("PUT YOUR STRING HERE").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```

{:.nlu-block}
```python
import nlu
nlu.load("id.ner.xlmr_roberta").predict("""PUT YOUR STRING HERE""")
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|xlmroberta_ner_shopee|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[ner]|
|Language:|id|
|Size:|864.7 MB|

## References

References

- https://huggingface.co/vkhangpham/shopee-ner