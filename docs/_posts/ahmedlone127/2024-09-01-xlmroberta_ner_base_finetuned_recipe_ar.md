---
layout: model
title: Arabic XLMRobertaForTokenClassification Base Cased model (from edwardjross)
author: John Snow Labs
name: xlmroberta_ner_base_finetuned_recipe
date: 2024-09-01
tags: [ar, open_source, xlm_roberta, ner, onnx]
task: Named Entity Recognition
language: ar
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

Pretrained XLMRobertaForTokenClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `xlm-roberta-base-finetuned-recipe-ar` is a Arabic model originally trained by `edwardjross`.

## Predicted Entities

`UNIT`, `DF`, `QUANTITY`, `TEMP`, `SIZE`, `NAME`, `STATE`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/xlmroberta_ner_base_finetuned_recipe_ar_5.4.2_3.0_1725152882185.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/xlmroberta_ner_base_finetuned_recipe_ar_5.4.2_3.0_1725152882185.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

token_classifier = XlmRoBertaForTokenClassification.pretrained("xlmroberta_ner_base_finetuned_recipe","ar") \
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

val token_classifier = XlmRoBertaForTokenClassification.pretrained("xlmroberta_ner_base_finetuned_recipe","ar")
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
nlu.load("ar.ner.xlmr_roberta.base_finetuned").predict("""PUT YOUR STRING HERE""")
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|xlmroberta_ner_base_finetuned_recipe|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[ner]|
|Language:|ar|
|Size:|823.4 MB|

## References

References

- https://huggingface.co/edwardjross/xlm-roberta-base-finetuned-recipe-ar