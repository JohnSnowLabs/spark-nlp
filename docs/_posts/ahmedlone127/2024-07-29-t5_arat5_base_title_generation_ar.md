---
layout: model
title: Arabic T5ForConditionalGeneration Base Cased model (from UBC-NLP)
author: John Snow Labs
name: t5_arat5_base_title_generation
date: 2024-07-29
tags: [ar, open_source, t5, onnx]
task: Text Generation
language: ar
edition: Spark NLP 5.4.2
spark_version: 3.0
supported: true
engine: onnx
annotator: T5Transformer
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained T5ForConditionalGeneration model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `AraT5-base-title-generation` is a Arabic model originally trained by `UBC-NLP`.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/t5_arat5_base_title_generation_ar_5.4.2_3.0_1722263916922.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/t5_arat5_base_title_generation_ar_5.4.2_3.0_1722263916922.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

t5 = T5Transformer.pretrained("t5_arat5_base_title_generation","ar") \
    .setInputCols(["document"]) \
    .setOutputCol("answers")

pipeline = Pipeline(stages=[documentAssembler, t5])

data = spark.createDataFrame([["PUT YOUR STRING HERE"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler()
      .setInputCols("text")
      .setOutputCols("document")

val t5 = T5Transformer.pretrained("t5_arat5_base_title_generation","ar")
    .setInputCols("document")
    .setOutputCol("answers")

val pipeline = new Pipeline().setStages(Array(documentAssembler, t5))

val data = Seq("PUT YOUR STRING HERE").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|t5_arat5_base_title_generation|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document]|
|Output Labels:|[output]|
|Language:|ar|
|Size:|1.7 GB|

## References

References

- https://huggingface.co/UBC-NLP/AraT5-base-title-generation
- https://aclanthology.org/2022.acl-long.47/
- https://doi.org/10.14288/SOCKEYE
- https://www.tensorflow.org/tfrc