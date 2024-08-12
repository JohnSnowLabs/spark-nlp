---
layout: model
title: Malay T5ForConditionalGeneration Base Cased model (from mesolitica)
author: John Snow Labs
name: t5_finetune_paraphrase_base_standard_bahasa_cased
date: 2024-07-29
tags: [ms, open_source, t5, onnx]
task: Text Generation
language: ms
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

Pretrained T5ForConditionalGeneration model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `finetune-paraphrase-t5-base-standard-bahasa-cased` is a Malay model originally trained by `mesolitica`.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/t5_finetune_paraphrase_base_standard_bahasa_cased_ms_5.4.2_3.0_1722269184029.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/t5_finetune_paraphrase_base_standard_bahasa_cased_ms_5.4.2_3.0_1722269184029.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler() \
    .setInputCols("text") \
    .setOutputCols("document")

t5 = T5Transformer.pretrained("t5_finetune_paraphrase_base_standard_bahasa_cased","ms") \
    .setInputCols("document") \
    .setOutputCol("answers")
    
pipeline = Pipeline(stages=[documentAssembler, t5])

data = spark.createDataFrame([["PUT YOUR STRING HERE"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler() 
      .setInputCols("text")
      .setOutputCols("document")
       
val t5 = T5Transformer.pretrained("t5_finetune_paraphrase_base_standard_bahasa_cased","ms") 
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
|Model Name:|t5_finetune_paraphrase_base_standard_bahasa_cased|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document]|
|Output Labels:|[output]|
|Language:|ms|
|Size:|1.0 GB|

## References

References

- https://huggingface.co/mesolitica/finetune-paraphrase-t5-base-standard-bahasa-cased
- https://github.com/huseinzol05/malaya/tree/master/session/paraphrase/hf-t5