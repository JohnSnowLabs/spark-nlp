---
layout: model
title: Multilingual T5ForConditionalGeneration Small Cased model (from google)
author: John Snow Labs
name: t5_flan_small
date: 2023-01-30
tags: [vi, ne, fi, ur, ku, yo, si, ru, it, zh, la, hi, he, xh, so, ca, ar, as, sw, en, ro, ig, te, th, ta, ce, es, gu, or, fr, ka, "no", li, cr, ch, be, ha, ga, ja, pa, ko, sl, open_source, t5, xx, tensorflow]
task: Text Generation
language: xx
edition: Spark NLP 4.3.0
spark_version: 3.0
supported: true
engine: tensorflow
annotator: T5Transformer
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained T5ForConditionalGeneration model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `flan-t5-small` is a Multilingual model originally trained by `google`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/t5_flan_small_xx_4.3.0_3.0_1675102370004.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/t5_flan_small_xx_4.3.0_3.0_1675102370004.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler() \
    .setInputCols("text") \
    .setOutputCols("document")

t5 = T5Transformer.pretrained("t5_flan_small","xx") \
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
       
val t5 = T5Transformer.pretrained("t5_flan_small","xx") 
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
|Model Name:|t5_flan_small|
|Compatibility:|Spark NLP 4.3.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[documents]|
|Output Labels:|[t5]|
|Language:|xx|
|Size:|349.5 MB|

## References

- https://huggingface.co/google/flan-t5-small
- https://s3.amazonaws.com/moonup/production/uploads/1666363435475-62441d1d9fdefb55a0b7d12c.png
- https://github.com/google-research/t5x/blob/main/docs/models.md#flan-t5-checkpoints
- https://arxiv.org/pdf/2210.11416.pdf
- https://github.com/google-research/t5x
- https://arxiv.org/pdf/2210.11416.pdf
- https://arxiv.org/pdf/2210.11416.pdf
- https://arxiv.org/pdf/2210.11416.pdf
- https://s3.amazonaws.com/moonup/production/uploads/1666363265279-62441d1d9fdefb55a0b7d12c.png
- https://arxiv.org/pdf/2210.11416.pdf
- https://github.com/google-research/t5x
- https://github.com/google/jax
- https://s3.amazonaws.com/moonup/production/uploads/1668072995230-62441d1d9fdefb55a0b7d12c.png
- https://arxiv.org/pdf/2210.11416.pdf
- https://arxiv.org/pdf/2210.11416.pdf
- https://mlco2.github.io/impact#compute
- https://arxiv.org/abs/1910.09700