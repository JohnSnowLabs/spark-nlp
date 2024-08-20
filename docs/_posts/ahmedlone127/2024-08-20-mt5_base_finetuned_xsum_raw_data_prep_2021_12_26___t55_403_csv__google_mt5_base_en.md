---
layout: model
title: English mt5_base_finetuned_xsum_raw_data_prep_2021_12_26___t55_403_csv__google_mt5_base T5Transformer from nestoralvaro
author: John Snow Labs
name: mt5_base_finetuned_xsum_raw_data_prep_2021_12_26___t55_403_csv__google_mt5_base
date: 2024-08-20
tags: [en, open_source, onnx, t5, question_answering, summarization, translation, text_generation]
task: [Question Answering, Summarization, Translation, Text Generation]
language: en
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

Pretrained T5Transformer model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`mt5_base_finetuned_xsum_raw_data_prep_2021_12_26___t55_403_csv__google_mt5_base` is a English model originally trained by nestoralvaro.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/mt5_base_finetuned_xsum_raw_data_prep_2021_12_26___t55_403_csv__google_mt5_base_en_5.4.2_3.0_1724174882779.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/mt5_base_finetuned_xsum_raw_data_prep_2021_12_26___t55_403_csv__google_mt5_base_en_5.4.2_3.0_1724174882779.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
     
documentAssembler = DocumentAssembler() \
    .setInputCol('text') \
    .setOutputCol('document')

t5  = T5Transformer.pretrained("mt5_base_finetuned_xsum_raw_data_prep_2021_12_26___t55_403_csv__google_mt5_base","en") \
     .setInputCols(["document"]) \
     .setOutputCol("output")

pipeline = Pipeline().setStages([documentAssembler, t5])
data = spark.createDataFrame([["I love spark-nlp"]]).toDF("text")
pipelineModel = pipeline.fit(data)
pipelineDF = pipelineModel.transform(data)

```
```scala

val documentAssembler = new DocumentAssembler()
    .setInputCols("text")
    .setOutputCols("document")

val t5 = T5Transformer.pretrained("mt5_base_finetuned_xsum_raw_data_prep_2021_12_26___t55_403_csv__google_mt5_base", "en")
    .setInputCols(Array("documents")) 
    .setOutputCol("output") 
    
val pipeline = new Pipeline().setStages(Array(documentAssembler, t5))
val data = Seq("I love spark-nlp").toDS.toDF("text")
val pipelineModel = pipeline.fit(data)
val pipelineDF = pipelineModel.transform(data)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|mt5_base_finetuned_xsum_raw_data_prep_2021_12_26___t55_403_csv__google_mt5_base|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document]|
|Output Labels:|[output]|
|Language:|en|
|Size:|1.5 GB|

## References

https://huggingface.co/nestoralvaro/mt5-base-finetuned-xsum-RAW_data_prep_2021_12_26___t55_403.csv__google_mt5_base