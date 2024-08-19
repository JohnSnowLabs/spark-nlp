---
layout: model
title: Multilingual pyabsa_hospital_multilingual_allenai_turkmen_instruct_base_def_sayula_popoluca_finedtuned_model T5Transformer from amir22010
author: John Snow Labs
name: pyabsa_hospital_multilingual_allenai_turkmen_instruct_base_def_sayula_popoluca_finedtuned_model
date: 2024-08-19
tags: [xx, open_source, onnx, t5, question_answering, summarization, translation, text_generation]
task: [Question Answering, Summarization, Translation, Text Generation]
language: xx
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

Pretrained T5Transformer model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`pyabsa_hospital_multilingual_allenai_turkmen_instruct_base_def_sayula_popoluca_finedtuned_model` is a Multilingual model originally trained by amir22010.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/pyabsa_hospital_multilingual_allenai_turkmen_instruct_base_def_sayula_popoluca_finedtuned_model_xx_5.4.2_3.0_1724053415618.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/pyabsa_hospital_multilingual_allenai_turkmen_instruct_base_def_sayula_popoluca_finedtuned_model_xx_5.4.2_3.0_1724053415618.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
     
documentAssembler = DocumentAssembler() \
    .setInputCol('text') \
    .setOutputCol('document')

t5  = T5Transformer.pretrained("pyabsa_hospital_multilingual_allenai_turkmen_instruct_base_def_sayula_popoluca_finedtuned_model","xx") \
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

val t5 = T5Transformer.pretrained("pyabsa_hospital_multilingual_allenai_turkmen_instruct_base_def_sayula_popoluca_finedtuned_model", "xx")
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
|Model Name:|pyabsa_hospital_multilingual_allenai_turkmen_instruct_base_def_sayula_popoluca_finedtuned_model|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document]|
|Output Labels:|[output]|
|Language:|xx|
|Size:|936.9 MB|

## References

https://huggingface.co/amir22010/PyABSA_Hospital_Multilingual_allenai_tk-instruct-base-def-pos_FinedTuned_Model