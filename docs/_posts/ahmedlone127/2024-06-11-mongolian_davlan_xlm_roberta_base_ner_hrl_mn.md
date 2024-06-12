---
layout: model
title: Mongolian mongolian_davlan_xlm_roberta_base_ner_hrl XlmRoBertaForTokenClassification from Blgn94
author: John Snow Labs
name: mongolian_davlan_xlm_roberta_base_ner_hrl
date: 2024-06-11
tags: [mn, open_source, onnx, token_classification, xlm_roberta, ner]
task: Named Entity Recognition
language: mn
edition: Spark NLP 5.4.0
spark_version: 3.0
supported: true
engine: onnx
annotator: XlmRoBertaForTokenClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained XlmRoBertaForTokenClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`mongolian_davlan_xlm_roberta_base_ner_hrl` is a Mongolian model originally trained by Blgn94.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/mongolian_davlan_xlm_roberta_base_ner_hrl_mn_5.4.0_3.0_1718117634412.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/mongolian_davlan_xlm_roberta_base_ner_hrl_mn_5.4.0_3.0_1718117634412.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
     
documentAssembler = DocumentAssembler() \
    .setInputCol('text') \
    .setOutputCol('document')
    
tokenizer = Tokenizer() \
    .setInputCols(['document']) \
    .setOutputCol('token')

tokenClassifier  = XlmRoBertaForTokenClassification.pretrained("mongolian_davlan_xlm_roberta_base_ner_hrl","mn") \
     .setInputCols(["documents","token"]) \
     .setOutputCol("ner")

pipeline = Pipeline().setStages([documentAssembler, tokenizer, tokenClassifier])
data = spark.createDataFrame([["I love spark-nlp"]]).toDF("text")
pipelineModel = pipeline.fit(data)
pipelineDF = pipelineModel.transform(data)

```
```scala

val documentAssembler = new DocumentAssembler()
    .setInputCols("text")
    .setOutputCols("document")
    
val tokenizer = new Tokenizer()
    .setInputCols("document")
    .setOutputCol("token")

val tokenClassifier = XlmRoBertaForTokenClassification.pretrained("mongolian_davlan_xlm_roberta_base_ner_hrl", "mn")
    .setInputCols(Array("documents","token")) 
    .setOutputCol("ner") 
    
val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, tokenClassifier))
val data = Seq("I love spark-nlp").toDS.toDF("text")
val pipelineModel = pipeline.fit(data)
val pipelineDF = pipelineModel.transform(data)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|mongolian_davlan_xlm_roberta_base_ner_hrl|
|Compatibility:|Spark NLP 5.4.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[ner]|
|Language:|mn|
|Size:|911.8 MB|

## References

https://huggingface.co/Blgn94/mongolian-davlan-xlm-roberta-base-ner-hrl