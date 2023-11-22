---
layout: model
title: Multilingual distilbert_base_multilingual_cased_ner_demo_amarsanaa1525 DistilBertForTokenClassification from Amarsanaa1525
author: John Snow Labs
name: distilbert_base_multilingual_cased_ner_demo_amarsanaa1525
date: 2023-11-22
tags: [bert, xx, open_source, token_classification, onnx]
task: Named Entity Recognition
language: xx
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

Pretrained DistilBertForTokenClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`distilbert_base_multilingual_cased_ner_demo_amarsanaa1525` is a Multilingual model originally trained by Amarsanaa1525.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/distilbert_base_multilingual_cased_ner_demo_amarsanaa1525_xx_5.2.0_3.0_1700680345229.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/distilbert_base_multilingual_cased_ner_demo_amarsanaa1525_xx_5.2.0_3.0_1700680345229.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python


documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("documents")
    
    
tokenClassifier = DistilBertForTokenClassification.pretrained("distilbert_base_multilingual_cased_ner_demo_amarsanaa1525","xx") \
            .setInputCols(["documents","token"]) \
            .setOutputCol("ner")

pipeline = Pipeline().setStages([documentAssembler, tokenClassifier])

pipelineModel = pipeline.fit(data)

pipelineDF = pipelineModel.transform(data)

```
```scala


val documentAssembler = new DocumentAssembler()
    .setInputCol("text") 
    .setOutputCol("embeddings")
    
val tokenClassifier = DistilBertForTokenClassification  
    .pretrained("distilbert_base_multilingual_cased_ner_demo_amarsanaa1525", "xx")
    .setInputCols(Array("documents","token")) 
    .setOutputCol("ner") 

val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenClassifier))

val pipelineModel = pipeline.fit(data)

val pipelineDF = pipelineModel.transform(data)


```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|distilbert_base_multilingual_cased_ner_demo_amarsanaa1525|
|Compatibility:|Spark NLP 5.2.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[documents, token]|
|Output Labels:|[ner]|
|Language:|xx|
|Size:|505.4 MB|

## References

https://huggingface.co/Amarsanaa1525/distilbert-base-multilingual-cased-ner-demo