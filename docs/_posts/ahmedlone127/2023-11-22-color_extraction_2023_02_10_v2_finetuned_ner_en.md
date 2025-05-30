---
layout: model
title: English color_extraction_2023_02_10_v2_finetuned_ner DistilBertForTokenClassification from plpkpjph
author: John Snow Labs
name: color_extraction_2023_02_10_v2_finetuned_ner
date: 2023-11-22
tags: [bert, en, open_source, token_classification, onnx]
task: Named Entity Recognition
language: en
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

Pretrained DistilBertForTokenClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`color_extraction_2023_02_10_v2_finetuned_ner` is a English model originally trained by plpkpjph.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/color_extraction_2023_02_10_v2_finetuned_ner_en_5.2.0_3.0_1700663999283.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/color_extraction_2023_02_10_v2_finetuned_ner_en_5.2.0_3.0_1700663999283.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python


documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("documents")
    
    
tokenClassifier = DistilBertForTokenClassification.pretrained("color_extraction_2023_02_10_v2_finetuned_ner","en") \
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
    .pretrained("color_extraction_2023_02_10_v2_finetuned_ner", "en")
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
|Model Name:|color_extraction_2023_02_10_v2_finetuned_ner|
|Compatibility:|Spark NLP 5.2.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[documents, token]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|250.3 MB|

## References

https://huggingface.co/plpkpjph/color_extraction_2023_02_10_v2-finetuned-ner