---
layout: model
title: French m2_joint_label_ocr_ptrn_cmbert_iob2 CamemBertForTokenClassification from nlpso
author: John Snow Labs
name: m2_joint_label_ocr_ptrn_cmbert_iob2
date: 2024-01-21
tags: [camembert, fr, open_source, token_classification, onnx]
task: Named Entity Recognition
language: fr
edition: Spark NLP 5.2.4
spark_version: 3.0
supported: true
engine: onnx
annotator: CamemBertForTokenClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained CamemBertForTokenClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`m2_joint_label_ocr_ptrn_cmbert_iob2` is a French model originally trained by nlpso.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/m2_joint_label_ocr_ptrn_cmbert_iob2_fr_5.2.4_3.0_1705837865210.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/m2_joint_label_ocr_ptrn_cmbert_iob2_fr_5.2.4_3.0_1705837865210.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python


documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")
    
tokenizer = Tokenizer() \
        .setInputCols(["document"]) \
        .setOutputCol("token")
        
    
tokenClassifier = CamemBertForTokenClassification.pretrained("m2_joint_label_ocr_ptrn_cmbert_iob2","fr") \
            .setInputCols(["document","token"]) \
            .setOutputCol("ner")

pipeline = Pipeline().setStages([documentAssembler, tokenizer, tokenClassifier])

pipelineModel = pipeline.fit(data)

pipelineDF = pipelineModel.transform(data)

```
```scala


val documentAssembler = new DocumentAssembler()
    .setInputCol("text") 
    .setOutputCol("document")

val tokenizer = Tokenizer() \
        .setInputCols(Array("document")) \
        .setOutputCol("token")

val tokenClassifier = CamemBertForTokenClassification  
    .pretrained("m2_joint_label_ocr_ptrn_cmbert_iob2", "fr")
    .setInputCols(Array("document","token")) 
    .setOutputCol("ner") 

val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, tokenClassifier))

val pipelineModel = pipeline.fit(data)

val pipelineDF = pipelineModel.transform(data)


```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|m2_joint_label_ocr_ptrn_cmbert_iob2|
|Compatibility:|Spark NLP 5.2.4+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[documents, token]|
|Output Labels:|[ner]|
|Language:|fr|
|Size:|412.9 MB|

## References

https://huggingface.co/nlpso/m2_joint_label_ocr_ptrn_cmbert_iob2