---
layout: model
title: French m1_ind_layers_ref_cmbert_io_level_2 CamemBertForTokenClassification from nlpso
author: John Snow Labs
name: m1_ind_layers_ref_cmbert_io_level_2
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

Pretrained CamemBertForTokenClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`m1_ind_layers_ref_cmbert_io_level_2` is a French model originally trained by nlpso.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/m1_ind_layers_ref_cmbert_io_level_2_fr_5.2.4_3.0_1705837017049.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/m1_ind_layers_ref_cmbert_io_level_2_fr_5.2.4_3.0_1705837017049.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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
        
    
tokenClassifier = CamemBertForTokenClassification.pretrained("m1_ind_layers_ref_cmbert_io_level_2","fr") \
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
    .pretrained("m1_ind_layers_ref_cmbert_io_level_2", "fr")
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
|Model Name:|m1_ind_layers_ref_cmbert_io_level_2|
|Compatibility:|Spark NLP 5.2.4+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[documents, token]|
|Output Labels:|[ner]|
|Language:|fr|
|Size:|412.0 MB|

## References

https://huggingface.co/nlpso/m1_ind_layers_ref_cmbert_io_level_2