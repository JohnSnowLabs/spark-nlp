---
layout: model
title: Castilian, Spanish roberta_token_classifier_bertin_base_sayula_popoluca_conll2002 RoBertaForTokenClassification from bertin-project
author: John Snow Labs
name: roberta_token_classifier_bertin_base_sayula_popoluca_conll2002
date: 2023-12-14
tags: [roberta, es, open_source, token_classification, onnx]
task: Named Entity Recognition
language: es
edition: Spark NLP 5.2.1
spark_version: 3.0
supported: true
engine: onnx
annotator: RoBertaForTokenClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RoBertaForTokenClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`roberta_token_classifier_bertin_base_sayula_popoluca_conll2002` is a Castilian, Spanish model originally trained by bertin-project.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/roberta_token_classifier_bertin_base_sayula_popoluca_conll2002_es_5.2.1_3.0_1702512255258.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/roberta_token_classifier_bertin_base_sayula_popoluca_conll2002_es_5.2.1_3.0_1702512255258.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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
        
    
tokenClassifier = RoBertaForTokenClassification.pretrained("roberta_token_classifier_bertin_base_sayula_popoluca_conll2002","es") \
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

val tokenClassifier = RoBertaForTokenClassification  
    .pretrained("roberta_token_classifier_bertin_base_sayula_popoluca_conll2002", "es")
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
|Model Name:|roberta_token_classifier_bertin_base_sayula_popoluca_conll2002|
|Compatibility:|Spark NLP 5.2.1+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[documents, token]|
|Output Labels:|[ner]|
|Language:|es|
|Size:|425.8 MB|

## References

https://huggingface.co/bertin-project/bertin-base-pos-conll2002-es