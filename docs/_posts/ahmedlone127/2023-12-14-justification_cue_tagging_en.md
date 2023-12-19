---
layout: model
title: English justification_cue_tagging RoBertaForTokenClassification from chefkoch24
author: John Snow Labs
name: justification_cue_tagging
date: 2023-12-14
tags: [roberta, en, open_source, token_classification, onnx]
task: Named Entity Recognition
language: en
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

Pretrained RoBertaForTokenClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`justification_cue_tagging` is a English model originally trained by chefkoch24.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/justification_cue_tagging_en_5.2.1_3.0_1702514840664.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/justification_cue_tagging_en_5.2.1_3.0_1702514840664.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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
        
    
tokenClassifier = RoBertaForTokenClassification.pretrained("justification_cue_tagging","en") \
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
    .pretrained("justification_cue_tagging", "en")
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
|Model Name:|justification_cue_tagging|
|Compatibility:|Spark NLP 5.2.1+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[documents, token]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|306.8 MB|

## References

https://huggingface.co/chefkoch24/justification-cue-tagging