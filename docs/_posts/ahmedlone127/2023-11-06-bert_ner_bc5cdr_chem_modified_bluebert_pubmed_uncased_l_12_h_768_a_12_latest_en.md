---
layout: model
title: English bert_ner_bc5cdr_chem_modified_bluebert_pubmed_uncased_l_12_h_768_a_12_latest BertForTokenClassification from ghadeermobasher
author: John Snow Labs
name: bert_ner_bc5cdr_chem_modified_bluebert_pubmed_uncased_l_12_h_768_a_12_latest
date: 2023-11-06
tags: [bert, en, open_source, token_classification, onnx]
task: Named Entity Recognition
language: en
edition: Spark NLP 5.2.0
spark_version: 3.0
supported: true
engine: onnx
annotator: BertForTokenClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForTokenClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`bert_ner_bc5cdr_chem_modified_bluebert_pubmed_uncased_l_12_h_768_a_12_latest` is a English model originally trained by ghadeermobasher.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_ner_bc5cdr_chem_modified_bluebert_pubmed_uncased_l_12_h_768_a_12_latest_en_5.2.0_3.0_1699272396685.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_ner_bc5cdr_chem_modified_bluebert_pubmed_uncased_l_12_h_768_a_12_latest_en_5.2.0_3.0_1699272396685.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python


documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("documents")
    
    
tokenClassifier = BertForTokenClassification.pretrained("bert_ner_bc5cdr_chem_modified_bluebert_pubmed_uncased_l_12_h_768_a_12_latest","en") \
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
    
val tokenClassifier = BertForTokenClassification  
    .pretrained("bert_ner_bc5cdr_chem_modified_bluebert_pubmed_uncased_l_12_h_768_a_12_latest", "en")
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
|Model Name:|bert_ner_bc5cdr_chem_modified_bluebert_pubmed_uncased_l_12_h_768_a_12_latest|
|Compatibility:|Spark NLP 5.2.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[documents, token]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|407.1 MB|

## References

https://huggingface.co/ghadeermobasher/BC5CDR-Chem-Modified_bluebert_pubmed_uncased_L-12_H-768_A-12_latest