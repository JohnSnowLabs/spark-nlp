---
layout: model
title: English Legal BERT Sentence Embedding Cased model
author: John Snow Labs
name: sent_bert_distil_ita_legal_bert
date: 2024-09-04
tags: [bert, en, embeddings, sentence, open_source, onnx]
task: Embeddings
language: en
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
engine: onnx
annotator: BertSentenceEmbeddings
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained Legal BERT Sentence Embedding model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `distil-ita-legal-bert` is a English model originally trained by `dlicari`.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sent_bert_distil_ita_legal_bert_en_5.5.0_3.0_1725415787606.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/sent_bert_distil_ita_legal_bert_en_5.5.0_3.0_1725415787606.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
sent_embeddings = BertSentenceEmbeddings.pretrained("sent_bert_distil_ita_legal_bert", "en") \
.setInputCols("sentence") \
.setOutputCol("bert_sentence")

nlp_pipeline = Pipeline(stages=[document_assembler, sentence_detector, sent_embeddings ])
  result = pipeline.fit(data).transform(data)
```
```scala
vval sent_embeddings = BertSentenceEmbeddings.pretrained("sent_bert_distil_ita_legal_bert", "en")
.setInputCols("sentence")
.setOutputCol("bert_sentence")

val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, sent_embeddings ))
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sent_bert_distil_ita_legal_bert|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence]|
|Output Labels:|[embeddings]|
|Language:|en|
|Size:|200.4 MB|

## References

References

- https://huggingface.co/dlicari/distil-ita-legal-bert
- https://www.SBERT.net
- https://seb.sbert.net?model_name=%7BMODEL_NAME%7D