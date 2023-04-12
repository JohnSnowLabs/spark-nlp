---
layout: model
title: English Legal BERT Sentence Embedding Base Cased model
author: John Snow Labs
name: sent_bert_multi_qa_mpnet_base_dot_v1_legal_finetune
date: 2023-03-15
tags: [bert, en, embeddings, sentence, open_source, tensorflow]
task: Embeddings
language: en
edition: Spark NLP 4.3.2
spark_version: 3.0
supported: true
engine: tensorflow
annotator: BertSentenceEmbeddings
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained Legal BERT Sentence Embedding model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `multi-qa-mpnet-base-dot-v1_legal_finetune` is a English model originally trained by `oliviamga2`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sent_bert_multi_qa_mpnet_base_dot_v1_legal_finetune_en_4.3.2_3.0_1678892286526.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/sent_bert_multi_qa_mpnet_base_dot_v1_legal_finetune_en_4.3.2_3.0_1678892286526.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
sent_embeddings = BertSentenceEmbeddings.pretrained("sent_bert_multi_qa_mpnet_base_dot_v1_legal_finetune", "en") \
.setInputCols("sentence") \
.setOutputCol("bert_sentence")

nlp_pipeline = Pipeline(stages=[document_assembler, sentence_detector, sent_embeddings ])
  result = pipeline.fit(data).transform(data)
```
```scala
vval sent_embeddings = BertSentenceEmbeddings.pretrained("sent_bert_multi_qa_mpnet_base_dot_v1_legal_finetune", "en")
.setInputCols("sentence")
.setOutputCol("bert_sentence")

val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, sent_embeddings ))
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sent_bert_multi_qa_mpnet_base_dot_v1_legal_finetune|
|Compatibility:|Spark NLP 4.3.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence]|
|Output Labels:|[bert_sentence]|
|Language:|en|
|Size:|409.0 MB|
|Case sensitive:|true|

## References

- https://huggingface.co/oliviamga2/multi-qa-mpnet-base-dot-v1_legal_finetune
- https://www.SBERT.net
- https://seb.sbert.net?model_name=%7BMODEL_NAME%7D