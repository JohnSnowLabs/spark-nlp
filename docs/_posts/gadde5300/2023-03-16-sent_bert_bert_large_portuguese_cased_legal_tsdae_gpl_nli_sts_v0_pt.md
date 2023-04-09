---
layout: model
title: Portuguese Legal BERT Sentence Embedding Large Cased model
author: John Snow Labs
name: sent_bert_bert_large_portuguese_cased_legal_tsdae_gpl_nli_sts_v0
date: 2023-03-16
tags: [bert, pt, embeddings, sentence, open_source, tensorflow]
task: Embeddings
language: pt
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

Pretrained Legal BERT Sentence Embedding model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `bert-large-portuguese-cased-legal-tsdae-gpl-nli-sts-v0` is a Portuguese model originally trained by `stjiris`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sent_bert_bert_large_portuguese_cased_legal_tsdae_gpl_nli_sts_v0_pt_4.3.2_3.0_1678937300411.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/sent_bert_bert_large_portuguese_cased_legal_tsdae_gpl_nli_sts_v0_pt_4.3.2_3.0_1678937300411.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
sent_embeddings = BertSentenceEmbeddings.pretrained("sent_bert_bert_large_portuguese_cased_legal_tsdae_gpl_nli_sts_v0", "pt") \
.setInputCols("sentence") \
.setOutputCol("bert_sentence")

nlp_pipeline = Pipeline(stages=[document_assembler, sentence_detector, sent_embeddings ])
  result = pipeline.fit(data).transform(data)
```
```scala
val sent_embeddings = BertSentenceEmbeddings.pretrained("sent_bert_bert_large_portuguese_cased_legal_tsdae_gpl_nli_sts_v0", "pt")
.setInputCols("sentence")
.setOutputCol("bert_sentence")

val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, sent_embeddings ))
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sent_bert_bert_large_portuguese_cased_legal_tsdae_gpl_nli_sts_v0|
|Compatibility:|Spark NLP 4.3.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence]|
|Output Labels:|[bert_sentence]|
|Language:|pt|
|Size:|1.3 GB|
|Case sensitive:|true|

## References

- https://huggingface.co/stjiris/bert-large-portuguese-cased-legal-tsdae-gpl-nli-sts-v0
- https://rufimelo99.github.io/SemanticSearchSystemForSTJ/
- https://www.SBERT.net
- https://github.com/rufimelo99
- https://www.inesc-id.pt/projects/PR07005/