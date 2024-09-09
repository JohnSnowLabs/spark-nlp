---
layout: model
title: Portuguese Legal BERT Sentence Embedding Large Cased model
author: John Snow Labs
name: sent_bert_bert_large_portuguese_cased_legal_tsdae_gpl_nli_sts_MetaKD_v1
date: 2024-09-04
tags: [bert, pt, embeddings, sentence, open_source, onnx]
task: Embeddings
language: pt
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

Pretrained Legal BERT Sentence Embedding model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `bert-large-portuguese-cased-legal-tsdae-gpl-nli-sts-MetaKD-v1` is a Portuguese model originally trained by `stjiris`.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sent_bert_bert_large_portuguese_cased_legal_tsdae_gpl_nli_sts_MetaKD_v1_pt_5.5.0_3.0_1725454256937.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/sent_bert_bert_large_portuguese_cased_legal_tsdae_gpl_nli_sts_MetaKD_v1_pt_5.5.0_3.0_1725454256937.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
sent_embeddings = BertSentenceEmbeddings.pretrained("sent_bert_bert_large_portuguese_cased_legal_tsdae_gpl_nli_sts_MetaKD_v1", "pt") \
  .setInputCols("sentence") \
  .setOutputCol("bert_sentence")

  nlp_pipeline = Pipeline(stages=[document_assembler, sentence_detector, sent_embeddings ])
    result = pipeline.fit(data).transform(data)
```
```scala
val sent_embeddings = BertSentenceEmbeddings.pretrained("sent_bert_bert_large_portuguese_cased_legal_tsdae_gpl_nli_sts_MetaKD_v1", "pt")
  .setInputCols("sentence")
  .setOutputCol("bert_sentence")

  val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, sent_embeddings ))
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sent_bert_bert_large_portuguese_cased_legal_tsdae_gpl_nli_sts_MetaKD_v1|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence]|
|Output Labels:|[embeddings]|
|Language:|pt|
|Size:|1.2 GB|

## References

References

- https://huggingface.co/stjiris/bert-large-portuguese-cased-legal-tsdae-gpl-nli-sts-MetaKD-v1
- https://github.com/rufimelo99/metadata-knowledge-distillation
- https://github.com/rufimelo99
- https://rufimelo99.github.io/SemanticSearchSystemForSTJ/
- https://www.SBERT.net
- https://www.inesc-id.pt/projects/PR07005/