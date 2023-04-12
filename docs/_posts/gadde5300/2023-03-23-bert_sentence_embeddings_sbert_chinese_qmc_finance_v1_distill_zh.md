---
layout: model
title: Chinese Finance BERT Sentence Embedding Cased model
author: John Snow Labs
name: bert_sentence_embeddings_sbert_chinese_qmc_finance_v1_distill
date: 2023-03-23
tags: [bert, zh, embeddings, sentence, open_source, tensorflow]
task: Embeddings
language: zh
edition: Spark NLP 4.4.0
spark_version: 3.0
supported: true
engine: tensorflow
annotator: BertSentenceEmbeddings
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained Finance BERT Sentence Embedding model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `sbert-chinese-qmc-finance-v1-distill` is a Chinese model originally trained by `DMetaSoul`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_sentence_embeddings_sbert_chinese_qmc_finance_v1_distill_zh_4.3.2_3.0_1679546196565.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_sentence_embeddings_sbert_chinese_qmc_finance_v1_distill_zh_4.3.2_3.0_1679546196565.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
sent_embeddings = BertSentenceEmbeddings.pretrained("bert_sentence_embeddings_sbert_chinese_qmc_finance_v1_distill", "zh") \
  .setInputCols("sentence") \
  .setOutputCol("bert_sentence")

  nlp_pipeline = Pipeline(stages=[document_assembler, sentence_detector, sent_embeddings ])
    result = pipeline.fit(data).transform(data)
```
```scala
val sent_embeddings = BertSentenceEmbeddings.pretrained("bert_sentence_embeddings_sbert_chinese_qmc_finance_v1_distill", "zh")
  .setInputCols("sentence")
  .setOutputCol("bert_sentence")

  val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, sent_embeddings ))
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_sentence_embeddings_sbert_chinese_qmc_finance_v1_distill|
|Compatibility:|Spark NLP 4.3.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence]|
|Output Labels:|[bert_sentence]|
|Language:|zh|
|Size:|171.0 MB|
|Case sensitive:|true|

## References

- https://huggingface.co/DMetaSoul/sbert-chinese-qmc-finance-v1-distill
- https://www.SBERT.net
