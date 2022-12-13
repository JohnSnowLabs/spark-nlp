---
layout: model
title: Chinese BERT with Whole Word Masking
author: John Snow Labs
name: chinese_bert_wwm
date: 2021-05-20
tags: [chinese, zh, embeddings, bert, open_source]
task: Embeddings
language: zh
edition: Spark NLP 3.1.0
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

**[Pre-Training with Whole Word Masking for Chinese BERT](https://arxiv.org/abs/1906.08101)**  Yiming Cui, Wanxiang Che, Ting Liu, Bing Qin, Ziqing Yang, Shijin Wang, Guoping Hu

More resources by HFL: https://github.com/ymcui/HFL-Anthology

If you find the technical report or resource is useful, please cite the following technical report in your paper.
- Primary: [https://arxiv.org/abs/2004.13922](https://arxiv.org/abs/2004.13922)
- Secondary: [https://arxiv.org/abs/1906.08101](https://arxiv.org/abs/1906.08101)

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/chinese_bert_wwm_zh_3.1.0_2.4_1621511963425.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/chinese_bert_wwm_zh_3.1.0_2.4_1621511963425.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
embeddings = BertEmbeddings.pretrained("chinese_bert_wwm", "zh") \
      .setInputCols("sentence", "token") \
      .setOutputCol("embeddings")
nlp_pipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, embeddings])

```
```scala
val embeddings = BertEmbeddings.pretrained("chinese_bert_wwm", "zh")
      .setInputCols("sentence", "token")
      .setOutputCol("embeddings")
val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, embeddings))
```


{:.nlu-block}
```python
import nlu
nlu.load("zh.embed.bert.wwm").predict("""Put your text here.""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|chinese_bert_wwm|
|Compatibility:|Spark NLP 3.1.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[token, sentence]|
|Output Labels:|[embeddings]|
|Language:|zh|
|Case sensitive:|true|

## Data Source

[https://huggingface.co/hfl/chinese-bert-wwm](https://huggingface.co/hfl/chinese-bert-wwm)

## Benchmarking

```bash
-	BERTGoogle	BERT-wwm	BERT-wwm-ext	RoBERTa-wwm-ext	RoBERTa-wwm-ext-large
Masking	WordPiece	WWM[1]	WWM	WWM	WWM
Type	base	base	base	base	large
Data Source	wiki	wiki	wiki+ext[2]	wiki+ext	wiki+ext
Training Tokens #	0.4B	0.4B	5.4B	5.4B	5.4B
Device	TPU Pod v2	TPU v3	TPU v3	TPU v3	TPU Pod v3-32[3]
Training Steps	?	100KMAX128
+100KMAX512	1MMAX128
+400KMAX512	1MMAX512	2MMAX512
Batch Size	?	2,560 / 384	2,560 / 384	384	512
Optimizer	AdamW	LAMB	LAMB	AdamW	AdamW
Vocabulary	21,128	~BERT[4]	~BERT	~BERT	~BERT
Init Checkpoint	Random Init	~BERT	~BERT	~BERT	Random Init
```