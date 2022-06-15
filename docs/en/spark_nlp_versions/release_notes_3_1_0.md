---
layout: docs
header: true
seotitle: Spark NLP
title: Spark NLP release notes 3.1.0
permalink: /docs/en/spark_nlp_versions/release_notes_3_1_0
key: docs-release-notes
modify_date: "2022-01-06"
show_nav: true
sidebar:
    nav: sparknlp
---

### 3.1.0

#### John Snow Labs Spark-NLP 3.1.0: Over 2600+ new models and pipelines in 200+ languages, new DistilBERT, RoBERTa, and XLM-RoBERTa transformers, support for external Transformers, and lots more!

Overview

We are very excited to release Spark NLP 🚀  3.1.0! This is one of our biggest releases with lots of models, pipelines, and groundworks for future features that we are so proud to share it with our community.

Spark NLP 3.1.0 comes with over 2600+ new pretrained models and pipelines in over 200+ languages, new DistilBERT, RoBERTa, and XLM-RoBERTa annotators, support for HuggingFace 🤗 (Autoencoding) models in Spark NLP, and extends support for new Databricks and EMR instances.

As always, we would like to thank our community for their feedback, questions, and feature requests.

Major features and improvements

* **NEW:** Introducing DistiBertEmbeddings annotator. DistilBERT is a small, fast, cheap, and light Transformer model trained by distilling BERT base. It has 40% fewer parameters than `bert-base-uncased`, runs 60% faster while preserving over 95% of BERT’s performances
* **NEW:** Introducing RoBERTaEmbeddings annotator. RoBERTa (Robustly Optimized BERT-Pretraining Approach) models deliver state-of-the-art performance on NLP/NLU tasks and a sizable performance improvement on the GLUE benchmark. With a score of 88.5, RoBERTa reached the top position on the GLUE leaderboard
* **NEW:** Introducing XlmRoBERTaEmbeddings annotator. XLM-RoBERTa (Unsupervised Cross-lingual Representation Learning at Scale) is a large multi-lingual language model, trained on 2.5TB of filtered CommonCrawl data with 100 different languages. It also outperforms multilingual BERT (mBERT) on a variety of cross-lingual benchmarks, including +13.8% average accuracy on XNLI, +12.3% average F1 score on MLQA, and +2.1% average F1 score on NER. XLM-R performs particularly well on low-resource languages, improving 11.8% in XNLI accuracy for Swahili and 9.2% for Urdu over the previous XLM model
* **NEW:** Introducing support for HuggingFace exported models in equivalent Spark NLP annotators. Starting this release, you can easily use the `saved_model` feature in HuggingFace within a few lines of codes and import any BERT, DistilBERT, RoBERTa, and XLM-RoBERTa models to Spark NLP. We will work on the remaining annotators and extend this support to the rest with each release - For more information please visit [this discussion](https://github.com/JohnSnowLabs/spark-nlp/discussions/5669)
* **NEW:** Migrate MarianTransformer to BatchAnnotate to control the throughput when you are on accelerated hardware such as GPU to fully utilize it
* Upgrade to TensorFlow v2.4.1 with native support for Java to take advantage of many optimizations for CPU/GPU and new features/models introduced in TF v2.x
* Update to CUDA11 and cuDNN 8.0.2 for GPU support
* Implement ModelSignatureManager to automatically detect inputs, outputs, save and restore tensors from SavedModel in TF v2. This allows Spark NLP 3.1.x to extend support for external Encoders such as HuggingFace and TF Hub (coming soon!)
* Implement a new BPE tokenizer for RoBERTa and XLM models. This tokenizer will use the custom tokens from `Tokenizer` or `RegexTokenizer` and generates token pieces, encodes, and decodes the results
* Welcoming new Databricks runtimes to our Spark NLP family:
  * Databricks 8.1 ML & GPU
  * Databricks 8.2 ML & GPU
  * Databricks 8.3 ML & GPU
* Welcoming a new EMR 6.x series to our Spark NLP family:
  * EMR 6.3.0 (Apache Spark 3.1.1 / Hadoop 3.2.1)
 * Added examples to Spark NLP Scaladoc

**For more details please check the official [release notes](https://github.com/JohnSnowLabs/spark-nlp/releases/tag/3.1.0)**

<div class="prev_ver h3-box" markdown="1">

## Versions

</div>

<ul class="pagination">
    <li>
        <a href="release_notes_3_0_3">Version 3.0.3</a>
    </li>
    <li>
        <strong>Version 3.1.0</strong>
    </li>
    <li>
        <a href="release_notes_3_1_1">Version 3.1.1</a>
    </li>
</ul>

<ul class="pagination pagination_big">
  <li><a href="release_notes_3_4_0">3.4.0</a></li>
  <li><a href="release_notes_3_3_4">3.3.4</a></li>
  <li><a href="release_notes_3_3_3">3.3.3</a></li>
  <li><a href="release_notes_3_3_2">3.3.2</a></li>
  <li><a href="release_notes_3_3_1">3.3.1</a></li>
  <li><a href="release_notes_3_3_0">3.3.0</a></li>
  <li><a href="release_notes_3_2_3">3.2.3</a></li>
  <li><a href="release_notes_3_2_2">3.2.2</a></li>
  <li><a href="release_notes_3_2_1">3.2.1</a></li>
  <li><a href="release_notes_3_2_0">3.2.0</a></li>
  <li><a href="release_notes_3_1_3">3.1.3</a></li>
  <li><a href="release_notes_3_1_2">3.1.2</a></li>
  <li><a href="release_notes_3_1_1">3.1.1</a></li>
  <li class="active"><a href="release_notes_3_1_0">3.1.0</a></li>
  <li><a href="release_notes_3_0_3">3.0.3</a></li>
  <li><a href="release_notes_3_0_2">3.0.2</a></li>
  <li><a href="release_notes_3_0_1">3.0.1</a></li>
  <li><a href="release_notes_3_0_0">3.0.0</a></li>
  <li><a href="release_notes_2_7_5">2.7.5</a></li>
  <li><a href="release_notes_2_7_4">2.7.4</a></li>
  <li><a href="release_notes_2_7_3">2.7.3</a></li>
  <li><a href="release_notes_2_7_2">2.7.2</a></li>
  <li><a href="release_notes_2_7_1">2.7.1</a></li>
  <li><a href="release_notes_2_7_0">2.7.0</a></li>
  <li><a href="release_notes_2_6_5">2.6.5</a></li>
  <li><a href="release_notes_2_6_4">2.6.4</a></li>
  <li><a href="release_notes_2_6_3">2.6.3</a></li>
  <li><a href="release_notes_2_6_2">2.6.2</a></li>
  <li><a href="release_notes_2_6_1">2.6.1</a></li>
  <li><a href="release_notes_2_6_0">2.6.0</a></li>
  <li><a href="release_notes_2_5_5">2.5.5</a></li>
  <li><a href="release_notes_2_5_4">2.5.4</a></li>
  <li><a href="release_notes_2_5_3">2.5.3</a></li>
  <li><a href="release_notes_2_5_2">2.5.2</a></li>
  <li><a href="release_notes_2_5_1">2.5.1</a></li>
  <li><a href="release_notes_2_5_0">2.5.0</a></li>
  <li><a href="release_notes_2_4_5">2.4.5</a></li>
  <li><a href="release_notes_2_4_4">2.4.4</a></li>
  <li><a href="release_notes_2_4_3">2.4.3</a></li>
  <li><a href="release_notes_2_4_2">2.4.2</a></li>
  <li><a href="release_notes_2_4_1">2.4.1</a></li>
  <li><a href="release_notes_2_4_0">2.4.0</a></li>
</ul>