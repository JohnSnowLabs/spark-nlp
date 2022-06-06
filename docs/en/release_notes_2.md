---
layout: docs
header: true
seotitle: Spark NLP
title: Spark NLP release notes
permalink: /docs/en/release_notes_2
key: docs-release-notes-2
modify_date: "2022-01-06"
show_nav: true
sidebar:
    nav: sparknlp
---

### 2.7.5

#### John Snow Labs Spark-NLP 2.7.5: Supporting more EMR versions and other improvements!

Overview

We are glad to release Spark NLP 2.7.5 release! Starting this release we no longer ship Hadoop AWS and AWS Java SDK dependencies. This change allows users to avoid any conflicts in AWS environments and also results in more EMR 5.x versions support.

As always, we would like to thank our community for their feedback, questions, and feature requests.

New Features

* Support more EMR 5.x versions
  * emr-5.20.0
  * emr-5.21.0
  * emr-5.21.1
  * emr-5.22.0
  * emr-5.23.0
  * emr-5.24.0
  * emr-5.24.1
  * emr-5.25.0
  * emr-5.26.0
  * emr-5.27.0
  * emr-5.28.0
  * emr-5.29.0
  * emr-5.30.0
  * emr-5.30.1
  * emr-5.31.0
  * emr-5.32.0

**For more details please check the official [release notes](https://github.com/JohnSnowLabs/spark-nlp/releases/tag/2.7.5)**

### 2.7.4

#### John Snow Labs Spark-NLP 2.7.4: New Bengali NER and Word Embeddings models, new Intent Prediction models, bug fixes, and other improvements!

**For more details please check the official [release notes](https://github.com/JohnSnowLabs/spark-nlp/releases/tag/2.7.4)**

### 2.7.3

#### John Snow Labs Spark-NLP 2.7.3: 18 new state-of-the-art transformer-based OntoNotes models and pipelines, new support for Bengali NER and Hindi Word Embeddings, and other improvements!

**For more details please check the official [release notes](https://github.com/JohnSnowLabs/spark-nlp/releases/tag/2.7.3)**

### 2.7.2

#### John Snow Labs Spark-NLP 2.7.2: New multilingual models, GPU support to train a Spell Checker, bug fixes, and other improvements!

**For more details please check the official [release notes](https://github.com/JohnSnowLabs/spark-nlp/releases/tag/2.7.2)**

### 2.7.1

#### John Snow Labs Spark-NLP 2.7.1: New T5 models, new TREC pipelines, bug fixes, and other improvements!

**For more details please check the official [release notes](https://github.com/JohnSnowLabs/spark-nlp/releases/tag/2.7.1)**

### 2.7.0

#### John Snow Labs Spark-NLP 2.7.0: New T5 and MarianMT seq2seq transformers, detect up to 375 languages, word segmentation, over 720+ models and pipelines, support for 192+ languages, and many more!

Overview

We are very excited to release Spark NLP 2.7.0! This has been one of the biggest releases we have ever done that we are so proud to share it with our community!

In this release, we are bringing support to state-of-the-art Seq2Seq and Text2Text transformers. We have developed annotators for Google T5 (Text-To-Text Transfer Transformer) and MarianMNT for Neural Machine Translation with over 646 pretrained models and pipelines.

This release also comes with a refactored and brand new models for language detection and identification. They are more accurate, faster, and support up to 375 languages.

The 2.7.0 release has over 720+ new pretrained models and pipelines while extending our support of multi-lingual models to 192+ languages such as Chinese, Japanese, Korean, Arabic, Persian, Urdu, and Hebrew.

As always, we would like to thank our community for their feedback and support.

Major features and improvements

* **NEW:** Introducing MarianTransformer annotator for machine translation based on MarianNMT models. Marian is an efficient, free Neural Machine Translation framework mainly being developed by the Microsoft Translator team (646+ pretrained models & pipelines in 192+ languages)
* **NEW:** Introducing T5Transformer annotator for Text-To-Text Transfer Transformer (Google T5) models to achieve state-of-the-art results on multiple NLP tasks such as Translation, Summarization, Question Answering, Sentence Similarity, and so on
* **NEW:** Introducing brand new and refactored language detection and identification models. The new LanguageDetectorDL is faster, more accurate, and supports up to 375 languages
* **NEW:** Introducing WordSegmenter annotator, a trainable annotator for word segmentation of languages without any rule-based tokenization such as Chinese, Japanese, or Korean
* **NEW:** Introducing DocumentNormalizer annotator cleaning content from HTML or XML documents, applying either data cleansing using an arbitrary number of custom regular expressions either data extraction following the different parameters
* **NEW:** [Spark NLP Display](https://github.com/JohnSnowLabs/spark-nlp-display) for visualization of different types of annotations
* Add support for new multi-lingual models in UniversalSentenceEncoder annotator
* Add support to Lemmatizer to be trained directly from a DataFrame instead of a text file
* Add training helper to transform CoNLL-U into Spark NLP annotator type columns

**For more details please check the official [release notes](https://github.com/JohnSnowLabs/spark-nlp/releases/tag/2.7.0)**

### 2.6.5

#### John Snow Labs Spark-NLP 2.6.5: A few bug fixes and other improvements!

**For more details please check the official [release notes](https://github.com/JohnSnowLabs/spark-nlp/releases/tag/2.6.5)**

### 2.6.4

#### John Snow Labs Spark-NLP 2.6.4: A few bug fixes and other improvements!

**For more details please check the official [release notes](https://github.com/JohnSnowLabs/spark-nlp/releases/tag/2.6.4)**

### 2.6.3

#### John Snow Labs Spark-NLP 2.6.3: New refactored NerDL with memory optimization, bug fixes, and other improvements!

**For more details please check the official [release notes](https://github.com/JohnSnowLabs/spark-nlp/releases/tag/2.6.3)**

### 2.6.2

#### John Snow Labs Spark-NLP 2.6.2: New SentenceDetectorDL, improved BioBERT models, new Models Hub, and other improvements!

**For more details please check the official [release notes](https://github.com/JohnSnowLabs/spark-nlp/releases/tag/2.6.2)**

### 2.6.1

#### John Snow Labs Spark-NLP 2.6.1: New Portuguese BERT models, import any BERT models to Spark NLP, and a bug-fix for ClassifierDL

**For more details please check the official [release notes](https://github.com/JohnSnowLabs/spark-nlp/releases/tag/2.6.1)**

### 2.6.0

#### John Snow Labs Spark-NLP 2.6.0: New multi-label classifier, BERT sentence embeddings, unsupervised keyword extractions, over 110 pretrained pipelines, models, Transformers, and more!

Overview

We are very excited to finally release Spark NLP 2.6.0! This has been one of the biggest releases we have ever made and we are so proud to share it with our community!

This release comes with a brand new MultiClassifierDL for multi-label text classification, BertSentenceEmbeddings with 42 models, unsupervised keyword extractions annotator, and adding 28 new pretrained Transformers such as Small BERT, CovidBERT, ELECTRA, and the state-of-the-art language-agnostic BERT Sentence Embedding model(LaBSE).

The 2.6.0 release has over 110 new pretrained models, pipelines, and Transformers with extending full support for Danish, Finnish, and Swedish languages.

Major features and improvements

* **NEW:** A new MultiClassifierDL annotator for multi-label text classification built by using Bidirectional GRU and CNN inside TensorFlow that supports up to 100 classes
* **NEW:** A new BertSentenceEmbeddings annotator with 42 available pre-trained models for sentence embeddings used in SentimentDL, ClassifierDL, and MultiClassifierDL annotators
* **NEW:** A new YakeModel annotator for an unsupervised, corpus-independent, domain, and language-independent and single-document keyword extraction algorithm
* **NEW:** Integrate 24 new Small BERT models where the smallest model is 24x times smaller and 28x times faster compare to BERT base models
* **NEW:** Add 3 new ELECTRA small, base, and large models
* **NEW:** Add 4 new Finnish BERT models for BertEmbeddings and BertSentenceEmbeddings
* Improve BertEmbeddings memory consumption by 30%
* Improve BertEmbeddings performance by more than 70% with a new built-in dynamic shape inputs
* Remove the poolingLayer parameter in BertEmbeddings in favor of sequence_output that is provided by TF Hub models for new BERT models
* Add validation loss, validation accuracy, validation F1, and validation True Positive Rate during the training in MultiClassifierDL
* Add parameter to enable/disable list detection in SentenceDetector
* Unify the loggings in ClassifierDL and SentimentDL during training

**For more details please check the official [release notes](https://github.com/JohnSnowLabs/spark-nlp/releases/tag/2.6.0)**

### 2.5.5

#### John Snow Labs Spark-NLP 2.5.5: 28 new Lemma and POS models in 14 languages, bug fixes, and lots of new notebooks!

**For more details please check the official [release notes](https://github.com/JohnSnowLabs/spark-nlp/releases/tag/2.5.5)**

### 2.5.4

#### John Snow Labs Spark-NLP 2.5.4: Supporting Apache Spark 2.3, 43 new models and 26 new languages, new RegexTokenizer, lots of new notebooks, and more

**For more details please check the official [release notes](https://github.com/JohnSnowLabs/spark-nlp/releases/tag/2.5.4)**

### 2.5.3

#### John Snow Labs Spark-NLP 2.5.3: Detect Fake news, emotions, spams, and more classification models, enhancements, and bug fixes

**For more details please check the official [release notes](https://github.com/JohnSnowLabs/spark-nlp/releases/tag/2.5.3)**

### 2.5.2

#### John Snow Labs Spark-NLP 2.5.2: New Language Detection annotator, enhancements, and bug fixes

**For more details please check the official [release notes](https://github.com/JohnSnowLabs/spark-nlp/releases/tag/2.5.2)**

### 2.5.1

#### John Snow Labs Spark-NLP 2.5.1: Adding support for 6 new BioBERT and ClinicalBERT models

**For more details please check the official [release notes](https://github.com/JohnSnowLabs/spark-nlp/releases/tag/2.5.1)**

### 2.5.0

#### John Snow Labs Spark-NLP 2.5.0: ALBERT & XLNet transformers, state-of-the-art spell checker, multi-class sentiment detector, 80+ new models & pipelines in 14 new languages & more

Overview

When we started planning for Spark NLP 2.5.0 release a few months ago the world was a different place!

We have been blown away by the use of Natural Language Processing for early outbreak detections, question-answering chatbot services, text analysis of medical records, monitoring efforts to minimize the virus spread, and many more.

In that spirit, we are honored to announce Spark NLP 2.5.0 release! Witnessing the world coming together to fight coronavirus has driven us to deliver perhaps one of the biggest releases we have ever made.

As always, we thank our community for their feedback, bug reports, and contributions that made this release possible.

Major features and improvements

* **NEW:** A new AlbertEmbeddings annotator with 4 available pre-trained models
* **NEW:** A new XlnetEmbeddings annotator with 2 available pre-trained models
* **NEW:** A new ContextSpellChecker annotator, the state-of-the-art annotator for spell checking
* **NEW:** A new SentimentDL annotator for multi-class sentiment analysis. This annotator comes with 2 available pre-trained models trained on IMDB and Twitter datasets
* **NEW:** Support for 14 new languages with 80+ pretrained models and pipelines!
* Add new PubTator reader to convert automatic annotations of the biomedical datasets into DataFrame
* Introducing a new outputLogsPath param for NerDLApproach, ClassifierDLApproach and SentimentDLApproach annotators
* Refactored CoNLLGenerator to actually use NER labels from the DataFrame
* Unified params in NerDLModel in both Scala and Python
* Extend and complete Scaladoc APIs for all the annotators

**For more details please check the official [release notes](https://github.com/JohnSnowLabs/spark-nlp/releases/tag/2.5.0)**

### 2.4.5

#### John Snow Labs Spark-NLP 2.4.5: Supporting more Databricks runtimes and YARN in cluster mode

**For more details please check the official [release notes](https://github.com/JohnSnowLabs/spark-nlp/releases/tag/2.4.5)**

### 2.4.4

#### John Snow Labs Spark-NLP 2.4.4: The very first native multi-class text classifier and pre-trained models and pipelines in Russian

**For more details please check the official [release notes](https://github.com/JohnSnowLabs/spark-nlp/releases/tag/2.4.4)**

### 2.4.3

#### John Snow Labs Spark-NLP 2.4.3: Minor bug fix in Python

**For more details please check the official [release notes](https://github.com/JohnSnowLabs/spark-nlp/releases/tag/2.4.3)**

### 2.4.2

#### John Snow Labs Spark-NLP 2.4.2: Minor bug fixes and improvements

**For more details please check the official [release notes](https://github.com/JohnSnowLabs/spark-nlp/releases/tag/2.4.2)**

### 2.4.1

#### John Snow Labs Spark-NLP 2.4.1: Bug fixes and the very first Spanish models & pipelines

**For more details please check the official [release notes](https://github.com/JohnSnowLabs/spark-nlp/releases/tag/2.4.1)**

### 2.4.0

#### John Snow Labs Spark-NLP 2.4.0: New TensorFlow 1.15, Universal Sentence Encoder, Elmo, faster Word Embeddings & more

We are very excited to finally release Spark NLP v2.4.0! This has been one of the largest releases we have ever made since the inception of the library! The new release of Spark NLP `2.4.0` has been migrated to TensorFlow `1.15.0` which takes advantage of the latest deep learning technologies and pre-trained models.

Major features and improvements

* **NEW:** TensorFlow 1.15.0 now works behind Spark NLP. This brings implicit improvements in performance, accuracy, and functionalities
* **NEW:** UniversalSentenceEncoder annotator with 2 pre-trained models from TF Hub
* **NEW:** ElmoEmbeddings with a pre-trained model from TF Hub
* **NEW:** All our pre-trained models are now cross-platform!
* **NEW:** For the first time, all the multi-lingual models and pipelines are available for Windows users (French, German and Italian)
* **NEW:** MultiDateMatcher capable of matching more than one date per sentence (Extends DateMatcher algorithm)
* **NEW:** BigTextMatcher works best with large amounts of input data
* BertEmbeddings improvements with 5 new models from TF Hub
* RecursivePipelineModel as an enhanced PipelineModel allows Annotators to access previous annotators in the pipeline for more ML strategies
* LazyAnnotators: A new Param in Annotators allows them to stand idle in the Pipeline and do nothing. Can be called by other Annotators in a RecursivePipeline
* RocksDB is now available as a flexible API called `Storage`. Allows any annotator to have it's own distributed local index database
* Now our Tensorflow pre-trained models are cross-platform. Enabling multi-language models and other improvements to Windows users.
* Improved IO performance in general for handling embeddings
* Improved cache cleanup and GC by liberating open files utilized in RocksDB (to be improved further)
* Tokenizer and SentenceDetector Params minLength and MaxLength to filter out annotations outside these bounds
* Tokenizer improvements in splitChars and simplified rules
* DateMatcher improvements
* TextMatcher improvements preload algorithm information within the model for faster prediction
* Annotators the utilize embeddings have now a strict validation to be using exactly the embeddings they were trained with
* Improvements in the API allow Annotators with Storage to save and load their RocksDB database independently and let it be shared across Annotators and let it be shared across Annotators

<div class="prev_ver h3-box" markdown="1">

## Next versions

</div>
<ul class="pagination">
    <li>
        <strong>Versions 2.0.0</strong>
    </li>
    <li>
        <a href="release_notes">Versions 3.0.0</a>
    </li>
</ul>