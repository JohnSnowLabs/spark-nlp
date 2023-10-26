---
layout: docs
header: true
seotitle: Spark NLP - Annotators
title: Spark NLP - Annotators
permalink: /docs/en/annotators
key: docs-annotators
modify_date: "2021-04-17"
show_nav: true
sidebar:
    nav: sparknlp
---

<div class="h3-box" markdown="1">

## How to read this section

All annotators in Spark NLP share a common interface, this is:

- **Annotation**: `Annotation(annotatorType, begin, end, result, meta-data,
embeddings)`
- **AnnotatorType**: some annotators share a type. This is not only
figurative, but also tells about the structure of the `metadata` map in
the Annotation. This is the one referred in the input and output of
annotators.
- **Inputs**: Represents how many and which annotator types are expected
in `setInputCols()`. These are column names of output of other annotators
in the DataFrames.
- **Output** Represents the type of the output in the column
`setOutputCol()`.

There are two types of Annotators:

- **Approach**: AnnotatorApproach extend Estimators, which are meant to be trained through `fit()`
- **Model**: AnnotatorModel extend from Transformers, which are meant to transform DataFrames through `transform()`

> **`Model`** suffix is explicitly stated when the annotator is the result of a training process. Some annotators, such as ***Tokenizer*** are transformers, but do not contain the word Model since they are not trained annotators.

`Model` annotators have a `pretrained()` on it's static object, to retrieve the public pre-trained version of a model.

- `pretrained(name, language, extra_location)` -> by default, pre-trained will bring a default model, sometimes we offer more than one model, in this case, you may have to use name, language or extra location to download them.

## Available Annotators

{:.table-model-big}
|Annotator|Description|Version |
|---|---|---|
{% include templates/anno_table_entry.md path="" name="BigTextMatcher" summary="Annotator to match exact phrases (by token) provided in a file against a Document."%}
{% include templates/anno_table_entry.md path="" name="Chunk2Doc" summary="Converts a `CHUNK` type column back into `DOCUMENT`. Useful when trying to re-tokenize or do further analysis on a `CHUNK` result."%}
{% include templates/anno_table_entry.md path="" name="ChunkEmbeddings" summary="This annotator utilizes WordEmbeddings, BertEmbeddings etc. to generate chunk embeddings from either Chunker, NGramGenerator, or NerConverter outputs."%}
{% include templates/anno_table_entry.md path="" name="ChunkTokenizer" summary="Tokenizes and flattens extracted NER chunks."%}
{% include templates/anno_table_entry.md path="" name="Chunker" summary="This annotator matches a pattern of part-of-speech tags in order to return meaningful phrases from document."%}
{% include templates/anno_table_entry.md path="" name="ClassifierDL" summary="ClassifierDL for generic Multi-class Text Classification."%}
{% include templates/anno_table_entry.md path="" name="ContextSpellChecker" summary="Implements a deep-learning based Noisy Channel Model Spell Algorithm."%}
{% include templates/anno_table_entry.md path="" name="Date2Chunk" summary="Converts `DATE` type Annotations to `CHUNK` type."%}
{% include templates/anno_table_entry.md path="" name="DateMatcher" summary="Matches standard date formats into a provided format."%}
{% include templates/anno_table_entry.md path="" name="DependencyParser" summary="Unlabeled parser that finds a grammatical relation between two words in a sentence."%}
{% include templates/anno_table_entry.md path="" name="Doc2Chunk" summary="Converts `DOCUMENT` type annotations into `CHUNK` type with the contents of a `chunkCol`."%}
{% include templates/anno_table_entry.md path="" name="Doc2Vec" summary="Word2Vec model that creates vector representations of words in a text corpus."%}
{% include templates/anno_table_entry.md path="" name="DocumentAssembler" summary="Prepares data into a format that is processable by Spark NLP. This is the entry point for every Spark NLP pipeline."%}
{% include templates/anno_table_entry.md path="" name="DocumentCharacterTextSplitter" summary="Annotator which splits large documents into chunks of roughly given size."%}
{% include templates/anno_table_entry.md path="" name="DocumentNormalizer" summary="Annotator which normalizes raw text from tagged text, e.g. scraped web pages or xml documents, from document type columns into Sentence."%}
{% include templates/anno_table_entry.md path="" name="DocumentSimilarityRanker" summary="Annotator that uses LSH techniques present in Spark ML lib to execute approximate nearest neighbors search on top of sentence embeddings."%}
{% include templates/anno_table_entry.md path="" name="EntityRuler" summary="Fits an Annotator to match exact strings or regex patterns provided in a file against a Document and assigns them an named entity."%}
{% include templates/anno_table_entry.md path="" name="EmbeddingsFinisher" summary="Extracts embeddings from Annotations into a more easily usable form."%}
{% include templates/anno_table_entry.md path="" name="Finisher" summary="Converts annotation results into a format that easier to use. It is useful to extract the results from Spark NLP Pipelines."%}
{% include templates/anno_table_entry.md path="" name="GraphExtraction" summary="Extracts a dependency graph between entities."%}
{% include templates/anno_table_entry.md path="" name="GraphFinisher" summary="Helper class to convert the knowledge graph from GraphExtraction into a generic format, such as RDF."%}
{% include templates/anno_table_entry.md path="" name="ImageAssembler" summary="Prepares images read by Spark into a format that is processable by Spark NLP."%}
{% include templates/anno_table_entry.md path="" name="LanguageDetectorDL" summary="Language Identification and Detection by using CNN and RNN architectures in TensorFlow."%}
{% include templates/anno_table_entry.md path="" name="Lemmatizer" summary="Finds lemmas out of words with the objective of returning a base dictionary word."%}
{% include templates/anno_table_entry.md path="" name="MultiClassifierDL" summary="Multi-label Text Classification."%}
{% include templates/anno_table_entry.md path="" name="MultiDateMatcher" summary="Matches standard date formats into a provided format."%}
{% include templates/anno_table_entry.md path="" name="MultiDocumentAssembler" summary="Prepares data into a format that is processable by Spark NLP."%}
{% include templates/anno_table_entry.md path="" name="NGramGenerator" summary="A feature transformer that converts the input array of strings (annotatorType TOKEN) into an array of n-grams (annotatorType CHUNK)."%}
{% include templates/anno_table_entry.md path="" name="NerConverter" summary="Converts a IOB or IOB2 representation of NER to a user-friendly one, by associating the tokens of recognized entities and their label."%}
{% include templates/anno_table_entry.md path="" name="NerCrf" summary="Extracts Named Entities based on a CRF Model."%}
{% include templates/anno_table_entry.md path="" name="NerDL" summary="This Named Entity recognition annotator is a generic NER model based on Neural Networks."%}
{% include templates/anno_table_entry.md path="" name="NerOverwriter" summary="Overwrites entities of specified strings."%}
{% include templates/anno_table_entry.md path="" name="Normalizer" summary="Removes all dirty characters from text following a regex pattern and transforms words based on a provided dictionary."%}
{% include templates/anno_table_entry.md path="" name="NorvigSweeting Spellchecker" summary="Retrieves tokens and makes corrections automatically if not found in an English dictionary."%}
{% include templates/anno_table_entry.md path="" name="POSTagger (Part of speech tagger)" summary="Averaged Perceptron model to tag words part-of-speech."%}
{% include templates/anno_table_entry.md path="" name="RecursiveTokenizer" summary="Tokenizes raw text recursively based on a handful of definable rules."%}
{% include templates/anno_table_entry.md path="" name="RegexMatcher" summary="Uses rules to match a set of regular expressions and associate them with a provided identifier."%}
{% include templates/anno_table_entry.md path="" name="RegexTokenizer" summary="A tokenizer that splits text by a regex pattern."%}
{% include templates/anno_table_entry.md path="" name="SentenceDetector" summary="Annotator that detects sentence boundaries using regular expressions."%}
{% include templates/anno_table_entry.md path="" name="SentenceDetectorDL" summary="Detects sentence boundaries using a deep learning approach."%}
{% include templates/anno_table_entry.md path="" name="SentenceEmbeddings" summary="Converts the results from WordEmbeddings, BertEmbeddings, or ElmoEmbeddings into sentence or document embeddings by either summing up or averaging all the word embeddings in a sentence or a document (depending on the inputCols)."%}
{% include templates/anno_table_entry.md path="" name="SentimentDL" summary="Annotator for multi-class sentiment analysis."%}
{% include templates/anno_table_entry.md path="" name="SentimentDetector" summary="Rule based sentiment detector, which calculates a score based on predefined keywords."%}
{% include templates/anno_table_entry.md path="" name="Stemmer" summary="Returns hard-stems out of words with the objective of retrieving the meaningful part of the word."%}
{% include templates/anno_table_entry.md path="" name="StopWordsCleaner" summary="This annotator takes a sequence of strings (e.g. the output of a Tokenizer, Normalizer, Lemmatizer, and Stemmer) and drops all the stop words from the input sequences."%}
{% include templates/anno_table_entry.md path="" name="SymmetricDelete Spellchecker" summary="Symmetric Delete spelling correction algorithm."%}
{% include templates/anno_table_entry.md path="" name="TextMatcher" summary="Matches exact phrases (by token) provided in a file against a Document."%}
{% include templates/anno_table_entry.md path="" name="Token2Chunk" summary="Converts `TOKEN` type Annotations to `CHUNK` type."%}
{% include templates/anno_table_entry.md path="" name="TokenAssembler" summary="This transformer reconstructs a DOCUMENT type annotation from tokens, usually after these have been normalized, lemmatized, normalized, spell checked, etc, in order to use this document annotation in further annotators."%}
{% include templates/anno_table_entry.md path="" name="Tokenizer" summary="Tokenizes raw text into word pieces, tokens. Identifies tokens with tokenization open standards. A few rules will help customizing it if defaults do not fit user needs."%}
{% include templates/anno_table_entry.md path="" name="TypedDependencyParser" summary="Labeled parser that finds a grammatical relation between two words in a sentence."%}
{% include templates/anno_table_entry.md path="" name="ViveknSentiment" summary="Sentiment analyser inspired by the algorithm by Vivek Narayanan."%}
{% include templates/anno_table_entry.md path="" name="WordEmbeddings" summary="Word Embeddings lookup annotator that maps tokens to vectors."%}
{% include templates/anno_table_entry.md path="" name="Word2Vec" summary="Word2Vec model that creates vector representations of words in a text corpus."%}
{% include templates/anno_table_entry.md path="" name="WordSegmenter" summary="Tokenizes non-english or non-whitespace separated texts."%}
{% include templates/anno_table_entry.md path="" name="YakeKeywordExtraction" summary="Unsupervised, Corpus-Independent, Domain and Language-Independent and Single-Document keyword extraction."%}

## Available Transformers

Additionally, these transformers are available.

{:.table-model-big}
|Transformer|Description|Version|
|---|---|---|
{% include templates/anno_table_entry.md path="./transformers" name="AlbertEmbeddings" summary="ALBERT: A Lite BERT for Self-supervised Learning of Language Representations"%}
{% include templates/anno_table_entry.md path="./transformers" name="AlbertForQuestionAnswering" summary="AlbertForQuestionAnswering can load ALBERT Models with a span classification head on top for extractive question-answering tasks like SQuAD."%}
{% include templates/anno_table_entry.md path="./transformers" name="AlbertForTokenClassification" summary="AlbertForTokenClassification can load ALBERT Models with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for Named-Entity-Recognition (NER) tasks."%}
{% include templates/anno_table_entry.md path="./transformers" name="AlbertForSequenceClassification" summary="AlbertForSequenceClassification can load ALBERT Models with sequence classification/regression head on top e.g. for multi-class document classification tasks."%}
{% include templates/anno_table_entry.md path="./transformers" name="BartForZeroShotClassification" summary="BartForZeroShotClassification using a `ModelForSequenceClassification` trained on NLI (natural language inference) tasks."%}
{% include templates/anno_table_entry.md path="./transformers" name="BartTransformer" summary="BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension Transformer"%}
{% include templates/anno_table_entry.md path="./transformers" name="BertForQuestionAnswering" summary="BertForQuestionAnswering can load Bert Models with a span classification head on top for extractive question-answering tasks like SQuAD."%}
{% include templates/anno_table_entry.md path="./transformers" name="BertForSequenceClassification" summary="Bert Models with sequence classification/regression head on top."%}
{% include templates/anno_table_entry.md path="./transformers" name="BertForTokenClassification" summary="BertForTokenClassification can load Bert Models with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for Named-Entity-Recognition (NER) tasks."%}
{% include templates/anno_table_entry.md path="./transformers" name="BertForZeroShotClassification" summary="BertForZeroShotClassification using a ModelForSequenceClassification trained on NLI (natural language inference) tasks."%}
{% include templates/anno_table_entry.md path="./transformers" name="BertSentenceEmbeddings" summary="Sentence-level embeddings using BERT. BERT (Bidirectional Encoder Representations from Transformers) provides dense vector representations for natural language by using a deep, pre-trained neural network with the Transformer architecture."%}
{% include templates/anno_table_entry.md path="./transformers" name="CamemBertEmbeddings" summary="CamemBert is based on Facebook’s RoBERTa model released in 2019."%}
{% include templates/anno_table_entry.md path="./transformers" name="CamemBertForQuestionAnswering" summary="CamemBertForQuestionAnswering can load CamemBERT Models with a span classification head on top for extractive question-answering tasks like SQuAD"%}
{% include templates/anno_table_entry.md path="./transformers" name="CamemBertForSequenceClassification" summary="amemBertForSequenceClassification can load CamemBERT Models with sequence classification/regression head on top (a linear layer on top of the pooled output) e.g. for multi-class document classification tasks."%}
{% include templates/anno_table_entry.md path="./transformers" name="CamemBertForTokenClassification" summary="CamemBertForTokenClassification can load CamemBERT Models with a token classification head on top"%}
{% include templates/anno_table_entry.md path="./transformers" name="ConvNextForImageClassification" summary="ConvNextForImageClassification is an image classifier based on ConvNet models"%}
{% include templates/anno_table_entry.md path="./transformers" name="DeBertaEmbeddings" summary="DeBERTa builds on RoBERTa with disentangled attention and enhanced mask decoder training with half of the data used in RoBERTa."%}
{% include templates/anno_table_entry.md path="./transformers" name="DeBertaForQuestionAnswering" summary="DeBertaForQuestionAnswering can load DeBERTa Models with a span classification head on top for extractive question-answering tasks like SQuAD."%}
{% include templates/anno_table_entry.md path="./transformers" name="DeBertaForSequenceClassification" summary="DeBertaForSequenceClassification can load DeBerta v2 & v3 Models with sequence classification/regression head on top."%}
{% include templates/anno_table_entry.md path="./transformers" name="DeBertaForTokenClassification" summary="DeBertaForTokenClassification can load DeBERTA Models v2 and v3 with a token classification head on top."%}
{% include templates/anno_table_entry.md path="./transformers" name="DistilBertEmbeddings" summary="DistilBERT is a small, fast, cheap and light Transformer model trained by distilling BERT base."%}
{% include templates/anno_table_entry.md path="./transformers" name="DistilBertForQuestionAnswering" summary="DistilBertForQuestionAnswering can load DistilBert Models with a span classification head on top for extractive question-answering tasks like SQuAD."%}
{% include templates/anno_table_entry.md path="./transformers" name="DistilBertForSequenceClassification" summary="DistilBertForSequenceClassification can load DistilBERT Models with sequence classification/regression head on top (a linear layer on top of the pooled output) e.g. for multi-class document classification tasks."%}
{% include templates/anno_table_entry.md path="./transformers" name="DistilBertForTokenClassification" summary="DistilBertForTokenClassification can load DistilBERT Models with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for Named-Entity-Recognition (NER) tasks."%}
{% include templates/anno_table_entry.md path="./transformers" name="DistilBertForZeroShotClassification" summary="DistilBertForZeroShotClassification using a ModelForSequenceClassification trained on NLI (natural language inference) tasks."%}
{% include templates/anno_table_entry.md path="./transformers" name="E5Embeddings" summary="Sentence embeddings using E5, an instruction-finetuned text embedding model that can generate text embeddings tailored to any task."%}
{% include templates/anno_table_entry.md path="./transformers" name="ElmoEmbeddings" summary="Word embeddings from ELMo (Embeddings from Language Models), a language model trained on the 1 Billion Word Benchmark."%}
{% include templates/anno_table_entry.md path="./transformers" name="GPT2Transformer" summary="GPT-2 is a large transformer-based language model with 1.5 billion parameters, trained on a dataset of 8 million web pages."%}
{% include templates/anno_table_entry.md path="./transformers" name="HubertForCTC" summary="Hubert Model with a language modeling head on top for Connectionist Temporal Classification (CTC)."%}
{% include templates/anno_table_entry.md path="./transformers" name="InstructorEmbeddings" summary="Sentence embeddings using INSTRUCTOR."%}
{% include templates/anno_table_entry.md path="./transformers" name="LongformerEmbeddings" summary="Longformer is a BERT-like model started from the RoBERTa checkpoint and pretrained for MLM on long documents."%}
{% include templates/anno_table_entry.md path="./transformers" name="LongformerForQuestionAnswering" summary="LongformerForQuestionAnswering can load Longformer Models with a span classification head on top for extractive question-answering tasks like SQuAD."%}
{% include templates/anno_table_entry.md path="./transformers" name="LongformerForSequenceClassification" summary="LongformerForSequenceClassification can load Longformer Models with sequence classification/regression head on top e.g. for multi-class document classification tasks."%}
{% include templates/anno_table_entry.md path="./transformers" name="LongformerForTokenClassification" summary="LongformerForTokenClassification can load Longformer Models with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for Named-Entity-Recognition (NER) tasks."%}
{% include templates/anno_table_entry.md path="./transformers" name="MarianTransformer" summary="Marian is an efficient, free Neural Machine Translation framework written in pure C++ with minimal dependencies."%}
{% include templates/anno_table_entry.md path="./transformers" name="MPNetEmbeddings" summary="Sentence embeddings using MPNet."%}
{% include templates/anno_table_entry.md path="./transformers" name="OpenAICompletion" summary="Transformer that makes a request for OpenAI Completion API for each executor."%}
{% include templates/anno_table_entry.md path="./transformers" name="RoBertaEmbeddings" summary="RoBERTa: A Robustly Optimized BERT Pretraining Approach"%}
{% include templates/anno_table_entry.md path="./transformers" name="RoBertaForQuestionAnswering" summary="RoBertaForQuestionAnswering can load RoBERTa Models with a span classification head on top for extractive question-answering tasks like SQuAD."%}
{% include templates/anno_table_entry.md path="./transformers" name="RoBertaForSequenceClassification" summary="RoBertaForSequenceClassification can load RoBERTa Models with sequence classification/regression head on top e.g. for multi-class document classification tasks."%}
{% include templates/anno_table_entry.md path="./transformers" name="RoBertaForTokenClassification" summary="RoBertaForTokenClassification can load RoBERTa Models with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for Named-Entity-Recognition (NER) tasks."%}
{% include templates/anno_table_entry.md path="./transformers" name="RoBertaForZeroShotClassification" summary="RoBertaForZeroShotClassification using a `ModelForSequenceClassification` trained on NLI (natural language inference) tasks."%}
{% include templates/anno_table_entry.md path="./transformers" name="RoBertaForZeroShotClassification" summary="RoBertaForZeroShotClassification using a ModelForSequenceClassification trained on NLI (natural language inference) tasks."%}
{% include templates/anno_table_entry.md path="./transformers" name="RoBertaSentenceEmbeddings" summary="Sentence-level embeddings using RoBERTa."%}
{% include templates/anno_table_entry.md path="./transformers" name="SpanBertCoref" summary="A coreference resolution model based on SpanBert."%}
{% include templates/anno_table_entry.md path="./transformers" name="SwinForImageClassification" summary="SwinImageClassification is an image classifier based on Swin."%}
{% include templates/anno_table_entry.md path="./transformers" name="T5Transformer" summary="T5 reconsiders all NLP tasks into a unified text-to-text-format where the input and output are always text strings, in contrast to BERT-style models that can only output either a class label or a span of the input."%}
{% include templates/anno_table_entry.md path="./transformers" name="TapasForQuestionAnswering" summary="TapasForQuestionAnswering is an implementation of TaPas - a BERT-based model specifically designed for answering questions about tabular data."%}
{% include templates/anno_table_entry.md path="./transformers" name="UniversalSentenceEncoder" summary="The Universal Sentence Encoder encodes text into high dimensional vectors that can be used for text classification, semantic similarity, clustering and other natural language tasks."%}
{% include templates/anno_table_entry.md path="./transformers" name="VisionEncoderDecoderForImageCaptioning" summary="VisionEncoderDecoder model that converts images into text captions."%}
{% include templates/anno_table_entry.md path="./transformers" name="ViTForImageClassification" summary="Vision Transformer (ViT) for image classification."%}
{% include templates/anno_table_entry.md path="./transformers" name="Wav2Vec2ForCTC" summary="Wav2Vec2 Model with a language modeling head on top for Connectionist Temporal Classification (CTC)."%}
{% include templates/anno_table_entry.md path="./transformers" name="WhisperForCTC" summary="Whisper Model with a language modeling head on top for Connectionist Temporal Classification (CTC)."%}
{% include templates/anno_table_entry.md path="./transformers" name="XlmRoBertaEmbeddings" summary="XlmRoBerta is a large multi-lingual language model, trained on 2.5TB of filtered CommonCrawl"%}
{% include templates/anno_table_entry.md path="./transformers" name="XlmRoBertaForQuestionAnswering" summary="XlmRoBertaForQuestionAnswering can load XLM-RoBERTa Models with a span classification head on top for extractive question-answering tasks like SQuAD."%}
{% include templates/anno_table_entry.md path="./transformers" name="XlmRoBertaForSequenceClassification" summary="XlmRoBertaForSequenceClassification can load XLM-RoBERTa Models with sequence classification/regression head on top e.g. for multi-class document classification tasks."%}
{% include templates/anno_table_entry.md path="./transformers" name="XlmRoBertaForTokenClassification" summary="XlmRoBertaForTokenClassification can load XLM-RoBERTa Models with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for Named-Entity-Recognition (NER) tasks."%}
{% include templates/anno_table_entry.md path="./transformers" name="XlmRoBertaForZeroShotClassification" summary="XlmRoBertaForZeroShotClassification using a `ModelForSequenceClassification` trained on NLI (natural language inference) tasks."%}
{% include templates/anno_table_entry.md path="./transformers" name="XlmRoBertaSentenceEmbeddings" summary="Sentence-level embeddings using XLM-RoBERTa."%}
{% include templates/anno_table_entry.md path="./transformers" name="XlnetEmbeddings" summary="XLNet is a new unsupervised language representation learning method based on a novel generalized permutation language modeling objective."%}
{% include templates/anno_table_entry.md path="./transformers" name="XlnetForTokenClassification" summary="XlnetForTokenClassification can load XLNet Models with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for Named-Entity-Recognition (NER) tasks."%}
{% include templates/anno_table_entry.md path="./transformers" name="XlnetForSequenceClassification" summary="XlnetForSequenceClassification can load XLNet Models with sequence classification/regression head on top e.g. for multi-class document classification tasks."%}
{% include templates/anno_table_entry.md path="./transformers" name="ZeroShotNer" summary="ZeroShotNerModel implements zero shot named entity recognition by utilizing RoBERTa transformer models fine tuned on a question answering task."%}

</div>


<script> {% include scripts/approachModelSwitcher.js %} </script>

{% assign parent_path = "en/annotator_entries" %}

{% for file in site.static_files %}
    {% if file.path contains parent_path %}
        {% assign file_name = file.path | remove:  parent_path | remove:  "/" | prepend: "annotator_entries/" %}
        {% include_relative {{ file_name }} %}
    {% endif %}
{% endfor %}
