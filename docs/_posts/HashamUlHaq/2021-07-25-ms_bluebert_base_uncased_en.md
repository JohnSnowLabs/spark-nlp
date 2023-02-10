---
layout: model
title: MS-BERT base model (uncased)
author: John Snow Labs
name: ms_bluebert_base_uncased
date: 2021-07-25
tags: [embeddings, bert, open_source, en, clinical]
task: Embeddings
language: en
edition: Spark NLP 3.1.3
spark_version: 3.0
supported: true
annotator: BertEmbeddings
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model is trained by taking BlueBert as the base model, and training on dataset contained approximately 75,000 clinical notes, for about 5000 patients, totaling to over 35.7 million words. These notes were collected from patients who visited St. Michael's Hospital MS Clinic between 2015 to 2019. The notes contained a variety of information pertaining to a neurological exam. For example, a note can contain information on the patient's condition, their progress over time and diagnosis.

BERT is a transformers model pretrained on a large corpus of English data in a self-supervised fashion. This means it was pretrained on the raw texts only, with no humans labeling them in any way (which is why it can use lots of publicly available data) with an automatic process to generate inputs and labels from those texts. More precisely, it was pretrained with two objectives:

Masked language modeling (MLM): taking a sentence, the model randomly masks 15% of the words in the input then runs the entire masked sentence through the model and has to predict the masked words. This is different from traditional recurrent neural networks (RNNs) that usually see the words one after the other, or from autoregressive models like GPT which internally mask the future tokens. It allows the model to learn a bidirectional representation of the sentence.
Next sentence prediction (NSP): the models concatenate two masked sentences as inputs during pretraining. Sometimes they correspond to sentences that were next to each other in the original text, sometimes not. The model then has to predict if the two sentences were following each other or not. This way, the model learns an inner representation of the English language that can then be used to extract features useful for downstream tasks: if you have a dataset of labeled sentences, for instance, you can train a standard classifier using the features produced by the BERT model as inputs.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/ms_bluebert_base_uncased_en_3.1.3_3.0_1627225948184.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/ms_bluebert_base_uncased_en_3.1.3_3.0_1627225948184.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
embeddings = BertEmbeddings.pretrained("ms_bluebert_base_uncased", "en") \
      .setInputCols(["sentence", "token"]) \
      .setOutputCol("embeddings")

nlp_pipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, embeddings])
```
```scala
val embeddings = BertEmbeddings.pretrained("ms_bluebert_base_uncased", "en")
      .setInputCols("sentence", "token")
      .setOutputCol("embeddings")

val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, embeddings))
```
</div>

## Results

```bash
Generates 768 dimensional embeddings per token
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ms_bluebert_base_uncased|
|Compatibility:|Spark NLP 3.1.3+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[bert]|
|Language:|en|
|Case sensitive:|false|

## Data Source

https://huggingface.co/NLP4H/ms_bert
