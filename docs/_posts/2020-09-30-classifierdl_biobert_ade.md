---
layout: model
title: Classifier for adverse drug reactions
author: John Snow Labs
name: classifierdl_biobert_ade
date: 2020-09-30
tags: [classification, en, licensed]
article_header:
type: cover
use_language_switcher: "Python"
---

## Description
This model can be used to detect clinical events in medical text.

## Predicted Entities
Negative, Neutral

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/classifierdl_biobert_ade_en_2.6.0_2.4_1600201949450.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use
Use as part of an nlp pipeline with the following stages: DocumentAssembler, SentenceDetector, Tokenizer, WordEmbeddingsModel, NerDLModel. Add the NerConverter to the end of the pipeline to convert entity tokens into full entity chunks.

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPython.html %}


```python

clinical_ner = ClassifierDLModel.pretrained("classifierdl_biobert_ade", "en", "clinical/models") \
  .setInputCols(["sentence_embeddings"]) \
  .setOutputCol("class")

nlp_pipeline = Pipeline(stages=[document_assembler, tokenizer, word_embeddings, sentence_embeddings])

light_pipeline = LightPipeline(nlp_pipeline.fit(spark.createDataFrame([['']]).toDF("text")))

annotations = light_pipeline.fullAnnotate(text)

```

</div>

{:.h2_title}
## Results
A dictionary containing class labels for each sentence.


{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|classifierdl_biobert_ade|
|Type:|ClassifierDLModel|
|Compatibility:|Spark NLP for Healthcare 2.6.2 +|
|Edition:|Official|
|License:|Licensed|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[class]|
|Language:|[en]|
|Case sensitive:|false|

