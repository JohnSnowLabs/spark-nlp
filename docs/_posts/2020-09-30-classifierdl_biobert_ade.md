---
layout: model
title: Classifier for Adverse Drug Events
author: John Snow Labs
name: classifierdl_biobert_ade
date: 2020-09-30
tags: [classifier, en, clinical, licensed]
article_header:
type: cover
use_language_switcher: "Python"
---

## Description
This model can be used to detect clinical events in medical text.

## Predicted Entities
Negative, Neutral

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/PP_ADE/){:.button.button-orange}{:target="_blank"}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/16.Adverse_Drug_Event_ADE_NER_and_Classifier.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}{:target="_blank"}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/classifierdl_biobert_ade_en_2.6.0_2.4_1600201949450.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use
Use as part of an nlp pipeline with the following stages: DocumentAssembler, SentenceDetector, Tokenizer, BertEmbeddings (biobert_pubmed_base_cased), SentenceEmbeddings, ClassifierDLModel.

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPython.html %}


```python

embeddings = BertEmbeddings.pretrained('biobert_pubmed_base_cased')\
    .setInputCols(["document", 'token'])\
    .setOutputCol("word_embeddings")

sentence_embeddings = SentenceEmbeddings() \
      .setInputCols(["document", "word_embeddings"]) \
      .setOutputCol("sentence_embeddings") \
      .setPoolingStrategy("AVERAGE")

classifier = ClassifierDLModel.pretrained('classifierdl_biobert_ade', 'en', 'clinical/models')\
    .setInputCols(['document', 'token', 'sentence_embeddings']).setOutputCol('class')

nlp_pipeline = Pipeline(stages=[document_assembler, tokenizer, embeddings, sentence_embeddings, classifier])

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
|Case sensitive:|True|

{:.h2_title}
## Data Source
Trained on a custom dataset comprising of CADEC, DRUG-AE, Twimed using 'biobert_pubmed_base_cased' embeddings.

