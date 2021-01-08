---
layout: model
title: PICO Classifier
author: John Snow Labs
name: classifierdl_pico_biobert
date: 2020-11-12
tags: [classifier, en, licensed, clinical]
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description
Classify medical text according to PICO framework.

## Predicted Classes 
CONCLUSIONS, DESIGN_SETTING, INTERVENTION, PARTICIPANTS, FINDINGS, MEASUREMENTS, AIMS

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/CLASSIFICATION_PICO/){:.button.button-orange}{:target="_blank"}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/CLINICAL_CLASSIFICATION.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}{:target="_blank"}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/classifierdl_pico_biobert_en_2.6.2_2.4_1601901791781.zip){:.button.button-orange.button-orange-trans.arr.button-icon}


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

classifier = ClassifierDLModel.pretrained('classifierdl_pico_biobert', 'en', 'clinical/models')\
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
|Model Name:|classifierdl_pico_biobert|
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
Trained on a custom dataset derived from PICO classification dataset, using 'biobert_pubmed_base_cased' embeddings.

{:.h2_title}
## Benchmarking
```bash
|    |                | precision |   recall | f1-score | support |
|---:|:---------------|----------:|---------:|---------:|--------:|
|  0 | AIMS           |    0.9197 |   0.9121 |   0.9159 |    3845 |
|  1 | CONCLUSIONS    |    0.8426 |   0.8571 |   0.8498 |    4241 |
|  2 | DESIGN_SETTING |    0.7703 |   0.8351 |   0.8014 |    5191 |
|  3 | FINDINGS       |    0.9214 |   0.8964 |   0.9088 |    9500 |
|  4 | INTERVENTION   |    0.7529 |   0.6758 |   0.7123 |    2597 |
|  5 | MEASUREMENTS   |    0.8409 |   0.7734 |   0.8058 |    3500 |
|  6 | PARTICIPANTS   |    0.7521 |   0.8548 |   0.8002 |    2396 |
|  7 | accuracy       |                      |   0.8476 |   31270 |
|  8 | macro avg      |    0.8286 |   0.8292 |   0.8277 |   31270 |
|  9 | weighted avg   |    0.8495 |   0.8476 |   0.8476 |   31270 |

```