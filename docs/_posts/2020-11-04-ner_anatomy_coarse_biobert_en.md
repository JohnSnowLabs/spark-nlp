---
layout: model
title: Detect Anatomical Structures (Single Entity - biobert)
author: John Snow Labs
name: ner_anatomy_coarse_biobert_en
date: 2020-11-04
tags: [ner, en, licensed, clinical]
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

An NER model to extract all types of anatomical references in text using "biobert_pubmed_base_cased" embeddings. It is a single entity model and generalizes all anatomical references to a single entity.

## Predicted Entities 
Anatomy

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/NER_ANATOMY/){:.button.button-orange}{:target="_blank"}
[Open in Colab](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/1.Clinical_Named_Entity_Recognition_Model.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}{:target="_blank"}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_anatomy_coarse_biobert_en_2.6.1_2.4_1604435983087.zip){:.button.button-orange.button-orange-trans.arr.button-icon}


## How to use

Use as part of an nlp pipeline with the following stages: DocumentAssembler, SentenceDetector, Tokenizer, WordEmbeddingsModel, NerDLModel. Add the NerConverter to the end of the pipeline to convert entity tokens into full entity chunks.

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python

clinical_ner = NerDLModel.pretrained("ner_anatomy_coarse_biobert", "en", "clinical/models") \
  .setInputCols(["sentence", "token", "embeddings"]) \
  .setOutputCol("ner")

nlpPipeline = Pipeline(stages=[clinical_ner])

empty_data = spark.createDataFrame([["content in the lung tissue"]]).toDF("text")

model = nlpPipeline.fit(empty_data)

results = model.transform(data)

```

```scala

val ner = NerDLModel.pretrained("ner_anatomy_coarse_biobert", "en", "clinical/models") \
  .setInputCols(["sentence", "token", "embeddings"]) \
  .setOutputCol("ner")

val pipeline = new Pipeline().setStages(Array(ner))

val result = pipeline.fit(Seq.empty["content in the lung tissue"].toDS.toDF("text")).transform(data)


```

</div>

{:.h2_title}
## Results
The output is a dataframe with a sentence per row and a "ner" column containing all of the entity labels in the sentence, entity character indices, and other metadata. To get only the tokens and entity labels, without the metadata, select "token.result" and "ner.result" from your output dataframe or add the "Finisher" to the end of your pipeline.me:
```bash
|    | ner_chunk         | entity    |
|---:|:------------------|:----------|
|  0 | lung tissue       | Anatomy   |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_anatomy_coarse_biobert|
|Type:|NerDLModel|
|Compatibility:|Spark NLP 2.6.1 +|
|Edition:|Official|
|License:|Licensed|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|[en]|
|Case sensitive:|True|

{:.h2_title}
## Data Source
Trained on a custom dataset using 'biobert_pubmed_base_cased'.


{:.h2_title}
## Benchmarking
```bash
|    | label         |    tp |    fp |    fn |     prec |      rec |       f1 |
|---:|:--------------|------:|------:|------:|---------:|---------:|---------:|
|  0 | B-Anatomy     |  2499 |   155 |   162 | 0.941598 | 0.939121 | 0.940357 |
|  1 | I-Anatomy     |  1695 |   116 |   158 | 0.935947 | 0.914733 | 0.925218 |
|  2 | Macro-average | 4194  |  271  |   320 | 0.938772 | 0.926927 | 0.932812 |
|  3 | Micro-average | 4194  |  271  |   320 | 0.939306 | 0.929109 | 0.93418  |

```