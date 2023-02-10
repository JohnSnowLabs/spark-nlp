---
layout: model
title: Classifier for Genders - SBERT
author: John Snow Labs
name: classifierdl_gender_sbert
date: 2020-12-16
task: Text Classification
language: en
edition: Healthcare NLP 2.6.5
spark_version: 2.4
tags: [classifier, en, clinical, licensed]
supported: true
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description

This model classifies the gender of the patient in the clinical document. 

{:.h2_title}
## Predicted Entities

`Female`, `Male`, `Unknown`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/CLINICAL_CLASSIFICATION.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}{:target="_blank"}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/classifierdl_gender_sbert_en_2.6.4_2.4_1608119379496.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/classifierdl_gender_sbert_en_2.6.4_2.4_1608119379496.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

{:.h2_title}
## How to use
To classify your text, you can use this model as part of an nlp pipeline with the following stages: DocumentAssembler, BertSentenceEmbeddings (``sbiobert_base_cased_mli``), ClassifierDLModel.

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}


```python
document_assembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

sbert_embedder = BertSentenceEmbeddings.pretrained("sbiobert_base_cased_mli", "en", "clinical/models")\
    .setInputCols(["document"])\
    .setOutputCol("sentence_embeddings")\
    .setMaxSentenceLength(512)

gender_classifier = ClassifierDLModel.pretrained("classifierdl_gender_sbert", "en", "clinical/models") \
    .setInputCols(["document", "sentence_embeddings"]) \
    .setOutputCol("class")

nlp_pipeline = Pipeline(stages=[document_assembler, sbert_embedder, gender_classifier])

light_pipeline = LightPipeline(nlp_pipeline.fit(spark.createDataFrame([[""]]).toDF("text")))

annotations = light_pipeline.fullAnnotate("""social history: shows that  does not smoke cigarettes or drink alcohol, lives in a nursing home. family history: shows a family history of breast cancer.""")

```
```scala
val document_assembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")

val sentence_embeddings = BertSentenceEmbeddings.pretrained("sbiobert_base_cased_mli", "en", "clinical/models")
    .setInputCols("document")
    .setOutputCol("sentence_embeddings")
    .setMaxSentenceLength(512)

val gender_classifier = ClassifierDLModel.pretrained("classifierdl_gender_sbert", "en", "clinical/models")
    .setInputCols(Array("document", "sentence_embeddings"))
    .setOutputCol("class")

val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_embeddings, gender_classifier))

val data = Seq("""social history: shows that  does not smoke cigarettes or drink alcohol, lives in a nursing home. family history: shows a family history of breast cancer.""").toDS().toDF("text")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.classify.gender.sbert").predict("""social history: shows that  does not smoke cigarettes or drink alcohol, lives in a nursing home. family history: shows a family history of breast cancer.""")
```

</div>

{:.h2_title}
## Results

```bash
Female
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|classifierdl_gender_sbert|
|Type:|ClassifierDLModel|
|Compatibility:|Healthcare NLP 2.6.5 +|
|Edition:|Official|
|License:|Licensed|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[class]|
|Language:|[en]|
|Case sensitive:|True|

{:.h2_title}
## Data Source
This model is trained on more than four thousands clinical documents (radiology reports, pathology reports, clinical visits etc.), annotated internally.

{:.h2_title}
## Benchmarking
```bash
label           precision    recall    f1-score   support
Female           0.9224      0.8954    0.9087       239
Male             0.7895      0.8468    0.8171       124
Unknown          0.8077      0.7778    0.7925        54
accuracy          -           -        0.8657       417
macro-avg        0.8399      0.8400    0.8394       417
weighted-avg     0.8680      0.8657    0.8664       417
```
