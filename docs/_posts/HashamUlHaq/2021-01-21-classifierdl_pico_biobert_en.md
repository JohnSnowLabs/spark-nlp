---
layout: model
title: PICO Classifier
author: John Snow Labs
name: classifierdl_pico_biobert
date: 2021-01-21
task: Text Classification
language: en
edition: Healthcare NLP 2.7.1
spark_version: 2.4
tags: [en, licensed, clinical, classifier]
supported: true
annotator: ClassifierDLModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Classify medical text according to PICO framework.

## Predicted Entities

`CONCLUSIONS`, `DESIGN_SETTING`, `INTERVENTION`, `PARTICIPANTS`, `FINDINGS`, `MEASUREMENTS`, `AIMS`.

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/CLASSIFICATION_PICO/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/CLINICAL_CLASSIFICATION.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/classifierdl_pico_biobert_en_2.7.1_2.4_1611248887230.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/classifierdl_pico_biobert_en_2.7.1_2.4_1611248887230.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
document_assembler = DocumentAssembler().setInputCol('text').setOutputCol('document')

tokenizer = Tokenizer().setInputCols('document').setOutputCol('token')

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

annotations = light_pipeline.fullAnnotate(["""A total of 10 adult daily smokers who reported at least one stressful event and coping episode and provided post-quit data.""", """When carbamazepine is withdrawn from the combination therapy, aripiprazole dose should then be reduced."""])
```



{:.nlu-block}
```python
import nlu
nlu.load("en.classify.pico").predict("""A total of 10 adult daily smokers who reported at least one stressful event and coping episode and provided post-quit data.""")
```

</div>

## Results

```bash
|                                            sentences | class        |
|------------------------------------------------------+--------------+
| A total of 10 adult daily smokers who reported at... | PARTICIPANTS |
| When carbamazepine is withdrawn from the combinat... | CONCLUSIONS  |

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|classifierdl_pico_biobert|
|Compatibility:|Spark NLP 2.7.1+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[class]|
|Language:|en|
|Dependencies:|biobert_pubmed_base_cased|

## Data Source

Trained on a custom dataset derived from PICO classification dataset.

## Benchmarking

```bash
precision    recall  f1-score   support

AIMS     0.9229    0.9186    0.9207      7815
CONCLUSIONS     0.8556    0.8401    0.8478      8837
DESIGN_SETTING     0.8556    0.7494    0.7990     11551
FINDINGS     0.8949    0.9342    0.9142     18827
INTERVENTION     0.6866    0.7508    0.7173      4920
MEASUREMENTS     0.7564    0.8664    0.8077      6505
PARTICIPANTS     0.8483    0.7559    0.7994      5539

accuracy                         0.8495     63994
macro avg     0.8315    0.8308    0.8294     63994
weighted avg     0.8517    0.8495    0.8491     63994

```