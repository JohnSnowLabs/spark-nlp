---
layout: model
title: Multilabel Text Classification for Mental Disorder
author: OanaStoicaE
name: multiclassifierdl_mental_disorder
date: 2023-10-09
tags: [en, licensed, text_classification, multiclassifier, mental_disorder, depression, anxiety_disorder, bipolar_disorder, schizophrenia, open_source, tensorflow]
task: Text Classification
language: en
edition: Spark NLP 5.1.2
spark_version: 3.0
supported: false
engine: tensorflow
annotator: MultiClassifierDLModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

The PHS-BERT Mental Disorder Classifier Model is a specialized text classification system, engineered to accurately identify and categorize textual mentions of four prominent mental disorders: Schizophrenia, Depression, Bipolar disorder, Anxiety disorder. More detailed information about classes as follows:

`Schizophrenia`: This category is designated for text mentions that correspond to Schizophrenia, a mental disorder characterized by thought disorders, hallucinations, and often involving delusional beliefs and cognitive challenges. Example: "My brother’s hallucinations, a symptom of his schizophrenia, make daily tasks challenging for him."

`Depression`: Textual content which implicates Depression, a pervasive mental health disorder marked by persistent feelings of sadness, loss of interest in activities, and potential physical health alterations, is classified here. Example: "The heavy weight of depression often leaves me struggling to find joy in activities I once loved."

`Bipolar Disorder`: Entries here incorporate text alluding to Bipolar Disorder, a mental health condition distinguished by stark fluctuations in mood, transitioning between manic highs and depressive lows. Example: "Bipolar disorder has my mood swinging from exhilarating happiness to debilitating lows without warning."

`Anxiety Disorder`: Textual mentions signifying Anxiety Disorder, which involves persistent, excessive worry, nervousness, and physiological effects, are classified under this label. Example: "My anxiety disorder frequently sets my heart racing and mind spiralling over mere hypothetical scenarios."

## Predicted Entities

`Depression`, `Anxiety disorder`, `Bipolar disorder`, `Schizophrenia`, `No`, `Other/Unknown`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/community.johnsnowlabs.com/OanaStoicaE/multiclassifierdl_mental_disorder_en_5.1.2_3.0_1696878798061.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://community.johnsnowlabs.com/OanaStoicaE/multiclassifierdl_mental_disorder_en_5.1.2_3.0_1696878798061.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
# Importing required libraries
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.sql.types import StringType
from sparknlp.base import DocumentAssembler
from sparknlp.annotator import BertSentenceEmbeddings
from sparknlp_jsl.annotator import MultiClassifierDLModel

# Initiating a Spark session
spark = SparkSession.builder \
    .appName("Mental Disorder Classifier") \
    .getOrCreate()

# Setting up the pipeline components

# 1. Document assembler
document_assembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

# 2. BERT embeddings
sentence_embeddings = BertSentenceEmbeddings.pretrained("embeddings_clinical", 'en', 'clinical/models') \
    .setInputCols(["document"]) \
    .setOutputCol("sentence_embeddings")

# 3. Mental Disorder MultiClassifier (Using the provided model name)
multilabel_classifier = MultiClassifierDLModel.pretrained("multiclassifierdl_mental_disorder", 'en', 'clinical/models') \
    .setInputCols(["sentence_embeddings"]) \
    .setOutputCol("prediction")

# Creating the pipeline
pipeline = Pipeline(stages=[
    document_assembler,
    sentence_embeddings,
    multilabel_classifier
])

# Sample text list
text_list = [
    "My brother’s hallucinations, a symptom of his schizophrenia, make daily tasks challenging for him.",
    "The heavy weight of depression often leaves me struggling to find joy in activities I once loved.",
    "Bipolar disorder has my mood swinging from exhilarating happiness to debilitating lows without warning.",
    "My anxiety disorder frequently sets my heart racing and mind spiralling over mere hypothetical scenarios."
]

# Creating a DataFrame
df = spark.createDataFrame(text_list, StringType()).toDF("text")

# Applying the pipeline to the data
result = pipeline.fit(df).transform(df)

# Displaying results
result.select("text", "prediction.result").show(truncate=100)

```
```scala
val documentAssembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")

val tokenizer = new Tokenizer()
    .setInputCols(Array("document"))
    .setOutputCol("token")

val wordEmbeddings = WordEmbeddingsModel
    .pretrained("embeddings_clinical", "en", "clinical/models")
    .setInputCols(Array("document", "token"))
    .setOutputCol("word_embeddings")

val sentenceEmbeddings = new SentenceEmbeddings()
    .setInputCols(Array("document", "word_embeddings"))
    .setOutputCol("sentence_embeddings")
    .setPoolingStrategy("AVERAGE")

val multilabelClassifier = MultiClassifierDLModel
    .pretrained("multiclassifierdl_mental_disorder", "en", "clinical/models")
    .setInputCols(Array("sentence_embeddings"))
    .setOutputCol("prediction")

val clfPipeline = new Pipeline()
    .setStages(Array(
        documentAssembler,
        tokenizer,
        wordEmbeddings,
        sentenceEmbeddings,
        multilabelClassifier
    ))

val textList = Seq(
    "My brother’s hallucinations, a symptom of his schizophrenia, make daily tasks challenging for him.",
    "The heavy weight of depression often leaves me struggling to find joy in activities I once loved.",
    "Bipolar disorder has my mood swinging from exhilarating happiness to debilitating lows without warning.",
    "My anxiety disorder frequently sets my heart racing and mind spiralling over mere hypothetical scenarios."
)

val df = spark.createDataFrame(textList).toDF("text")

val result = clfPipeline.fit(df).transform(df)

```
</div>

## Results

```bash
+----------------------------------------------------------------------------------------------------+---------------------------------------------------+\n","|                                                                                                text|                                             result|\n","+----------------------------------------------------------------------------------------------------+---------------------------------------------------+\n","|  My brother’s hallucinations, a symptom of his schizophrenia, make daily tasks challenging for him.|      [Anxiety disorder, Depression, Schizophrenia]|\n","|   The heavy weight of depression often leaves me struggling to find joy in activities I once loved.|                     [Anxiety disorder, Depression]|\n","|Bipolar disorder has my mood swinging from exhilarating happiness to debilitating lows without wa...|               [Anxiety disorder, Bipolar disorder]|\n","|My anxiety disorder frequently sets my heart racing and mind spiralling over mere hypothetical sc...|[Anxiety disorder, Schizophrenia, Bipolar disorder]|\n","+----------------------------------------------------------------------------------------------------+---------------------------------------------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|multiclassifierdl_mental_disorder|
|Compatibility:|Spark NLP 5.1.2+|
|License:|Open Source|
|Edition:|Community|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[class]|
|Language:|en|
|Size:|87.8 MB|
|Dependencies:|`embeddings_clinical`|

## References

Trained with the in-house dataset

## Sample text from the training dataset

Schizophrenia: The patient was diagnosed with schizophrenia at the age of 22 after a series of intense psychotic episodes. The patient often describes hearing voices that others don't hear and harbors a persistent belief in unfounded conspiracies, indicative of schizophrenia. The patient has been prescribed second-generation antipsychotics and will be undergoing cognitive behavioral therapy to help manage the symptoms of schizophrenia.

Depression: Mrs. Thompson first sought treatment for depression during her late 20s, shortly after a series of personal losses. She often expresses feelings of hopelessness, has a persistent low mood, and lacks interest in activities she once enjoyed, classic signs of depression. Mrs. Thompson will continue her selective serotonin reuptake inhibitor (SSRI) medication and engage in weekly psychotherapy sessions to address her depressive symptoms.

Bipolar Disorder: Mr. Gray was diagnosed with bipolar disorder at the age of 27 following an episode of mania that was preceded by a deep depressive phase. He cycles between periods of elevated mood, increased energy, and impulsiveness, and phases of intense sadness and lethargy, consistent with bipolar disorder. The management approach for Mr. Gray includes mood stabilizers like lithium, coupled with cognitive-behavioral therapy to monitor and address the mood swings of bipolar disorder.

Anxiety Disorder: The patient has been experiencing generalized anxiety disorder symptoms for the past six years, often linked to work-related stress. The patient describes persistent feelings of restlessness, muscle tension, and overwhelming worry about everyday events and activities, common to those with an anxiety disorder. A combination of cognitive behavioral therapy and an anxiolytic medication has been advised to alleviate the distressing symptoms of the anxiety disorder.

## Benchmarking

```bash
label          	        tp	       fp	     fn	       prec	      rec	                    f1
Other/Unknown       39	      10	    22	   0.79591835	   0.6393443	  0.7090909
Depression                120	      18	    19	   0.8695652	   0.8633093	  0.866426
No                 	        28        21	    27	   0.93203884	   0.9142857	  0.9230769
Anxiety disorder       136	      15	    18	   0.90066224	   0.8831169	  0.8918033
Bipolar disorder        101       5	    10	   0.9528302	   0.9099099	  0.9308756
Schizophrenia             66	       4	    23	   0.94285715	   0.74157304	  0.8301887
Macro-average          750      73       119        0.89897865        0.8252565        0.8605415
Micro-average           750      73       119        0.9113001          0.863061           0.88652486

```