{%- capture title -%}
DistilBertForSequenceClassification
{%- endcapture -%}

{%- capture description -%}
DistilBertForSequenceClassification can load Bert Models with sequence
classification/regression head on top (a linear layer on top of the pooled
output) e.g. for multi-class document classification tasks.

Since Spark NLP 3.2.x, this annotator needs to be trained externally using the
transformers library. After the training process is done, the model checkpoint
can be loaded by this annotator. This is done with `loadSavedModel` (for loading
the transformers model) and `load` for the saved Spark NLP model.

For an extended example see the [Spark NLP Workshop](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/transformers/HuggingFace%20in%20Spark%20NLP%20-%20DistilBertForSequenceClassification.ipynb).

Example for loading a saved transformers model:
```python
# Installing prerequisites
!pip install -q transformers==4.8.1 tensorflow==2.4.1

# Loading the external transformers model
from transformers import TFDistilBertForSequenceClassification, DistilBertTokenizer

MODEL_NAME = 'distilbert-base-uncased-finetuned-sst-2-english'

tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
tokenizer.save_pretrained('./{}_tokenizer/'.format(MODEL_NAME))

try:
    model = TFDistilBertForSequenceClassification.from_pretrained(MODEL_NAME)
except:
    model = TFDistilBertForSequenceClassification.from_pretrained(MODEL_NAME, from_pt=True)

model.save_pretrained("./{}".format(MODEL_NAME), saved_model=True)

# Extracting the tokenizer resources
asset_path = '{}/saved_model/1/assets'.format(MODEL_NAME)

!cp {MODEL_NAME}_tokenizer/vocab.txt {asset_path}

# Get label2id dictionary
labels = model.config.label2id
# Sort the dictionary based on the id
labels = sorted(labels, key=labels.get)

with open(asset_path+'/labels.txt', 'w') as f:
    f.write('\n'.join(labels))
```

Then the model can be loaded and used into Spark NLP in the following examples:
{%- endcapture -%}

{%- capture input_anno -%}
DOCUMENT, TOKEN
{%- endcapture -%}

{%- capture output_anno -%}
NAMED_ENTITY
{%- endcapture -%}

{%- capture python_example -%}
import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline

documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

tokenizer = Tokenizer() \
    .setInputCols(["document"]) \
    .setOutputCol("token")

MODEL_NAME = 'distilbert-base-uncased-finetuned-sst-2-english'
tokenClassifier = DistilBertForSequenceClassification.loadSavedModel(
    '{}/saved_model/1'.format(MODEL_NAME),
    spark) \
    .setInputCols(["document",'token']) \
    .setOutputCol("ner") \
    .setCaseSensitive(True) \
    .setMaxSentenceLength(128)

# Optionally the classifier can be saved to load it more conveniently into Spark NLP
tokenClassifier.write().overwrite().save("./{}_spark_nlp".format(MODEL_NAME))

tokenClassifier = DistilBertForSequenceClassification.load("./{}_spark_nlp".format(MODEL_NAME))\
  .setInputCols(["document",'token'])\
  .setOutputCol("label")

pipeline = Pipeline().setStages([
    documentAssembler,
    tokenizer,
    tokenClassifier
])

data = spark.createDataFrame([["The movie was great!"]]).toDF("text")
result = pipeline.fit(data).transform(data)

result.select("label.result").show(truncate=False)
+----------+
|result    |
+----------+
|[POSITIVE]|
+----------+
{%- endcapture -%}

{%- capture scala_example -%}
// The model needs to be trained with the transformers library.
// Afterwards it can be loaded into the scala version of Spark NLP.

import spark.implicits._
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.annotator._
import org.apache.spark.ml.Pipeline

val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val tokenizer = new Tokenizer()
  .setInputCols("document")
  .setOutputCol("token")

val ModelName = "distilbert-base-uncased-finetuned-sst-2-english"
val tokenClassifier = DistilBertForSequenceClassification
  .loadSavedModel(s"$ModelName/saved_model/1", spark)
  .setInputCols("document", "token")
  .setOutputCol("ner")
  .setCaseSensitive(false)
  .setMaxSentenceLength(128)

// Optionally the classifier can be saved to load it more conveniently into Spark NLP
tokenClassifier.write.overwrite().save(s"${ModelName}_spark_nlp")

val tokenClassifierLoaded = DistilBertForSequenceClassification.load(s"${ModelName}_spark_nlp")
  .setInputCols("document", "token")
  .setOutputCol("ner")

val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
  tokenizer,
  tokenClassifierLoaded
))

val data = Seq("The movie was great!").toDF("text")
val result = pipeline.fit(data).transform(data)

result.select("ner.result").show(truncate = false)
+----------+
|result    |
+----------+
|[POSITIVE]|
+----------+
{%- endcapture -%}

{%- capture api_link -%}
[DistilBertForSequenceClassification](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/annotators/classifier/dl/DistilBertForSequenceClassification)
{%- endcapture -%}

{%- capture python_api_link -%}
[DistilBertForSequenceClassification](/api/python/reference/autosummary/python/sparknlp/annotator/classifier_dl/distil_bert_for_sequence_classification/index.html#sparknlp.annotator.classifier_dl.distil_bert_for_sequence_classification.DistilBertForSequenceClassification)
{%- endcapture -%}

{%- capture source_link -%}
[DistilBertForSequenceClassification](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/classifier/dl/DistilBertForSequenceClassification.scala)
{%- endcapture -%}

{% include templates/training_anno_template.md
title=title
description=description
input_anno=input_anno
output_anno=output_anno
python_example=python_example
scala_example=scala_example
python_api_link=python_api_link
api_link=api_link
source_link=source_link
%}
