{%- capture title -%}
BertForTokenClassification
{%- endcapture -%}

{%- capture description -%}
BertForTokenClassification can load Bert Models with a token classification head
on top (a linear layer on top of the hidden-states output) e.g. for
Named-Entity-Recognition (NER) tasks.

Since Spark NLP 3.2.x, this annotator needs to be trained externally using the
transformers library. After the training process is done, the model checkpoint
can be loaded by this annotator. This is done with `loadSavedModel` (for loading
the transformers model) and `load` for the saved Spark NLP model.

For an extended example see the [Spark NLP Workshop](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/transformers/HuggingFace%20in%20Spark%20NLP%20-%20BertForTokenClassification.ipynb).

Example for loading a saved transformers model:
```python
# Installing prerequisites
!pip install -q transformers==4.8.1 tensorflow==2.4.1 spark-nlp>=3.2.0

# Loading the external transformers model
from transformers import TFBertForTokenClassification, BertTokenizer

MODEL_NAME = 'dslim/bert-base-NER'

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
tokenizer.save_pretrained('./{}_tokenizer/'.format(MODEL_NAME))

model = TFBertForTokenClassification.from_pretrained(MODEL_NAME)
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

MODEL_NAME = 'dslim/bert-base-NER'
tokenClassifier = BertForTokenClassification.loadSavedModel(
    '{}/saved_model/1'.format(MODEL_NAME),
    spark) \
    .setInputCols(["document",'token']) \
    .setOutputCol("ner") \
    .setCaseSensitive(True) \
    .setMaxSentenceLength(128)

# Optionally the classifier can be saved to load it more conveniently into Spark NLP
tokenClassifier.write().overwrite().save("./{}_spark_nlp".format(MODEL_NAME))

tokenClassifier = BertForTokenClassification.load("./{}_spark_nlp".format(MODEL_NAME))\
  .setInputCols(["document",'token'])\
  .setOutputCol("label")

pipeline = Pipeline().setStages([
    documentAssembler,
    tokenizer,
    tokenClassifier
])

data = spark.createDataFrame([["John Lenon was born in London and lived in Paris. My name is Sarah and I live in London"]]).toDF("text")
result = pipeline.fit(data).transform(data)

result.select("label.result").show(truncate=False)
+------------------------------------------------------------------------------------+
|result                                                                              |
+------------------------------------------------------------------------------------+
|[B-PER, I-PER, O, O, O, B-LOC, O, O, O, B-LOC, O, O, O, O, B-PER, O, O, O, O, B-LOC]|
+------------------------------------------------------------------------------------+
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

val modelName = "dslim/bert-base-NER"
var tokenClassifier = BertForTokenClassification.loadSavedModel(s"$modelName/saved_model/1", spark)
  .setInputCols("token", "document")
  .setOutputCol("label")
  .setCaseSensitive(true)
  .setMaxSentenceLength(128)

// Optionally the classifier can be saved to load it more conveniently into Spark NLP
tokenClassifier.write.overwrite.save(s"${modelName}_spark_nlp")

tokenClassifier = BertForTokenClassification.load(s"${modelName}_spark_nlp")
  .setInputCols("token", "document")
  .setOutputCol("label")
  .setCaseSensitive(true)

val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
  tokenizer,
  tokenClassifier
))

val data = Seq("John Lenon was born in London and lived in Paris. My name is Sarah and I live in London").toDF("text")
val result = pipeline.fit(data).transform(data)

result.select("label.result").show(false)
+------------------------------------------------------------------------------------+
|result                                                                              |
+------------------------------------------------------------------------------------+
|[B-PER, I-PER, O, O, O, B-LOC, O, O, O, B-LOC, O, O, O, O, B-PER, O, O, O, O, B-LOC]|
+------------------------------------------------------------------------------------+

{%- endcapture -%}

{%- capture api_link -%}
[BertForTokenClassification](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/annotators/classifier/dl/BertForTokenClassification)
{%- endcapture -%}

{%- capture python_api_link -%}
[BertForTokenClassification](/api/python/reference/autosummary/python/sparknlp/annotator/classifier_dl/bert_for_token_classification/index.html#sparknlp.annotator.classifier_dl.bert_for_token_classification.BertForTokenClassification)
{%- endcapture -%}

{%- capture source_link -%}
[BertForTokenClassification](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/classifier/dl/BertForTokenClassification.scala)
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
