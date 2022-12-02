{%- capture title -%}
RoBertaForTokenClassification
{%- endcapture -%}

{%- capture description -%}
RoBertaForTokenClassification can load RoBERTa Models with a token
classification head on top (a linear layer on top of the hidden-states output)
e.g. for Named-Entity-Recognition (NER) tasks.

Since Spark NLP 3.2.x, this annotator needs to be trained externally using the
transformers library. After the training process is done, the model checkpoint
can be loaded by this annotator. This is done with `loadSavedModel` (for loading
the transformers model) and `load` for the saved Spark NLP model.

For an extended example see the [Spark NLP Workshop](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/transformers/HuggingFace%20in%20Spark%20NLP%20-%20RoBertaForTokenClassification.ipynb).

Example for loading a saved transformers model:
```python
# Installing prerequisites
!pip install -q transformers==4.10.0 tensorflow==2.4.1

# Loading the external transformers model
from transformers import TFRobertaForTokenClassification, RobertaTokenizer

MODEL_NAME = 'philschmid/distilroberta-base-ner-wikiann-conll2003-3-class'

tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
tokenizer.save_pretrained('./{}_tokenizer/'.format(MODEL_NAME))

# just in case if there is no TF/Keras file provided in the model
# we can just use `from_pt` and convert PyTorch to TensorFlow
try:
  print('try downloading TF weights')
  model = TFRobertaForTokenClassification.from_pretrained(MODEL_NAME)
except:
  print('try downloading PyTorch weights')
  model = TFRobertaForTokenClassification.from_pretrained(MODEL_NAME, from_pt=True)

model.save_pretrained("./{}".format(MODEL_NAME), saved_model=True)

# get assets
asset_path = '{}/saved_model/1/assets'.format(MODEL_NAME)

# let's save the vocab as txt file
with open('{}_tokenizer/vocab.txt'.format(MODEL_NAME), 'w') as f:
    for item in tokenizer.get_vocab().keys():
        f.write("%s\n" % item)

# let's copy both vocab.txt and merges.txt files to saved_model/1/assets
!cp {MODEL_NAME}_tokenizer/vocab.txt {asset_path}
!cp {MODEL_NAME}_tokenizer/merges.txt {asset_path}

# get label2id dictionary
labels = model.config.label2id
# sort the dictionary based on the id
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

MODEL_NAME = 'philschmid/distilroberta-base-ner-wikiann-conll2003-3-class'
tokenClassifier = RoBertaForTokenClassification.loadSavedModel(
    '{}/saved_model/1'.format(MODEL_NAME),
    spark) \
    .setInputCols(["document",'token']) \
    .setOutputCol("ner") \
    .setCaseSensitive(True) \
    .setMaxSentenceLength(128)

# Optionally the classifier can be saved to load it more conveniently into Spark NLP
tokenClassifier.write().overwrite().save("./{}_spark_nlp".format(MODEL_NAME))

tokenClassifier = RoBertaForTokenClassification.load("./{}_spark_nlp".format(MODEL_NAME))\
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

val modelName = "philschmid/distilroberta-base-ner-wikiann-conll2003-3-class"
var tokenClassifier = RoBertaForTokenClassification.loadSavedModel(s"$modelName/saved_model/1", spark)
  .setInputCols("token", "document")
  .setOutputCol("label")
  .setCaseSensitive(true)
  .setMaxSentenceLength(128)

// Optionally the classifier can be saved to load it more conveniently into Spark NLP
tokenClassifier.write.overwrite.save(s"${modelName}_spark_nlp")

tokenClassifier = RoBertaForTokenClassification.load(s"${modelName}_spark_nlp")
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
[RoBertaForTokenClassification](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/annotators/classifier/dl/RoBertaForTokenClassification)
{%- endcapture -%}

{%- capture python_api_link -%}
[RoBertaForTokenClassification](/api/python/reference/autosummary/python/sparknlp/annotator/classifier_dl/roberta_for_token_classification/index.html#sparknlp.annotator.classifier_dl.roberta_for_token_classification.RoBertaForTokenClassification)
{%- endcapture -%}

{%- capture source_link -%}
[RoBertaForTokenClassification](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/classifier/dl/RoBertaForTokenClassification.scala)
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
