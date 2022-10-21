{%- capture title -%}
XlmRoBertaForTokenClassification
{%- endcapture -%}

{%- capture description -%}
XlmRoBertaForTokenClassification can load XLM-RoBERTa Models with a token
classification head on top (a linear layer on top of the hidden-states output)
e.g. for Named-Entity-Recognition (NER) tasks.

Since Spark NLP 3.2.x, this annotator needs to be trained externally using the
transformers library. After the training process is done, the model checkpoint
can be loaded by this annotator. This is done with `loadSavedModel` (for loading
the transformers model) and `load` for the saved Spark NLP model.

For an extended example see the [Spark NLP Workshop](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/transformers/HuggingFace%20in%20Spark%20NLP%20-%20XlmRoBertaForTokenClassification.ipynb).

Example for loading a saved transformers model:
```python
# Installing prerequisites
!pip install -q transformers==4.10.0 tensorflow==2.4.1

# Loading the external transformers model
from transformers import TFXLMRobertaForTokenClassification, XLMRobertaTokenizer

MODEL_NAME = 'wpnbos/xlm-roberta-base-conll2002-dutch'

tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_NAME)
tokenizer.save_pretrained('./{}_tokenizer/'.format(MODEL_NAME))

# just in case if there is no TF/Keras file provided in the model
# we can just use `from_pt` and convert PyTorch to TensorFlow
try:
  print('try downloading TF weights')
  model = TFXLMRobertaForTokenClassification.from_pretrained(MODEL_NAME)
except:
  print('try downloading PyTorch weights')
  model = TFXLMRobertaForTokenClassification.from_pretrained(MODEL_NAME, from_pt=True)

model.save_pretrained("./{}".format(MODEL_NAME), saved_model=True)

# get assets
asset_path = '{}/saved_model/1/assets'.format(MODEL_NAME)

# let's copy sentencepiece.bpe.model file to saved_model/1/assets
!cp {MODEL_NAME}_tokenizer/sentencepiece.bpe.model {asset_path}

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

MODEL_NAME = 'wpnbos/xlm-roberta-base-conll2002-dutch'
tokenClassifier = XlmRoBertaForTokenClassification.loadSavedModel(
    '{}/saved_model/1'.format(MODEL_NAME),
    spark) \
    .setInputCols(["document",'token']) \
    .setOutputCol("ner") \
    .setCaseSensitive(True) \
    .setMaxSentenceLength(128)

# Optionally the classifier can be saved to load it more conveniently into Spark NLP
tokenClassifier.write().overwrite().save("./{}_spark_nlp".format(MODEL_NAME))

tokenClassifier = XlmRoBertaForTokenClassification.load("./{}_spark_nlp".format(MODEL_NAME))\
  .setInputCols(["document",'token'])\
  .setOutputCol("label")

pipeline = Pipeline().setStages([
    documentAssembler,
    tokenizer,
    tokenClassifier
])

data = spark.createDataFrame([["John Lenon is geboren in Londen en heeft in Parijs gewoond. Mijn naam is Sarah en ik woon in Londen"]]).toDF("text")
result = pipeline.fit(data).transform(data)

result.select("label.result").show(truncate=False)
+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|result                                                                                                                                                                                       |
+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|[LABEL_1, LABEL_2, LABEL_0, LABEL_0, LABEL_0, LABEL_5, LABEL_0, LABEL_0, LABEL_0, LABEL_5, LABEL_0, LABEL_0, LABEL_0, LABEL_0, LABEL_0, LABEL_1, LABEL_0, LABEL_0, LABEL_0, LABEL_0, LABEL_5]|
+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
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

val modelName = "wpnbos/xlm-roberta-base-conll2002-dutch"
var tokenClassifier = XlmRoBertaForTokenClassification.loadSavedModel(s"$modelName/saved_model/1", spark)
  .setInputCols("token", "document")
  .setOutputCol("label")
  .setCaseSensitive(true)
  .setMaxSentenceLength(128)

// Optionally the classifier can be saved to load it more conveniently into Spark NLP
tokenClassifier.write.overwrite.save(s"${modelName}_spark_nlp")

tokenClassifier = XlmRoBertaForTokenClassification.load(s"${modelName}_spark_nlp")
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
+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|result                                                                                                                                                                                       |
+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|[LABEL_1, LABEL_2, LABEL_0, LABEL_0, LABEL_0, LABEL_5, LABEL_0, LABEL_0, LABEL_0, LABEL_5, LABEL_0, LABEL_0, LABEL_0, LABEL_0, LABEL_0, LABEL_1, LABEL_0, LABEL_0, LABEL_0, LABEL_0, LABEL_5]|
+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

{%- endcapture -%}

{%- capture api_link -%}
[XlmRoBertaForTokenClassification](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/annotators/classifier/dl/XlmRoBertaForTokenClassification)
{%- endcapture -%}

{%- capture python_api_link -%}
[XlmRoBertaForTokenClassification](/api/python/reference/autosummary/python/sparknlp/annotator/classifier_dl/xlm_roberta_for_token_classification/index.html#sparknlp.annotator.classifier_dl.xlm_roberta_for_token_classification.XlmRoBertaForTokenClassification)
{%- endcapture -%}

{%- capture source_link -%}
[XlmRoBertaForTokenClassification](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/classifier/dl/XlmRoBertaForTokenClassification.scala)
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
