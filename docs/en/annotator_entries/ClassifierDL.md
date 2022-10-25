{%- capture title -%}
ClassifierDL
{%- endcapture -%}

{%- capture model_description -%}
ClassifierDL for generic Multi-class Text Classification.

ClassifierDL uses the state-of-the-art Universal Sentence Encoder as an input for text classifications.
The ClassifierDL annotator uses a deep learning model (DNNs) we have built inside TensorFlow and supports up to
100 classes.

This is the instantiated model of the ClassifierDLApproach.
For training your own model, please see the documentation of that class.

Pretrained models can be loaded with `pretrained` of the companion object:
```
val classifierDL = ClassifierDLModel.pretrained()
  .setInputCols("sentence_embeddings")
  .setOutputCol("classification")
```
The default model is `"classifierdl_use_trec6"`, if no name is provided. It uses embeddings from the
[UniversalSentenceEncoder](/docs/en/transformers#universalsentenceencoder) and is trained on the
[TREC-6](https://deepai.org/dataset/trec-6#:~:text=The%20TREC%20dataset%20is%20dataset,50%20has%20finer%2Dgrained%20labels) dataset.
For available pretrained models please see the [Models Hub](https://nlp.johnsnowlabs.com/models?task=Text+Classification).

For extended examples of usage, see the
[Spark NLP Workshop](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/5.Text_Classification_with_ClassifierDL.ipynb)
and the [ClassifierDLTestSpec](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/classifier/dl/ClassifierDLTestSpec.scala).
{%- endcapture -%}

{%- capture model_input_anno -%}
SENTENCE_EMBEDDINGS
{%- endcapture -%}

{%- capture model_output_anno -%}
CATEGORY
{%- endcapture -%}

{%- capture model_python_example -%}
import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline

documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

sentence = SentenceDetector() \
    .setInputCols("document") \
    .setOutputCol("sentence")

useEmbeddings = UniversalSentenceEncoder.pretrained() \
    .setInputCols("document") \
    .setOutputCol("sentence_embeddings")

sarcasmDL = ClassifierDLModel.pretrained("classifierdl_use_sarcasm") \
    .setInputCols("sentence_embeddings") \
    .setOutputCol("sarcasm")

pipeline = Pipeline() \
    .setStages([
      documentAssembler,
      sentence,
      useEmbeddings,
      sarcasmDL
    ])

data = spark.createDataFrame([
    ["I'm ready!"],
    ["If I could put into words how much I love waking up at 6 am on Mondays I would."]
]).toDF("text")
result = pipeline.fit(data).transform(data)

result.selectExpr("explode(arrays_zip(sentence, sarcasm)) as out") \
    .selectExpr("out.sentence.result as sentence", "out.sarcasm.result as sarcasm") \
    .show(truncate=False)
+-------------------------------------------------------------------------------+-------+
|sentence                                                                       |sarcasm|
+-------------------------------------------------------------------------------+-------+
|I'm ready!                                                                     |normal |
|If I could put into words how much I love waking up at 6 am on Mondays I would.|sarcasm|
+-------------------------------------------------------------------------------+-------+

{%- endcapture -%}

{%- capture model_scala_example -%}
import spark.implicits._
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.annotator.SentenceDetector
import com.johnsnowlabs.nlp.annotators.classifier.dl.ClassifierDLModel
import com.johnsnowlabs.nlp.embeddings.UniversalSentenceEncoder
import org.apache.spark.ml.Pipeline

val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val sentence = new SentenceDetector()
  .setInputCols("document")
  .setOutputCol("sentence")

val useEmbeddings = UniversalSentenceEncoder.pretrained()
  .setInputCols("document")
  .setOutputCol("sentence_embeddings")

val sarcasmDL = ClassifierDLModel.pretrained("classifierdl_use_sarcasm")
  .setInputCols("sentence_embeddings")
  .setOutputCol("sarcasm")

val pipeline = new Pipeline()
  .setStages(Array(
    documentAssembler,
    sentence,
    useEmbeddings,
    sarcasmDL
  ))

val data = Seq(
  "I'm ready!",
  "If I could put into words how much I love waking up at 6 am on Mondays I would."
).toDF("text")
val result = pipeline.fit(data).transform(data)

result.selectExpr("explode(arrays_zip(sentence, sarcasm)) as out")
  .selectExpr("out.sentence.result as sentence", "out.sarcasm.result as sarcasm")
  .show(false)
+-------------------------------------------------------------------------------+-------+
|sentence                                                                       |sarcasm|
+-------------------------------------------------------------------------------+-------+
|I'm ready!                                                                     |normal |
|If I could put into words how much I love waking up at 6 am on Mondays I would.|sarcasm|
+-------------------------------------------------------------------------------+-------+

{%- endcapture -%}

{%- capture model_api_link -%}
[ClassifierDLModel](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/annotators/classifier/dl/ClassifierDLModel)
{%- endcapture -%}

{%- capture model_python_api_link -%}
[ClassifierDLModel](/api/python/reference/autosummary/python/sparknlp/annotator/classifier_dl/classifier_dl/index.html#sparknlp.annotator.classifier_dl.classifier_dl.ClassifierDLModel)
{%- endcapture -%}

{%- capture model_source_link -%}
[ClassifierDLModel](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/classifier/dl/ClassifierDLModel.scala)
{%- endcapture -%}

{%- capture approach_description -%}
Trains a ClassifierDL for generic Multi-class Text Classification.

ClassifierDL uses the state-of-the-art Universal Sentence Encoder as an input for text classifications.
The ClassifierDL annotator uses a deep learning model (DNNs) we have built inside TensorFlow and supports up to
100 classes.

For instantiated/pretrained models, see ClassifierDLModel.

For extended examples of usage, see the Spark NLP Workshop
[[1] ](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/scala/training/Train%20Multi-Class%20Text%20Classification%20on%20News%20Articles.scala)
[[2] ](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/5.Text_Classification_with_ClassifierDL.ipynb)
and the [ClassifierDLTestSpec](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/classifier/dl/ClassifierDLTestSpec.scala).
{%- endcapture -%}

{%- capture approach_input_anno -%}
SENTENCE_EMBEDDINGS
{%- endcapture -%}

{%- capture approach_output_anno -%}
CATEGORY
{%- endcapture -%}

{%- capture approach_python_example -%}
# In this example, the training data `"sentiment.csv"` has the form of
#
# text,label
# This movie is the best movie I have wached ever! In my opinion this movie can win an award.,0
# This was a terrible movie! The acting was bad really bad!,1
# ...
#
# Then traning can be done like so:

import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline

smallCorpus = spark.read.option("header","True").csv("src/test/resources/classifier/sentiment.csv")

documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

useEmbeddings = UniversalSentenceEncoder.pretrained() \
    .setInputCols("document") \
    .setOutputCol("sentence_embeddings")

docClassifier = ClassifierDLApproach() \
    .setInputCols("sentence_embeddings") \
    .setOutputCol("category") \
    .setLabelColumn("label") \
    .setBatchSize(64) \
    .setMaxEpochs(20) \
    .setLr(5e-3) \
    .setDropout(0.5)

pipeline = Pipeline() \
    .setStages(
      [
        documentAssembler,
        useEmbeddings,
        docClassifier
      ]
    )

pipelineModel = pipeline.fit(smallCorpus)

{%- endcapture -%}

{%- capture approach_scala_example -%}
// In this example, the training data `"sentiment.csv"` has the form of
//
// text,label
// This movie is the best movie I have wached ever! In my opinion this movie can win an award.,0
// This was a terrible movie! The acting was bad really bad!,1
// ...
//
// Then traning can be done like so:

import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.embeddings.UniversalSentenceEncoder
import com.johnsnowlabs.nlp.annotators.classifier.dl.ClassifierDLApproach
import org.apache.spark.ml.Pipeline

val smallCorpus = spark.read.option("header","true").csv("src/test/resources/classifier/sentiment.csv")

val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val useEmbeddings = UniversalSentenceEncoder.pretrained()
  .setInputCols("document")
  .setOutputCol("sentence_embeddings")

val docClassifier = new ClassifierDLApproach()
  .setInputCols("sentence_embeddings")
  .setOutputCol("category")
  .setLabelColumn("label")
  .setBatchSize(64)
  .setMaxEpochs(20)
  .setLr(5e-3f)
  .setDropout(0.5f)

val pipeline = new Pipeline()
  .setStages(
    Array(
      documentAssembler,
      useEmbeddings,
      docClassifier
    )
  )

val pipelineModel = pipeline.fit(smallCorpus)

{%- endcapture -%}

{%- capture approach_api_link -%}
[ClassifierDLApproach](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/annotators/classifier/dl/ClassifierDLApproach)
{%- endcapture -%}

{%- capture approach_python_api_link -%}
[ClassifierDLApproach](/api/python/reference/autosummary/python/sparknlp/annotator/classifier_dl/classifier_dl/index.html#sparknlp.annotator.classifier_dl.classifier_dl.ClassifierDLApproach)
{%- endcapture -%}

{%- capture approach_source_link -%}
[ClassifierDLApproach](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/classifier/dl/ClassifierDLApproach.scala)
{%- endcapture -%}


{% include templates/approach_model_template.md
title=title
model_description=model_description
model_input_anno=model_input_anno
model_output_anno=model_output_anno
model_python_example=model_python_example
model_scala_example=model_scala_example
model_python_api_link=model_python_api_link
model_api_link=model_api_link
model_source_link=model_source_link
approach_description=approach_description
approach_input_anno=approach_input_anno
approach_output_anno=approach_output_anno
approach_python_example=approach_python_example
approach_scala_example=approach_scala_example
approach_python_api_link=approach_python_api_link
approach_api_link=approach_api_link
approach_source_link=approach_source_link
approach_note="This annotator accepts a label column of a single item in either type of String, Int, Float, or Double. UniversalSentenceEncoder, BertSentenceEmbeddings, or SentenceEmbeddings can be used for the inputCol"
%}
