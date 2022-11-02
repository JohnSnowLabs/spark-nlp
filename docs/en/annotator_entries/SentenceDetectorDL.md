{%- capture title -%}
SentenceDetectorDL
{%- endcapture -%}

{%- capture model_description -%}
Annotator that detects sentence boundaries using a deep learning approach.

Instantiated Model of the SentenceDetectorDLApproach.
Detects sentence boundaries using a deep learning approach.

Pretrained models can be loaded with `pretrained` of the companion object:
```
val sentenceDL = SentenceDetectorDLModel.pretrained()
  .setInputCols("document")
  .setOutputCol("sentencesDL")
```
The default model is `"sentence_detector_dl"`, if no name is provided.
For available pretrained models please see the [Models Hub](https://nlp.johnsnowlabs.com/models?task=Sentence+Detection).

Each extracted sentence can be returned in an Array or exploded to separate rows,
if `explodeSentences` is set to `true`.

For extended examples of usage, see the [Spark NLP Workshop](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/2.Text_Preprocessing_with_SparkNLP_Annotators_Transformers.ipynb)
and the [SentenceDetectorDLSpec](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/sentence_detector_dl/SentenceDetectorDLSpec.scala).
{%- endcapture -%}

{%- capture model_input_anno -%}
DOCUMENT
{%- endcapture -%}

{%- capture model_output_anno -%}
DOCUMENT
{%- endcapture -%}

{%- capture model_python_example -%}
import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline
# In this example, the normal `SentenceDetector` is compared to the `SentenceDetectorDLModel`. In a pipeline,
# `SentenceDetectorDLModel` can be used as a replacement for the `SentenceDetector`.

documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

sentence = SentenceDetector() \
    .setInputCols(["document"]) \
    .setOutputCol("sentences")

sentenceDL = SentenceDetectorDLModel \
    .pretrained("sentence_detector_dl", "en") \
    .setInputCols(["document"]) \
    .setOutputCol("sentencesDL")

pipeline = Pipeline().setStages([
    documentAssembler,
    sentence,
    sentenceDL
])

data = spark.createDataFrame([["""John loves Mary.Mary loves Peter
    Peter loves Helen .Helen loves John;
    Total: four people involved."""]]).toDF("text")
result = pipeline.fit(data).transform(data)

result.selectExpr("explode(sentences.result) as sentences").show(truncate=False)
+----------------------------------------------------------+
|sentences                                                 |
+----------------------------------------------------------+
|John loves Mary.Mary loves Peter\n     Peter loves Helen .|
|Helen loves John;                                         |
|Total: four people involved.                              |
+----------------------------------------------------------+

result.selectExpr("explode(sentencesDL.result) as sentencesDL").show(truncate=False)
+----------------------------+
|sentencesDL                 |
+----------------------------+
|John loves Mary.            |
|Mary loves Peter            |
|Peter loves Helen .         |
|Helen loves John;           |
|Total: four people involved.|
+----------------------------+

{%- endcapture -%}

{%- capture model_scala_example -%}
// In this example, the normal `SentenceDetector` is compared to the `SentenceDetectorDLModel`. In a pipeline,
// `SentenceDetectorDLModel` can be used as a replacement for the `SentenceDetector`.
import spark.implicits._
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.annotator.SentenceDetector
import com.johnsnowlabs.nlp.annotators.sentence_detector_dl.SentenceDetectorDLModel
import org.apache.spark.ml.Pipeline

val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val sentence = new SentenceDetector()
  .setInputCols("document")
  .setOutputCol("sentences")

val sentenceDL = SentenceDetectorDLModel
  .pretrained("sentence_detector_dl", "en")
  .setInputCols("document")
  .setOutputCol("sentencesDL")

val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
  sentence,
  sentenceDL
))

val data = Seq("""John loves Mary.Mary loves Peter
  Peter loves Helen .Helen loves John;
  Total: four people involved.""").toDF("text")
val result = pipeline.fit(data).transform(data)

result.selectExpr("explode(sentences.result) as sentences").show(false)
+----------------------------------------------------------+
|sentences                                                 |
+----------------------------------------------------------+
|John loves Mary.Mary loves Peter\n     Peter loves Helen .|
|Helen loves John;                                         |
|Total: four people involved.                              |
+----------------------------------------------------------+

result.selectExpr("explode(sentencesDL.result) as sentencesDL").show(false)
+----------------------------+
|sentencesDL                 |
+----------------------------+
|John loves Mary.            |
|Mary loves Peter            |
|Peter loves Helen .         |
|Helen loves John;           |
|Total: four people involved.|
+----------------------------+

{%- endcapture -%}

{%- capture model_api_link -%}
[SentenceDetectorDLModel](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/annotators/sentence_detector_dl/SentenceDetectorDLModel)
{%- endcapture -%}

{%- capture model_python_api_link -%}
[SentenceDetectorDLModel](/api/python/reference/autosummary/python/sparknlp/annotator/sentence/sentence_detector_dl/index.html#sparknlp.annotator.sentence.sentence_detector_dl.SentenceDetectorDLModel)
{%- endcapture -%}

{%- capture model_source_link -%}
[SentenceDetectorDLModel](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/sentence_detector_dl/SentenceDetectorDLModel.scala)
{%- endcapture -%}

{%- capture approach_description -%}
Trains an annotator that detects sentence boundaries using a deep learning approach.

For pretrained models see SentenceDetectorDLModel.

Currently, only the CNN model is supported for training, but in the future the architecture of the model can
be set with `setModelArchitecture`.

The default model `"cnn"` is based on the paper
[Deep-EOS: General-Purpose Neural Networks for Sentence Boundary Detection (2020, Stefan Schweter, Sajawel Ahmed)](https://konvens.org/proceedings/2019/papers/KONVENS2019_paper_41.pdf)
using a CNN architecture. We also modified the original implementation a little bit to cover broken sentences and some impossible end of line chars.

Each extracted sentence can be returned in an Array or exploded to separate rows,
if `explodeSentences` is set to `true`.

For extended examples of usage, see the [Spark NLP Workshop](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/9.SentenceDetectorDL.ipynb) and the [SentenceDetectorDLSpec](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/sentence_detector_dl/SentenceDetectorDLSpec.scala).
{%- endcapture -%}

{%- capture approach_input_anno -%}
DOCUMENT
{%- endcapture -%}

{%- capture approach_output_anno -%}
DOCUMENT
{%- endcapture -%}

{%- capture approach_python_example -%}
# The training process needs data, where each data point is a sentence.
# In this example the `train.txt` file has the form of
#
# ...
# Slightly more moderate language would make our present situation – namely the lack of progress – a little easier.
# His political successors now have great responsibilities to history and to the heritage of values bequeathed to them by Nelson Mandela.
# ...
#
# where each line is one sentence.
# Training can then be started like so:

import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline

trainingData = spark.read.text("train.txt").toDF("text")

documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

sentenceDetector = SentenceDetectorDLApproach() \
    .setInputCols(["document"]) \
    .setOutputCol("sentences") \
    .setEpochsNumber(100)

pipeline = Pipeline().setStages([documentAssembler, sentenceDetector])

model = pipeline.fit(trainingData)

{%- endcapture -%}

{%- capture approach_scala_example -%}
// The training process needs data, where each data point is a sentence.
// In this example the `train.txt` file has the form of
//
// ...
// Slightly more moderate language would make our present situation – namely the lack of progress – a little easier.
// His political successors now have great responsibilities to history and to the heritage of values bequeathed to them by Nelson Mandela.
// ...
//
// where each line is one sentence.
// Training can then be started like so:
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.annotators.sentence_detector_dl.SentenceDetectorDLApproach
import org.apache.spark.ml.Pipeline

val trainingData = spark.read.text("train.txt").toDF("text")

val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val sentenceDetector = new SentenceDetectorDLApproach()
  .setInputCols(Array("document"))
  .setOutputCol("sentences")
  .setEpochsNumber(100)

val pipeline = new Pipeline().setStages(Array(documentAssembler, sentenceDetector))

val model = pipeline.fit(trainingData)

{%- endcapture -%}

{%- capture approach_api_link -%}
[SentenceDetectorDLApproach](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/annotators/sentence_detector_dl/SentenceDetectorDLApproach)
{%- endcapture -%}

{%- capture approach_python_api_link -%}
[SentenceDetectorDLApproach](/api/python/reference/autosummary/python/sparknlp/annotator/sentence/sentence_detector_dl/index.html#sparknlp.annotator.sentence.sentence_detector_dl.SentenceDetectorDLApproach)
{%- endcapture -%}

{%- capture approach_source_link -%}
[SentenceDetectorDLApproach](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/sentence_detector_dl/SentenceDetectorDLApproach.scala)
{%- endcapture -%}


{% include templates/approach_model_template.md
title=title
model_description=model_description
model_input_anno=model_input_anno
model_output_anno=model_output_anno
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
%}
