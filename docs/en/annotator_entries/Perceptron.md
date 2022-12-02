{%- capture title -%}
POSTagger (Part of speech tagger)
{%- endcapture -%}

{%- capture model_description -%}
Averaged Perceptron model to tag words part-of-speech.
Sets a POS tag to each word within a sentence.

This is the instantiated model of the PerceptronApproach.
For training your own model, please see the documentation of that class.

Pretrained models can be loaded with `pretrained` of the companion object:
```
val posTagger = PerceptronModel.pretrained()
  .setInputCols("document", "token")
  .setOutputCol("pos")
```
The default model is `"pos_anc"`, if no name is provided.

For available pretrained models please see the [Models Hub](https://nlp.johnsnowlabs.com/models?task=Part+of+Speech+Tagging).
Additionally, pretrained pipelines are available for this module, see [Pipelines](https://nlp.johnsnowlabs.com/docs/en/pipelines).

For extended examples of usage, see the [Spark NLP Workshop](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/3.SparkNLP_Pretrained_Models.ipynb).
{%- endcapture -%}

{%- capture model_input_anno -%}
TOKEN, DOCUMENT
{%- endcapture -%}

{%- capture model_output_anno -%}
POS
{%- endcapture -%}

{%- capture model_python_example -%}
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

posTagger = PerceptronModel.pretrained() \
    .setInputCols(["document", "token"]) \
    .setOutputCol("pos")

pipeline = Pipeline().setStages([
    documentAssembler,
    tokenizer,
    posTagger
])

data = spark.createDataFrame([["Peter Pipers employees are picking pecks of pickled peppers"]]).toDF("text")
result = pipeline.fit(data).transform(data)

result.selectExpr("explode(pos) as pos").show(truncate=False)
+-------------------------------------------+
|pos                                        |
+-------------------------------------------+
|[pos, 0, 4, NNP, [word -> Peter], []]      |
|[pos, 6, 11, NNP, [word -> Pipers], []]    |
|[pos, 13, 21, NNS, [word -> employees], []]|
|[pos, 23, 25, VBP, [word -> are], []]      |
|[pos, 27, 33, VBG, [word -> picking], []]  |
|[pos, 35, 39, NNS, [word -> pecks], []]    |
|[pos, 41, 42, IN, [word -> of], []]        |
|[pos, 44, 50, JJ, [word -> pickled], []]   |
|[pos, 52, 58, NNS, [word -> peppers], []]  |
+-------------------------------------------+

{%- endcapture -%}

{%- capture model_scala_example -%}
import spark.implicits._
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.annotators.Tokenizer
import com.johnsnowlabs.nlp.annotators.pos.perceptron.PerceptronModel
import org.apache.spark.ml.Pipeline

val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val tokenizer = new Tokenizer()
  .setInputCols("document")
  .setOutputCol("token")

val posTagger = PerceptronModel.pretrained()
  .setInputCols("document", "token")
  .setOutputCol("pos")

val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
  tokenizer,
  posTagger
))

val data = Seq("Peter Pipers employees are picking pecks of pickled peppers").toDF("text")
val result = pipeline.fit(data).transform(data)

result.selectExpr("explode(pos) as pos").show(false)
+-------------------------------------------+
|pos                                        |
+-------------------------------------------+
|[pos, 0, 4, NNP, [word -> Peter], []]      |
|[pos, 6, 11, NNP, [word -> Pipers], []]    |
|[pos, 13, 21, NNS, [word -> employees], []]|
|[pos, 23, 25, VBP, [word -> are], []]      |
|[pos, 27, 33, VBG, [word -> picking], []]  |
|[pos, 35, 39, NNS, [word -> pecks], []]    |
|[pos, 41, 42, IN, [word -> of], []]        |
|[pos, 44, 50, JJ, [word -> pickled], []]   |
|[pos, 52, 58, NNS, [word -> peppers], []]  |
+-------------------------------------------+

{%- endcapture -%}

{%- capture model_api_link -%}
[PerceptronModel](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/annotators/pos/perceptron/PerceptronModel)
{%- endcapture -%}

{%- capture model_python_api_link -%}
[PerceptronModel](/api/python/reference/autosummary/python/sparknlp/annotator/pos/perceptron/index.html#sparknlp.annotator.pos.perceptron.PerceptronModel)
{%- endcapture -%}

{%- capture model_source_link -%}
[PerceptronModel](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/pos/perceptron/PerceptronModel.scala)
{%- endcapture -%}

{%- capture approach_description -%}
Trains an averaged Perceptron model to tag words part-of-speech.
Sets a POS tag to each word within a sentence.

For pretrained models please see the PerceptronModel.

The training data needs to be in a Spark DataFrame, where the column needs to consist of
[Annotations](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/Annotation) of type `POS`. The `Annotation` needs to have member `result`
set to the POS tag and have a `"word"` mapping to its word inside of member `metadata`.
This DataFrame for training can easily created by the helper class [POS](/docs/en/training#pos-dataset).
```
POS().readDataset(spark, datasetPath).selectExpr("explode(tags) as tags").show(false)
+---------------------------------------------+
|tags                                         |
+---------------------------------------------+
|[pos, 0, 5, NNP, [word -> Pierre], []]       |
|[pos, 7, 12, NNP, [word -> Vinken], []]      |
|[pos, 14, 14, ,, [word -> ,], []]            |
|[pos, 31, 34, MD, [word -> will], []]        |
|[pos, 36, 39, VB, [word -> join], []]        |
|[pos, 41, 43, DT, [word -> the], []]         |
|[pos, 45, 49, NN, [word -> board], []]       |
                      ...
```

For extended examples of usage, see the [Spark NLP Workshop](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/training/french/Train-Perceptron-French.ipynb)
and [PerceptronApproach tests](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/test/scala/com/johnsnowlabs/nlp/annotators/pos/perceptron).
{%- endcapture -%}

{%- capture approach_input_anno -%}
TOKEN, DOCUMENT
{%- endcapture -%}

{%- capture approach_output_anno -%}
POS
{%- endcapture -%}

{%- capture approach_python_example -%}
import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from sparknlp.training import *
from pyspark.ml import Pipeline

documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

sentence = SentenceDetector() \
    .setInputCols(["document"]) \
    .setOutputCol("sentence")

tokenizer = Tokenizer() \
    .setInputCols(["sentence"]) \
    .setOutputCol("token")

datasetPath = "src/test/resources/anc-pos-corpus-small/test-training.txt"
trainingPerceptronDF = POS().readDataset(spark, datasetPath)

trainedPos = PerceptronApproach() \
    .setInputCols(["document", "token"]) \
    .setOutputCol("pos") \
    .setPosColumn("tags") \
    .fit(trainingPerceptronDF)

pipeline = Pipeline().setStages([
    documentAssembler,
    sentence,
    tokenizer,
    trainedPos
])

data = spark.createDataFrame([["To be or not to be, is this the question?"]]).toDF("text")
result = pipeline.fit(data).transform(data)

result.selectExpr("pos.result").show(truncate=False)
+--------------------------------------------------+
|result                                            |
+--------------------------------------------------+
|[NNP, NNP, CD, JJ, NNP, NNP, ,, MD, VB, DT, CD, .]|
+--------------------------------------------------+

{%- endcapture -%}

{%- capture approach_scala_example -%}
import spark.implicits._
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.annotator.SentenceDetector
import com.johnsnowlabs.nlp.annotators.Tokenizer
import com.johnsnowlabs.nlp.training.POS
import com.johnsnowlabs.nlp.annotators.pos.perceptron.PerceptronApproach
import org.apache.spark.ml.Pipeline

val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val sentence = new SentenceDetector()
  .setInputCols("document")
  .setOutputCol("sentence")

val tokenizer = new Tokenizer()
  .setInputCols("sentence")
  .setOutputCol("token")

val datasetPath = "src/test/resources/anc-pos-corpus-small/test-training.txt"
val trainingPerceptronDF = POS().readDataset(spark, datasetPath)

val trainedPos = new PerceptronApproach()
  .setInputCols("document", "token")
  .setOutputCol("pos")
  .setPosColumn("tags")
  .fit(trainingPerceptronDF)

val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
  sentence,
  tokenizer,
  trainedPos
))

val data = Seq("To be or not to be, is this the question?").toDF("text")
val result = pipeline.fit(data).transform(data)

result.selectExpr("pos.result").show(false)
+--------------------------------------------------+
|result                                            |
+--------------------------------------------------+
|[NNP, NNP, CD, JJ, NNP, NNP, ,, MD, VB, DT, CD, .]|
+--------------------------------------------------+

{%- endcapture -%}

{%- capture approach_api_link -%}
[PerceptronApproach](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/annotators/pos/perceptron/PerceptronApproach)
{%- endcapture -%}

{%- capture approach_python_api_link -%}
[PerceptronApproach](/api/python/reference/autosummary/python/sparknlp/annotator/pos/perceptron/index.html#sparknlp.annotator.pos.perceptron.PerceptronApproach)
{%- endcapture -%}

{%- capture approach_source_link -%}
[PerceptronApproach](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/pos/perceptron/PerceptronApproach.scala)
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
