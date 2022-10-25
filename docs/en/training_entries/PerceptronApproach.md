{%- capture title -%}
PerceptronApproach (Part of speech tagger)
{%- endcapture -%}

{%- capture description -%}
Trains an averaged Perceptron model to tag words part-of-speech.
Sets a POS tag to each word within a sentence.

The training data needs to be in a Spark DataFrame, where the column needs to consist of
[Annotations](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/Annotation) of type `POS`. The `Annotation` needs to have member `result`
set to the POS tag and have a `"word"` mapping to its word inside of member `metadata`.
This DataFrame for training can easily created by the helper class [POS](#pos-dataset).

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

{%- capture input_anno -%}
TOKEN, DOCUMENT
{%- endcapture -%}

{%- capture output_anno -%}
POS
{%- endcapture -%}

{%- capture python_example -%}
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

{%- capture scala_example -%}
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

{%- capture api_link -%}
[PerceptronApproach](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/annotators/pos/perceptron/PerceptronApproach)
{%- endcapture -%}

{%- capture python_api_link -%}
[PerceptronApproach](/api/python/reference/autosummary/python/sparknlp/annotator/pos/perceptron/index.html#sparknlp.annotator.pos.perceptron.PerceptronApproach)
{%- endcapture -%}

{%- capture source_link -%}
[PerceptronApproach](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/pos/perceptron/PerceptronApproach.scala)
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

