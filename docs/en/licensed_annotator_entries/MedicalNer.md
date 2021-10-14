{%- capture title -%}
MedicalNer
{%- endcapture -%}

{%- capture model_description -%}
This Named Entity recognition annotator is a generic NER model based on Neural Networks.


Pretrained models can be loaded with `pretrained` of the companion object:
```
val nerModel = NerDLModel.pretrained()
  .setInputCols("sentence", "token", "embeddings")
  .setOutputCol("ner")
```
The default model is `"ner_clinical"`, if no name is provided.

For available pretrained models please see the [Models Hub](https://nlp.johnsnowlabs.com/models?task=Named+Entity+Recognition).
Additionally, pretrained pipelines are available for this module, see [Pipelines](https://nlp.johnsnowlabs.com/docs/en/pipelines).

Note that some pretrained models require specific types of embeddings, depending on which they were trained on.
For example, the default model `"ner_dl"` requires the
[WordEmbeddings](/docs/en/annotators#wordembeddings) `"ner_clinical"`.


For extended examples of usage, see the [Spark NLP Workshop](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/1.Clinical_Named_Entity_Recognition_Model.ipynb)
(sections starting with `Training a Clinical NER`)

{%- endcapture -%}

{%- capture model_input_anno -%}
DOCUMENT, TOKEN, WORD_EMBEDDINGS
{%- endcapture -%}

{%- capture model_output_anno -%}
NAMED_ENTITY
{%- endcapture -%}

{%- capture model_python_example -%}
import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline

# First extract the prerequisites for the NerDLModel
from sparknlp.annotator import *
from sparknlp_jsl.annotator import *
from sparknlp.base import *
import sparknlp
from pyspark.ml import Pipeline

# First extract the prerequisites for the NerDLApproach
documentAssembler = DocumentAssembler() \
.setInputCol("text") \
.setOutputCol("document")

sentence = SentenceDetector() \
.setInputCols(["document"]) \
.setOutputCol("sentence")

tokenizer = Tokenizer() \
.setInputCols(["sentence"]) \
.setOutputCol("token")

clinical_embeddings = WordEmbeddingsModel.pretrained('embeddings_clinical', "en", "clinical/models")\
.setInputCols(["sentence", "token"])\
.setOutputCol("embeddings")

# Then the training can start
nerTagger = MedicalNerApproach()\
.setInputCols(["sentence", "token", "embeddings"])\
.setLabelColumn("label")\
.setOutputCol("ner")\
.setMaxEpochs(2)\
.setBatchSize(64)\
.setRandomSeed(0)\
.setVerbose(1)\
.setValidationSplit(0.2)\
.setEvaluationLogExtended(True) \
.setEnableOutputLogs(True)\
.setIncludeConfidence(True)\
.setOutputLogsPath('ner_logs')\
.setGraphFolder('medical_ner_graphs')\
.setEnableMemoryOptimizer(True) #>> if you have a limited memory and a large conll file, you can set this True to train batch by batch

pipeline = Pipeline().setStages([
documentAssembler,
sentence,
tokenizer,
clinical_embeddings,
nerTagger
])


conll = CoNLL()
conll_data = CoNLL().readDataset(spark, 'NER_NCBIconlltrain.txt')

result = pipeline.fit(data).transform(data)



{%- endcapture -%}

{%- capture model_scala_example -%}
import spark.implicits._
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.annotators.Tokenizer
import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector
import com.johnsnowlabs.nlp.embeddings.WordEmbeddingsModel
import com.johnsnowlabs.nlp.annotators.ner.dl.MedicalNerModel
import org.apache.spark.ml.Pipeline

// First extract the prerequisites for the NerDLModel
val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val sentence = new SentenceDetector()
  .setInputCols("document")
  .setOutputCol("sentence")

val tokenizer = new Tokenizer()
  .setInputCols("sentence")
  .setOutputCol("token")

val embeddings = WordEmbeddingsModel
   .pretrained("embeddings_clinical", "en", "clinical/models")
   .setInputCols(Array("sentences", "tokens"))
   .setOutputCol("embeddings")

// Then NER can be extracted
val nerTagger = MedicalNerModel .pretrained()
  .setInputCols("sentence", "token", "embeddings")
  .setOutputCol("ner")

val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
  sentence,
  tokenizer,
  embeddings,
  nerTagger
))

val data = Seq("A 28-year-old female with a history of gestational diabetes mellitus diagnosed eight years prior to" +
" presentation and subsequent type two diabetes mellitus ( T2DM ), one prior episode of HTG-induced." ).toDF("text")
val result = pipeline.fit(data).transform(data)

conll = CoNLL()
conll_data = CoNLL().readDataset(spark, 'NER_NCBIconlltrain.txt')

ner_model = ner_pipeline.fit(conll_data)

{%- endcapture -%}

{%- capture model_api_link -%}
[MedicalNerModel](https://nlp.johnsnowlabs.com/licensed/api/com/johnsnowlabs/nlp/annotators/ner/MedicalNerModel.html)
{%- endcapture -%}


{%- capture approach_description -%}
This Named Entity recognition annotator allows to train generic NER model based on Neural Networks.

The architecture of the neural network is a Char CNNs - BiLSTM - CRF that achieves state-of-the-art in most datasets.

For instantiated/pretrained models, see NerDLModel.

The training data should be a labeled Spark Dataset, in the format of [CoNLL](/docs/en/training#conll-dataset)
2003 IOB with `Annotation` type columns. The data should have columns of type `DOCUMENT, TOKEN, WORD_EMBEDDINGS` and an
additional label column of annotator type `NAMED_ENTITY`.
Excluding the label, this can be done with for example
- a [SentenceDetector](/docs/en/annotators#sentencedetector),
- a [Tokenizer](/docs/en/annotators#tokenizer) and
- a [WordEmbeddingsModel](/docs/en/annotators#wordembeddings) with clinical embeddings
  (any [clinical word embeddings](https://nlp.johnsnowlabs.com/models?task=Embeddings) can be chosen).

For extended examples of usage, see the [Spark NLP Workshop](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/1.Clinical_Named_Entity_Recognition_Model.ipynb)
(sections starting with `Training a Clinical NER`)

{%- endcapture -%}



{%- capture approach_input_anno -%}
DOCUMENT, TOKEN, WORD_EMBEDDINGS
{%- endcapture -%}

{%- capture approach_output_anno -%}
NAMED_ENTITY
{%- endcapture -%}

{%- capture approach_python_example -%}
import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from sparknlp_jsl.annotator import *
from sparknlp.training import *
from pyspark.ml import Pipeline

# First extract the prerequisites for the NerDLApproach
documentAssembler = DocumentAssembler() \
.setInputCol("text") \
.setOutputCol("document")

sentence = SentenceDetector() \
.setInputCols(["document"]) \
.setOutputCol("sentence")

tokenizer = Tokenizer() \
.setInputCols(["sentence"]) \
.setOutputCol("token")

clinical_embeddings = WordEmbeddingsModel.pretrained('embeddings_clinical', "en", "clinical/models")\
.setInputCols(["sentence", "token"])\
.setOutputCol("embeddings")

# Then the training can start
nerTagger = MedicalNerApproach()\
.setInputCols(["sentence", "token", "embeddings"])\
.setLabelColumn("label")\
.setOutputCol("ner")\
.setMaxEpochs(2)\
.setBatchSize(64)\
.setRandomSeed(0)\
.setVerbose(1)\
.setValidationSplit(0.2)\
.setEvaluationLogExtended(True) \
.setEnableOutputLogs(True)\
.setIncludeConfidence(True)\
.setOutputLogsPath('ner_logs')\
.setGraphFolder('medical_ner_graphs')\
.setEnableMemoryOptimizer(True) #>> if you have a limited memory and a large conll file, you can set this True to train batch by batch

pipeline = Pipeline().setStages([
documentAssembler,
sentence,
tokenizer,
clinical_embeddings,
nerTagger
])

# We use the text and labels from the CoNLL dataset
conll = CoNLL()
trainingData = conll.readDataset(spark, "src/test/resources/conll2003/eng.train")

pipelineModel = pipeline.fit(trainingData)

{%- endcapture -%}

{%- capture approach_scala_example -%}
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.annotators.Tokenizer
import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector
import com.johnsnowlabs.nlp.embeddings.WordEmbeddingsModel
import com.johnsnowlabs.nlp.annotators.ner.MedicalNerApproach
import com.johnsnowlabs.nlp.training.CoNLL
import org.apache.spark.ml.Pipeline

// First extract the prerequisites for the NerDLApproach
val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val sentence = new SentenceDetector()
  .setInputCols("document")
  .setOutputCol("sentence")

val tokenizer = new Tokenizer()
  .setInputCols("sentence")
  .setOutputCol("token")

val embeddings = BertEmbeddings.pretrained()
  .setInputCols("sentence", "token")
  .setOutputCol("embeddings")

// Then the training can start
val nerTagger =new MedicalNerApproach()
.setInputCols(Array("sentence", "token", "embeddings"))
.setLabelColumn("label")
.setOutputCol("ner")
.setMaxEpochs(5)
.setLr(0.003f)
.setBatchSize(8)
.setRandomSeed(0)
.setVerbose(1)
.setEvaluationLogExtended(false)
.setEnableOutputLogs(false)
.setIncludeConfidence(true)

val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
  sentence,
  tokenizer,
  embeddings,
  nerTagger
))

// We use the text and labels from the CoNLL dataset
val conll = CoNLL()
val trainingData = conll.readDataset(spark, "src/test/resources/conll2003/eng.train")

val pipelineModel = pipeline.fit(trainingData)

{%- endcapture -%}

{%- capture approach_api_link -%}
[MedicalNerApproach](https://nlp.johnsnowlabs.com/licensed/api/com/johnsnowlabs/nlp/annotators/ner/MedicalNerApproach.html)
{%- endcapture -%}




{% include templates/licensed_approach_model_template.md
title=title
model_description=model_description
model_input_anno=model_input_anno
model_output_anno=model_output_anno
model_python_example=model_python_example
model_scala_example=model_scala_example
model_api_link=model_api_link
approach_description=approach_description
approach_input_anno=approach_input_anno
approach_output_anno=approach_output_anno
approach_python_example=approach_python_example
approach_scala_example=approach_scala_example
approach_api_link=approach_api_link
%}
