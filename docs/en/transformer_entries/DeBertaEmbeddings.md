{%- capture title -%}
DeBertaEmbeddings
{%- endcapture -%}

{%- capture description -%}
The DeBERTa model was proposed in [DeBERTa: Decoding-enhanced BERT with Disentangled Attention](https://arxiv.org/abs/2006.03654) by Pengcheng He, Xiaodong Liu, Jianfeng Gao, Weizhu Chen. It is based on Google’s BERT model released in 2018 and Facebook’s RoBERTa model released in 2019.

This model requires input tokenization with SentencePiece model, which is provided by Spark NLP (See tokenizers package).

Pretrained models can be loaded with `pretrained` of the companion object:
```
val embeddings = DeBertaEmbeddings.pretrained()
 .setInputCols("sentence", "token")
 .setOutputCol("embeddings")
```
The default model is `"deberta_v3_base"`, if no name is provided.

For extended examples see [DeBertaEmbeddingsTestSpec](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/embeddings/DeBertaEmbeddingsTestSpec.scala).
Models from the HuggingFace 🤗 Transformers library are also compatible with Spark NLP 🚀. The Spark NLP Workshop
example shows how to import them https://github.com/JohnSnowLabs/spark-nlp/discussions/5669.

It builds on RoBERTa with disentangled attention and enhanced mask decoder training with half of the data used in RoBERTa.

**Sources:**

https://github.com/microsoft/DeBERTa

https://www.microsoft.com/en-us/research/blog/microsoft-deberta-surpasses-human-performance-on-the-superglue-benchmark/

**Paper abstract:**

*Recent progress in pre-trained neural language models has significantly improved the performance of many natural language processing (NLP) tasks. In this paper we propose a new model architecture DeBERTa (Decoding-enhanced BERT with disentangled attention) that improves the BERT and RoBERTa models using two novel techniques. The first is the disentangled attention mechanism, where each word is represented using two vectors that encode its content and position, respectively, and the attention weights among words are computed using disentangled matrices on their contents and relative positions. Second, an enhanced mask decoder is used to replace the output softmax layer to predict the masked tokens for model pretraining. We show that these two techniques significantly improve the efficiency of model pretraining and performance of downstream tasks. Compared to RoBERTa-Large, a DeBERTa model trained on half of the training data performs consistently better on a wide range of NLP tasks, achieving improvements on MNLI by +0.9% (90.2% vs. 91.1%), on SQuAD v2.0 by +2.3% (88.4% vs. 90.7%) and RACE by +3.6% (83.2% vs. 86.8%). The DeBERTa code and pre-trained models will be made publicly available at https://github.com/microsoft/DeBERTa.*
{%- endcapture -%}

{%- capture input_anno -%}
DOCUMENT, TOKEN
{%- endcapture -%}

{%- capture output_anno -%}
WORD_EMBEDDINGS
{%- endcapture -%}

{%- capture embeddings_python_example -%}
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

embeddings = DeBertaEmbeddings.pretrained() \
    .setInputCols(["token", "document"]) \
    .setOutputCol("embeddings")

embeddingsFinisher = EmbeddingsFinisher() \
    .setInputCols(["embeddings"]) \
    .setOutputCols("finished_embeddings") \
    .setOutputAsVector(True) \
    .setCleanAnnotations(False)

pipeline = Pipeline().setStages([
    documentAssembler,
    tokenizer,
    embeddings,
    embeddingsFinisher
])

data = spark.createDataFrame([["This is a sentence."]]).toDF("text")
result = pipeline.fit(data).transform(data)

result.selectExpr("explode(finished_embeddings) as result").show(5, 80)
+--------------------------------------------------------------------------------+
|                                                                          result|
+--------------------------------------------------------------------------------+
|[1.1342473030090332,-1.3855540752410889,0.9818322062492371,-0.784737348556518...|
|[0.847029983997345,-1.047153353691101,-0.1520637571811676,-0.6245765686035156...|
|[-0.009860038757324219,-0.13450059294700623,2.707749128341675,1.2916892766952...|
|[-0.04192575812339783,-0.5764210224151611,-0.3196685314178467,-0.527840495109...|
|[0.15583214163780212,-0.1614152491092682,-0.28423872590065,-0.135491415858268...|
+--------------------------------------------------------------------------------+

{%- endcapture -%}

{%- capture embeddings_scala_example -%}
import spark.implicits._
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.annotators.Tokenizer
import com.johnsnowlabs.nlp.embeddings.DeBertaEmbeddings
import com.johnsnowlabs.nlp.EmbeddingsFinisher
import org.apache.spark.ml.Pipeline

val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val tokenizer = new Tokenizer()
  .setInputCols("document")
  .setOutputCol("token")

val embeddings = DeBertaEmbeddings.pretrained()
  .setInputCols("token", "document")
  .setOutputCol("embeddings")

val embeddingsFinisher = new EmbeddingsFinisher()
  .setInputCols("embeddings")
  .setOutputCols("finished_embeddings")
  .setOutputAsVector(true)
  .setCleanAnnotations(false)

val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
  tokenizer,
  embeddings,
  embeddingsFinisher
))

val data = Seq("This is a sentence.").toDF("text")
val result = pipeline.fit(data).transform(data)

result.selectExpr("explode(finished_embeddings) as result").show(5, 80)
+--------------------------------------------------------------------------------+
|                                                                          result|
+--------------------------------------------------------------------------------+
|[1.1342473030090332,-1.3855540752410889,0.9818322062492371,-0.784737348556518...|
|[0.847029983997345,-1.047153353691101,-0.1520637571811676,-0.6245765686035156...|
|[-0.009860038757324219,-0.13450059294700623,2.707749128341675,1.2916892766952...|
|[-0.04192575812339783,-0.5764210224151611,-0.3196685314178467,-0.527840495109...|
|[0.15583214163780212,-0.1614152491092682,-0.28423872590065,-0.135491415858268...|
+--------------------------------------------------------------------------------+

{%- endcapture -%}

{%- capture prediction_python_example -%}
# Coming Soon!
{%- endcapture -%}

{%- capture prediction_scala_example -%}
// Coming Soon!
{%- endcapture -%}

{%- capture training_python_example -%}
import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
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

embeddings = DeBertaEmbeddings.pretrained() \
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("embeddings")

# Then the training can start with the transformer embeddings
nerTagger = NerDLApproach() \
    .setInputCols(["sentence", "token", "embeddings"]) \
    .setLabelColumn("label") \
    .setOutputCol("ner") \
    .setMaxEpochs(1) \
    .setVerbose(0)

pipeline = Pipeline().setStages([
    documentAssembler,
    sentence,
    tokenizer,
    embeddings,
    nerTagger
])

# We use the text and labels from the CoNLL dataset
conll = CoNLL()
trainingData = conll.readDataset(spark, "eng.train")

pipelineModel = pipeline.fit(trainingData)
{%- endcapture -%}

{%- capture training_scala_example -%}
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.annotator._
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

val embeddings = DeBertaEmbeddings.pretrained()
  .setInputCols("sentence", "token")
  .setOutputCol("embeddings")

// Then the training can start with the transformer embeddings
val nerTagger = new NerDLApproach()
  .setInputCols("sentence", "token", "embeddings")
  .setLabelColumn("label")
  .setOutputCol("ner")
  .setMaxEpochs(1)
  .setVerbose(0)

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

{%- capture api_link -%}
[DeBertaEmbeddings](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/embeddings/DeBertaEmbeddings)
{%- endcapture -%}

{%- capture python_api_link -%}
[DeBertaEmbeddings](/api/python/reference/autosummary/python/sparknlp/annotator/embeddings/deberta_embeddings/index.html#sparknlp.annotator.embeddings.deberta_embeddings.DeBertaEmbeddings)
{%- endcapture -%}

{%- capture source_link -%}
[DeBertaEmbeddings](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/embeddings/DeBertaEmbeddings.scala)
{%- endcapture -%}

{% include templates/transformer_usecases_template.md
title=title
description=description
input_anno=input_anno
output_anno=output_anno
python_api_link=python_api_link
api_link=api_link
source_link=source_link
prediction_python_example=prediction_python_example
prediction_scala_example=prediction_scala_example
training_python_example=training_python_example
training_scala_example=training_scala_example
embeddings_python_example=embeddings_python_example
embeddings_scala_example=embeddings_scala_example
%}