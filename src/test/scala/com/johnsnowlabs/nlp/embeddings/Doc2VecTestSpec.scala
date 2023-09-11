/*
 * Copyright 2017-2022 John Snow Labs
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.johnsnowlabs.nlp.embeddings

import com.johnsnowlabs.nlp.annotator._
import com.johnsnowlabs.nlp.annotators.SparkSessionTest
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.training.CoNLL
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.tags.{FastTest, SlowTest}
import com.johnsnowlabs.util.Benchmark
import org.apache.spark.ml.Pipeline
import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics, MulticlassMetrics}
import org.apache.spark.sql.functions.{explode, when}
import org.scalatest.flatspec.AnyFlatSpec

class Doc2VecTestSpec extends AnyFlatSpec with SparkSessionTest {

  "Doc2VecApproach" should "train, save, and load back the saved model" taggedAs FastTest in {

    import ResourceHelper.spark.implicits._

    val ddd = Seq(
      "Rare Hendrix song draft sells for almost $17,000. This is my second sentenece! The third one here!",
      "EU rejects German call to boycott British lamb .",
      "TORONTO 1996-08-21",
      " carbon emissions have come down without impinging on our growth . . .",
      "carbon emissions have come down without impinging on our growth .\\u2009.\\u2009.",
      "the ",
      "  ",
      " ").toDF("text")

    val stops = new StopWordsCleaner()
      .setInputCols("token")
      .setOutputCol("cleanedToken")
      .setStopWords(Array("the"))

    val doc2Vec = new Doc2VecApproach()
      .setInputCols("cleanedToken")
      .setOutputCol("sentence_embeddings")
      .setMaxSentenceLength(512)
      .setStorageRef("my_awesome_doc2vec")
      .setEnableCaching(true)

    val pipeline = new Pipeline().setStages(
      Array(documentAssembler, sentenceDetector, tokenizerWithSentence, stops, doc2Vec))

    val pipelineModel = pipeline.fit(ddd)
    val pipelineDF = pipelineModel.transform(ddd)

    pipelineDF.select("sentence_embeddings").show()

    pipelineModel.write.overwrite().save("./tmp_pipeline_doc2vec")
    pipelineModel.stages.last
      .asInstanceOf[Doc2VecModel]
      .write
      .overwrite()
      .save("./tmp_doc2vec_model")

    val loadedDoc2Vec = Doc2VecModel
      .load("./tmp_doc2vec_model")
      .setInputCols("token")
      .setOutputCol("sentence_embeddings")

    val loadedPipeline =
      new Pipeline().setStages(
        Array(documentAssembler, sentenceDetector, tokenizerWithSentence, loadedDoc2Vec))

    loadedPipeline.fit(ddd).transform(ddd).select("sentence_embeddings").show()

  }

  "Doc2VecModel" should "work with sentence and document" taggedAs FastTest in {

    import ResourceHelper.spark.implicits._

    val ddd = Seq(
      "Rare Hendrix song draft sells for almost $17,000. This is my second sentenece! The third one here!",
      "EU rejects German call to boycott British lamb .",
      "TORONTO 1996-08-21",
      " carbon emissions have come down without impinging on our growth . . .",
      "carbon emissions have come down without impinging on our growth .\\u2009.\\u2009.",
      "the ").toDF("text")

    val setence = new SentenceDetector()
      .setInputCols("document")
      .setOutputCol("sentence")

    val tokenizerDocument = new Tokenizer()
      .setInputCols(Array("document"))
      .setOutputCol("token_document")

    val tokenizerSentence = new Tokenizer()
      .setInputCols(Array("sentence"))
      .setOutputCol("token_sentence")

    val doc2VecDocument = new Doc2VecApproach()
      .setInputCols("token_document")
      .setOutputCol("document_embeddings")
      .setMaxSentenceLength(512)
      .setStorageRef("my_awesome_doc2vec")

    val doc2VecSentence = new Doc2VecApproach()
      .setInputCols("token_sentence")
      .setOutputCol("sentence_embeddings")
      .setMaxSentenceLength(512)
      .setStorageRef("my_awesome_doc2vec")

    val pipeline = new Pipeline().setStages(
      Array(
        documentAssembler,
        setence,
        tokenizerDocument,
        tokenizerSentence,
        doc2VecDocument,
        doc2VecSentence))

    val pipelineModel = pipeline.fit(ddd)
    val pipelineDF = pipelineModel.transform(ddd)

    val totalSentences = pipelineDF.select(explode($"sentence.result")).count.toInt
    val totalSentEmbeddings =
      pipelineDF.select(explode($"sentence_embeddings.embeddings")).count.toInt

    println(s"total sentences: $totalSentences")
    println(s"total sentence embeddings: $totalSentEmbeddings")

    assert(totalSentences == totalSentEmbeddings)

    val totalDocs = pipelineDF.select(explode($"document.result")).count.toInt
    val totalDocEmbeddings =
      pipelineDF.select(explode($"document_embeddings.embeddings")).count.toInt

    println(s"total documents: $totalDocs")
    println(s"total document embeddings: $totalDocEmbeddings")

    assert(totalDocs == totalDocEmbeddings)
  }

  "Doc2VecModel" should "Benchmark" taggedAs SlowTest in {

    val conll = CoNLL()
    val training_data = conll
      .readDataset(ResourceHelper.spark, "src/test/resources/conll2003/eng.train")
      .repartition(12)
    val test_data = conll
      .readDataset(ResourceHelper.spark, "src/test/resources/conll2003/eng.testa")
      .repartition(12)

    println(training_data.count())

    val doc2Vec = new Doc2VecApproach()
      .setInputCols("token")
      .setOutputCol("sentence_embeddings")
      .setMaxSentenceLength(1024)
      .setStepSize(0.001)
      .setMinCount(10)
      .setVectorSize(100)
      .setNumPartitions(1)
      .setMaxIter(4)
      .setSeed(42)
      .setStorageRef("doc2vec_conll_03")

    val pipeline = new Pipeline().setStages(Array(doc2Vec))

    val pipelineModel = pipeline.fit(training_data)

    Benchmark.time("Time to save Doc2Vec results") {
      pipelineModel
        .transform(training_data)
        .write
        .mode("overwrite")
        .parquet("./tmp_doc2vec_pipeline")
    }

    Benchmark.time("Time to save Doc2Vec results") {
      pipelineModel.transform(test_data).write.mode("overwrite").parquet("./tmp_doc2vec_pipeline")
    }

  }

  "Doc2VecModel" should "train classifierdl" taggedAs SlowTest in {

    val spark = ResourceHelper.spark
    import spark.implicits._

    val train = spark.read
      .parquet("src/test/resources/aclImdb/train")

    val test = spark.read
      .parquet("src/test/resources/aclImdb/test")

    println("count of training dataset: ", train.count)
    println("count of test dataset: ", test.count)

    train.groupBy("label").count().show()
    test.groupBy("label").count().show()

    val document = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("document"))
      .setOutputCol("token")

    val norm = new Normalizer()
      .setInputCols(Array("token"))
      .setOutputCol("normalized")
      .setLowercase(true)

    val stops = StopWordsCleaner
      .pretrained()
      .setInputCols("normalized")
      .setOutputCol("cleanedToken")

    val doc2Vec = new Doc2VecApproach()
      .setInputCols("cleanedToken")
      .setOutputCol("sentence_embeddings")
      .setMaxSentenceLength(512)
      .setStepSize(0.025)
      .setMinCount(5)
      .setVectorSize(512)
      .setNumPartitions(1)
      .setMaxIter(5)
      .setSeed(42)
      .setStorageRef("doc2vec_aclImdb")

    val docClassifier = new ClassifierDLApproach()
      .setInputCols("sentence_embeddings")
      .setOutputCol("category")
      .setLabelColumn("label")
      .setBatchSize(8)
      .setMaxEpochs(20)
      .setLr(1e-3f)
      .setDropout(0.5f)
    //      .setValidationSplit(0.2f)

    val pipeline =
      new Pipeline().setStages(Array(document, tokenizer, norm, stops, doc2Vec, docClassifier))

    val pipelineModel = pipeline.fit(train)
    val pipelineDF = pipelineModel.transform(test)

    pipelineDF.select("text", "category.result").show()

    val tmpDF = pipelineDF.select($"label", explode($"category.result").as("cat"))
    println("count of tmpDF: ", tmpDF.count)
    tmpDF.show(2)

    val newDF = tmpDF
      .select("label", "cat")
      .withColumn("original", when($"label" === 0, 0d).otherwise(1d))
      .withColumn("prediction", when($"cat" === 0, 0d).otherwise(1d))

    println("count of newDF: ", newDF.count)
    newDF.show()

    val rdd = newDF
      .select("prediction", "original")
      .rdd
      .map(row => (row.getAs[Double]("prediction"), row.getAs[Double]("original")))

    rdd.take(2)

    val metrics = new MulticlassMetrics(rdd)
    val accuracy = metrics.accuracy
    println("Summary Statistics")
    println(s"Accuracy = $accuracy")

    // Weighted stats
    println(s"Weighted precision: ${metrics.weightedPrecision}")
    println(s"Weighted recall: ${metrics.weightedRecall}")
    println(s"Weighted F1 score: ${metrics.weightedFMeasure}")
    println(s"Weighted false positive rate: ${metrics.weightedFalsePositiveRate}")

    val labels = metrics.labels
    // Precision by label
    labels.foreach { l =>
      println(s"Precision $l = " + metrics.precision(l))
    }

    // Recall by label
    labels.foreach { l =>
      println(s"Recall $l = " + metrics.recall(l))
    }

    // F-measure by label
    labels.foreach { l =>
      println(s"F1-Score $l = " + metrics.fMeasure(l))
    }

    val binaryMetrics = new BinaryClassificationMetrics(rdd)

    // AUPRC
    val auPRC = binaryMetrics.areaUnderPR
    println("Area under precision-recall curve = " + auPRC)

    // ROC Curve
    val roc = binaryMetrics.roc
    println("Area ROC Curve = " + roc)

    // AUROC
    val auROC = binaryMetrics.areaUnderROC
    println("Area under ROC = " + auROC)

  }

  it should "get word vectors as spark dataframe" taggedAs SlowTest in {

    import ResourceHelper.spark.implicits._

    val testDataset = Seq(
      "Rare Hendrix song draft sells for almost $17,000. This is my second sentenece! The third one here!")
      .toDF("text")

    val doc2Vec = Doc2VecModel
      .pretrained()
      .setInputCols("token")
      .setOutputCol("embeddings")

    val pipeline =
      new Pipeline().setStages(Array(documentAssembler, tokenizer, doc2Vec))

    val result = pipeline.fit(testDataset).transform(testDataset)
    result.show()

    doc2Vec.getVectors.show()
  }

  it should "raise an error when trying to retrieve empty word vectors" taggedAs SlowTest in {
    val word2Vec = Doc2VecModel
      .pretrained()
      .setInputCols("token")
      .setOutputCol("embeddings")

    intercept[UnsupportedOperationException] {
      word2Vec.getVectors
    }
  }

}
