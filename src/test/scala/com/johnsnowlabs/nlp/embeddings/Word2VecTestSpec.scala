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
import org.apache.spark.sql.functions.{col, explode, size}
import org.scalatest.flatspec.AnyFlatSpec

class Word2VecTestSpec extends AnyFlatSpec with SparkSessionTest {

  import spark.implicits._

  "Word2VecApproach" should "train, save, and load back the saved model" taggedAs FastTest in {

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

    val Word2Vec = new Word2VecApproach()
      .setInputCols("cleanedToken")
      .setOutputCol("embeddings")
      .setMaxSentenceLength(512)
      .setStorageRef("my_awesome_word2vec")
      .setEnableCaching(true)

    val pipeline = new Pipeline().setStages(
      Array(documentAssembler, sentenceDetector, tokenizer, stops, Word2Vec))

    val pipelineModel = pipeline.fit(ddd)
    val pipelineDF = pipelineModel.transform(ddd)

    pipelineDF.select("embeddings").show()

    pipelineModel.write.overwrite().save("./tmp_pipeline_word2vec")
    pipelineModel.stages.last
      .asInstanceOf[Word2VecModel]
      .write
      .overwrite()
      .save("./tmp_word2vec_model")

    val loadedWord2Vec = Word2VecModel
      .load("./tmp_word2vec_model")
      .setInputCols("token")
      .setOutputCol("embeddings")

    val loadedPipeline =
      new Pipeline().setStages(
        Array(documentAssembler, sentenceDetector, tokenizer, loadedWord2Vec))

    loadedPipeline.fit(ddd).transform(ddd).select("embeddings").show()

  }

  "Word2VecModel" should "Benchmark" taggedAs SlowTest in {

    val conll = CoNLL(explodeSentences = false)
    val training_data = conll
      .readDataset(ResourceHelper.spark, "src/test/resources/conll2003/eng.train")
      .repartition(12)
    val test_data = conll
      .readDataset(ResourceHelper.spark, "src/test/resources/conll2003/eng.testa")
      .repartition(12)

    println(training_data.count())

    val Word2Vec = new Word2VecApproach()
      .setInputCols("token")
      .setOutputCol("embeddings")
      .setMaxSentenceLength(1024)
      .setStepSize(0.001)
      .setMinCount(10)
      .setVectorSize(100)
      .setNumPartitions(1)
      .setMaxIter(4)
      .setSeed(42)
      .setStorageRef("Word2Vec_conll_03")

    val pipeline = new Pipeline().setStages(Array(Word2Vec))

    val pipelineModel = pipeline.fit(training_data)
    val pipelineDF = pipelineModel.transform(test_data)

    Benchmark.time("Time to save Word2Vec results") {
      pipelineModel
        .transform(training_data)
        .write
        .mode("overwrite")
        .parquet("./tmp_Word2Vec_pipeline")
    }

    Benchmark.time("Time to save Word2Vec results") {
      pipelineModel
        .transform(test_data)
        .write
        .mode("overwrite")
        .parquet("./tmp_Word2Vec_pipeline")
    }

    println("missing tokens/embeddings: ")
    pipelineDF
      .withColumn("sentence_size", size(col("sentence")))
      .withColumn("token_size", size(col("token")))
      .withColumn("embed_size", size(col("embeddings")))
      .where(col("token_size") =!= col("embed_size"))
      .select("sentence_size", "token_size", "embed_size", "token.result", "embeddings.result")
      .show(false)

    println("total sentences: ", pipelineDF.select(explode($"sentence.result")).count)
    val totalTokens = pipelineDF.select(explode($"token.result")).count.toInt
    val totalEmbeddings = pipelineDF.select(explode($"embeddings.embeddings")).count.toInt

    println(s"total tokens: $totalTokens")
    println(s"total embeddings: $totalEmbeddings")

    assert(totalTokens == totalEmbeddings)

  }

  "Word2VecModel" should "train classifierdl" taggedAs SlowTest in {

    val conll = CoNLL(explodeSentences = true)
    val trainingData =
      conll.readDataset(ResourceHelper.spark, "src/test/resources/conll2003/eng.train")

    println("count of training dataset: ", trainingData.count)

    val sentence = SentenceDetectorDLModel
      .pretrained()
      .setInputCols(Array("document"))
      .setOutputCol("sentence")

    val word2vec = new Word2VecApproach()
      .setInputCols("token")
      .setOutputCol("embeddings")
      .setMaxSentenceLength(512)
      .setStepSize(0.025)
      .setMinCount(5)
      .setVectorSize(512)
      .setNumPartitions(1)
      .setMaxIter(5)
      .setSeed(42)
      .setStorageRef("Word2Vec_conll03")

    val nerClassifier = new NerDLApproach()
      .setInputCols("sentence", "token", "embeddings")
      .setOutputCol("ner")
      .setLabelColumn("label")
      .setOutputCol("ner")
      .setLr(1e-3f) // 0.001
      .setPo(5e-3f) // 0.005
      .setDropout(5e-1f) // 0.5
      .setMaxEpochs(5)
      .setRandomSeed(0)
      .setVerbose(0)
      .setBatchSize(32)
      .setValidationSplit(0.1f)
      .setEvaluationLogExtended(true)

    val pipeline =
      new Pipeline().setStages(
        Array(documentAssembler, sentence, tokenizer, word2vec, nerClassifier))

    val pipelineModel = pipeline.fit(trainingData)
    val pipelineDF = pipelineModel.transform(trainingData)

    pipelineDF.select("text", "ner.result").show()

  }

  it should "get word vectors as spark dataframe" taggedAs SlowTest in {

    import ResourceHelper.spark.implicits._

    val testDataset = Seq(
      "Rare Hendrix song draft sells for almost $17,000. This is my second sentenece! The third one here!")
      .toDF("text")

    val word2Vec = Word2VecModel
      .pretrained()
      .setInputCols("token")
      .setOutputCol("embeddings")

    val pipeline =
      new Pipeline().setStages(Array(documentAssembler, tokenizer, word2Vec))

    val result = pipeline.fit(testDataset).transform(testDataset)
    result.show()

    word2Vec.getVectors.show()
  }

  it should "raise an error when trying to retrieve empty word vectors" taggedAs SlowTest in {
    val word2Vec = Word2VecModel
      .pretrained()
      .setInputCols("token")
      .setOutputCol("embeddings")

    intercept[UnsupportedOperationException] {
      word2Vec.getVectors
    }
  }

}
