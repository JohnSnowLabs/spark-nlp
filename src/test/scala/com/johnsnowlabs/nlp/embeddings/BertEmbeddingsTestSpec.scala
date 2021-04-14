package com.johnsnowlabs.nlp.embeddings

import com.johnsnowlabs.ml.tensorflow.{ModelSignature, TFSignatureFactory, TensorflowWrapper}
import com.johnsnowlabs.nlp.annotators.{StopWordsCleaner, Tokenizer}
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.training.CoNLL
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.tags.{FastTest, SlowTest}
import com.johnsnowlabs.util.Benchmark
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.functions.{col, explode, size}
import org.scalatest._

class BertEmbeddingsTestSpec extends FlatSpec {

  "Bert Embeddings" should "correctly embed tokens and sentences" taggedAs SlowTest in {

    import ResourceHelper.spark.implicits._

    val ddd = Seq(
      "Something is weird on the notebooks, something is happening."
    ).toDF("text")

    val data1 = Seq(
      "In the Seven Kingdoms of Westeros, a soldier of the ancient Night's Watch order survives an attack by supernatural creatures known as the White Walkers, thought until now to be mythical."
    ).toDF("text")

    val data2 = Seq(
      "In King's Landing, the capital, Jon Arryn, the King's Hand, dies under mysterious circumstances."
    ).toDF("text")

    val data3 = Seq(
      "Tyrion makes saddle modifications for Bran that will allow the paraplegic boy to ride."
    ).toDF("text")

    val document = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("document"))
      .setOutputCol("token")

    val embeddings = BertEmbeddings.pretrained("small_bert_L2_128", "en")
      .setInputCols(Array("token", "document"))
      .setOutputCol("bert")

    val pipeline = new Pipeline().setStages(Array(document, tokenizer, embeddings))

    val bertDDD = pipeline.fit(ddd).transform(ddd)
    val bertDF1 = pipeline.fit(data1).transform(data1)
    val bertDF2 = pipeline.fit(data2).transform(data2)
    val bertDF3 = pipeline.fit(data3).transform(data3)

  }

  "Bert Embeddings" should "correctly work with empty tokens" taggedAs SlowTest in {

    val smallCorpus = ResourceHelper.spark.read.option("header","true").csv("src/test/resources/embeddings/sentence_embeddings.csv")

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("document"))
      .setOutputCol("token")

    val stopWordsCleaner = new StopWordsCleaner()
      .setInputCols("token")
      .setOutputCol("cleanTokens")
      .setStopWords(Array("this", "is", "my", "document", "sentence", "second", "first", ",", "."))
      .setCaseSensitive(false)

    val embeddings = BertEmbeddings.pretrained("small_bert_L2_128", "en")
      .setInputCols("document", "cleanTokens")
      .setOutputCol("embeddings")
      .setCaseSensitive(true)

    val pipeline = new Pipeline()
      .setStages(Array(
        documentAssembler,
        tokenizer,
        stopWordsCleaner,
        embeddings
      ))

    val pipelineDF = pipeline.fit(smallCorpus).transform(smallCorpus)
    Benchmark.time("Time to save BertEmbeddings results") {
      pipelineDF.write.mode("overwrite").parquet("./tmp_bert_embeddings")
    }
  }

  "Bert Embeddings" should "benchmark test" taggedAs SlowTest in {
    import ResourceHelper.spark.implicits._

    val conll = CoNLL()
    val training_data = conll.readDataset(ResourceHelper.spark, "src/test/resources/conll2003/eng.train")

    val embeddings = BertEmbeddings.pretrained("small_bert_L2_128", "en")
      .setInputCols("sentence", "token")
      .setOutputCol("embeddings")
      .setCaseSensitive(false)
      .setMaxSentenceLength(512)
      .setBatchSize(16)

    val pipeline = new Pipeline()
      .setStages(Array(
        embeddings
      ))

    val pipelineDF = pipeline.fit(training_data).transform(training_data)
    Benchmark.time("Time to save BertEmbeddings results") {
      pipelineDF.write.mode("overwrite").parquet("./tmp_bert_embeddings")
    }

    println("missing tokens/embeddings: ")
    pipelineDF.withColumn("sentence_size", size(col("sentence")))
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

  "Bert Embeddings" should "correctly match TF model signatures" taggedAs FastTest in {
    val tfModelPath = "src/test/resources/custom-bert/model" // TF2 MODEL HUB

    val matchedSignatures: Option[Map[String, String]] =
      TensorflowWrapper
        .extractSignatures(
          tags = Array("serve"),
          savedModelDir = tfModelPath,
          modelProvider = "TF2")

    val signatures = matchedSignatures match {
      case Some(s) => s
      case _ => throw new Exception("Signature map is empty!")
    }

    assert(signatures.contains("input_ids"))
    assert(signatures.contains("input_mask"))
    assert(signatures.contains("segment_ids"))
    assert(signatures.contains("sequence_output"))
  }

  "Bert Embeddings" should "correctly load custom model with extracted signatures" taggedAs SlowTest in {

    import ResourceHelper.spark.implicits._

    val ddd = Seq(
      "Something is weird on the notebooks, something is happening."
    ).toDF("text")

    val document = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("document"))
      .setOutputCol("token")

    val tfModelPath = "src/test/resources/custom-bert/model" // TF2 hub model

    val signatures = TensorflowWrapper.extractSignatures(modelProvider = "TF2", savedModelDir = tfModelPath)

    val embeddings = BertEmbeddings.loadSavedModel(tfModelPath, ResourceHelper.spark, signatures)
      .setInputCols(Array("token", "document"))
      .setOutputCol("bert")

    val pipeline = new Pipeline().setStages(Array(document, tokenizer, embeddings))

    pipeline.fit(ddd).transform(ddd).show
  }
}
