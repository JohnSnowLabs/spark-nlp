package com.johnsnowlabs.nlp.annotators.coref
import com.johnsnowlabs.nlp.SparkAccessor.spark
import com.johnsnowlabs.nlp.annotator.SentenceDetectorDLModel
import com.johnsnowlabs.nlp.annotators.Tokenizer
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.embeddings.BertEmbeddings
import com.johnsnowlabs.nlp.{Annotation, DocumentAssembler, LightPipeline}
import com.johnsnowlabs.tags.{FastTest, SlowTest}
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.{Row, SparkSession}
import org.scalatest.flatspec.AnyFlatSpec

import java.nio.file.{Files, Paths}
import scala.io.Source

class SpanBertCorefModelSpec extends AnyFlatSpec {
  implicit val session: SparkSession = spark

  import spark.implicits._

  "SpanBertCoref" should "load TF graph" taggedAs SlowTest in {
    SpanBertCorefModel
      .loadSavedModel("/tmp/coref_tf1", spark)
      .setMaxSegmentLength(384)
      .setCaseSensitive(true)
      .write
      .overwrite
      .save("/tmp/spanbertcoref")
  }

  "SpanBertCoref" should "process some text" taggedAs FastTest in {
    val ddd = Seq(
      "Meanwhile Prime Minister Ehud Barak told Israeli television he doubts a peace deal can be reached before Israel's February 6th election. He said he will now focus on suppressing Palestinian violence.",
      "John loves Mary because she knows how to treat him. She is also fond of him.",
      "John said something to Mary but she didn't respond to him.",
      " ",
      "",
      "hey",
      "hey hey")
      .toDF("text")
      .repartition(numPartitions = 1)

    val document = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentencer = SentenceDetectorDLModel
      .pretrained()
      .setInputCols(Array("document"))
      .setOutputCol("sentences")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("sentences"))
      .setOutputCol("tokens")

    val corefs = SpanBertCorefModel
      .load("/tmp/spanbertcoref")
      .setInputCols(Array("sentences", "tokens"))
      .setOutputCol("corefs")

    val pipeline = new Pipeline().setStages(Array(document, sentencer, tokenizer, corefs))

    pipeline.fit(ddd).transform(ddd).select("corefs").show(truncate = false)
  }
}
