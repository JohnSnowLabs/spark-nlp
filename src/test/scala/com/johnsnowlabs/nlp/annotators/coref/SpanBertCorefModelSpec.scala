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

  "SpanBertCoref" should "load TF graph" taggedAs SlowTest ignore {
    SpanBertCorefModel
      .loadSavedModel("/tmp/coref_tf1", spark)
      .setMaxSegmentLength(384)
      .setCaseSensitive(true)
      .write
      .overwrite
      .save("/tmp/spanbertcoref")
  }

  "SpanBertCoref" should "process some text" taggedAs FastTest in {
    val data = Seq(
      "John told Mary he would like to borrow a book from her.")
      .toDF("text")

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
      .pretrained()
      .setMaxSegmentLength(384)
      .setInputCols(Array("sentences", "tokens"))
      .setOutputCol("corefs")

    val pipeline = new Pipeline().setStages(Array(document, sentencer, tokenizer, corefs))

    val result = pipeline.fit(data).transform(data)

    result.selectExpr("explode(corefs) as coref")
      .selectExpr("coref.result as token", "coref.metadata").show(8, truncate = false)
  }
}
