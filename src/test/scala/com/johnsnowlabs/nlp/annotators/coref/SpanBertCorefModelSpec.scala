package com.johnsnowlabs.nlp.annotators.coref
import com.johnsnowlabs.nlp.SparkAccessor.spark
import com.johnsnowlabs.nlp.annotator.SentenceDetectorDLModel
import com.johnsnowlabs.nlp.annotators.Tokenizer
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.tags.{FastTest, SlowTest}
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.SparkSession
import org.scalatest.flatspec.AnyFlatSpec

class SpanBertCorefModelSpec extends AnyFlatSpec {
  implicit val session: SparkSession = spark

  import spark.implicits._

  "SpanBertCoref" should "should be serialized" taggedAs SlowTest in {
    SpanBertCorefModel
      .pretrained()
      .setMaxSegmentLength(384)
      .setCaseSensitive(true)
      .write
      .overwrite
      .save("./tmp_spanbertcoref")

    SpanBertCorefModel.load("./tmp_spanbertcoref")
  }

  "SpanBertCoref" should "process some text" taggedAs SlowTest in {
    val data = Seq("John told Mary he would like to borrow a book from her.")
      .toDF("text")

    val document = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentence = SentenceDetectorDLModel
      .pretrained()
      .setInputCols(Array("document"))
      .setOutputCol("sentences")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("sentences"))
      .setOutputCol("tokens")

    val corefs = SpanBertCorefModel
      .pretrained()
      .setInputCols(Array("sentences", "tokens"))
      .setOutputCol("corefs")

    val pipeline = new Pipeline().setStages(Array(document, sentence, tokenizer, corefs))

    val result = pipeline.fit(data).transform(data)

    result
      .selectExpr("explode(corefs) as coref")
      .selectExpr("coref.result as token", "coref.metadata")
      .show(8, truncate = false)
  }
}
