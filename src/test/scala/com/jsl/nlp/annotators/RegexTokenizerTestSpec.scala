package com.jsl.nlp.annotators

import com.jsl.nlp._
import org.apache.spark.sql.{Dataset, Row}
import org.scalatest._
import java.util.Date

/**
  * Created by saif on 02/05/17.
  */
class RegexTokenizerTestSpec extends FlatSpec with RegexTokenizerBehaviors {

  val regexTokenizer = new RegexTokenizer
  "a RegexTokenizer" should s"be of type ${RegexTokenizer.annotatorType}" in {
    assert(regexTokenizer.annotatorType == RegexTokenizer.annotatorType)
  }

  "a spark based tokenizer" should "resolve big data" in {
    import SparkAccessor.spark.implicits._
    val data = ContentProvider.parquetData.limit(50000)
      .withColumn("document", Document.construct($"text"))
        .repartition(16)

    val tokenizer = new RegexTokenizer().setDocumentCol("document")
    val tokenized = tokenizer.transform(data)

    val date1 = new Date().getTime
    Annotation.take(tokenized, RegexTokenizer.annotatorType, 5000)
    info(s"Collected 5000 tokens took ${(new Date().getTime - date1) / 1000} seconds")
  }

  val latinBodyData: Dataset[Row] = DataBuilder.basicDataBuild(ContentProvider.latinBody)

  "A full RegexTokenizer pipeline with latin content" should behave like fullTokenizerPipeline(latinBodyData)

}
