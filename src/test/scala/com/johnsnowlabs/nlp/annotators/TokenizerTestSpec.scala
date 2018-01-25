package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp._
import org.apache.spark.sql.{Dataset, Row}
import org.scalatest._
import java.util.Date

/**
  * Created by saif on 02/05/17.
  */
class TokenizerTestSpec extends FlatSpec with RegexTokenizerBehaviors {

  val regexTokenizer = new Tokenizer

  "a RegexTokenizer" should s"be of type ${AnnotatorType.TOKEN}" in {
    assert(regexTokenizer.annotatorType == AnnotatorType.TOKEN)
  }

  "a spark based tokenizer" should "resolve big data" in {
    val data = ContentProvider.parquetData.limit(500000)
      .repartition(16)

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")

    val assembled = documentAssembler.transform(data)

    val tokenizer = new Tokenizer()
      .setOutputCol("token")
    val tokenized = tokenizer.transform(assembled)

    val date1 = new Date().getTime
    Annotation.take(tokenized, "token", 5000)
    info(s"Collected 5000 tokens took ${(new Date().getTime - date1) / 1000} seconds")
  }

  val latinBodyData: Dataset[Row] = DataBuilder.basicDataBuild(ContentProvider.latinBody)

  "A full RegexTokenizer pipeline with latin content" should behave like fullTokenizerPipeline(latinBodyData)

}
