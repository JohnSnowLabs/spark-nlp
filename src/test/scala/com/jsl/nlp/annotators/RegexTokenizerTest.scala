package com.jsl.nlp.annotators

import com.jsl.nlp.{Annotation, Document, SparkTest, TestRow}
import org.apache.spark.sql.Row
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner

@RunWith(classOf[JUnitRunner])
class RegexTokenizerTest extends SparkTest {
  test("simple tokenize test") {
    val docs = Seq(
      TestRow(Document(
        "id",
        testContent
      ))
    )
    val dataset = spark.createDataFrame(docs)
    println(dataset.schema)
    val tokenizer = new RegexTokenizer()
      .setDocumentCol("document")
      .setPattern("[a-zA-Z]+|[0-9]+|\\p{Punct}")
    tokenizer.transform(dataset).collect().foreach {
      row =>
        val document = Document(row.getAs[Row](0))
        println(document)
        row.getSeq[Row](1).map(Annotation(_)).foreach {
          token =>
            println(token, document.text.substring(token.begin, token.end))
        }
    }
  }
}
