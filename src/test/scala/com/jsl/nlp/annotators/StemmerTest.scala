package com.jsl.nlp.annotators

import com.jsl.nlp.{Annotation, Document, SparkTest, TestRow}
import org.apache.spark.sql.Row
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner

@RunWith(classOf[JUnitRunner])
class StemmerTest extends SparkTest {
  test("simple stemming test") {
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
    val stemmer = new Stemmer()
      .setDocumentCol("document")
      .setInputAnnotationCols(Array("token"))
    val processed = stemmer.transform(tokenizer.transform(dataset))
    println(processed.schema)
    processed.collect().foreach {
      row =>
        val document = Document(row.getAs[Row](0))
        println(document)
        row.getSeq[Row](2)
          .map(Annotation(_))
          .foreach {
            case stem: Annotation if stem.aType == Stemmer.aType =>
              println(stem, document.text.substring(stem.begin, stem.end))
            case _ => ()
        }
    }
  }
}
