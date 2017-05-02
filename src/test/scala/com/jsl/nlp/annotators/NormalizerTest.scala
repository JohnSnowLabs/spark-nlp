package com.jsl.nlp.annotators

import com.jsl.nlp.{Annotation, Document, SparkTest, TestRow}
import org.apache.spark.sql.Row
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner

@RunWith(classOf[JUnitRunner])
class NormalizerTest extends SparkTest {
  test("simple normalization test") {
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
    val normalizer = new Normalizer()
      .setDocumentCol("document")
      .setInputAnnotationCols(Array("stem"))
    val processed = normalizer.transform(stemmer.transform(tokenizer.transform(dataset)))
    println(processed.schema)
    processed.collect().foreach {
      row =>
        val document = Document(row.getAs[Row](0))
        println(document)
        row.getSeq[Row](3)
          .map(Annotation(_))
          .foreach {
            case stem: Annotation if stem.aType == Normalizer.aType =>
              println(stem, document.text.substring(stem.begin, stem.end), stem.metadata.mkString(", "))
            case _ => ()
          }
    }
  }
}
