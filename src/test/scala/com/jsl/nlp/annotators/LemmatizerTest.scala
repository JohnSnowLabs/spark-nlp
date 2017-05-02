package com.jsl.nlp.annotators

import com.jsl.nlp.{Annotation, Document, SparkTest, TestRow}
import org.apache.spark.sql.Row

/**
  * Created by saif on 01/05/17.
  */
class LemmatizerTest extends SparkTest {
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
    val lemmatizer = new Lemmatizer()
    val processed = lemmatizer.transform(normalizer.transform(stemmer.transform(tokenizer.transform(dataset))))
    println(processed.schema)
    processed.collect().foreach {
      row =>
        val document = Document(row.getAs[Row](0))
        println(document)
        row.getSeq[Row](4)
          .map(Annotation(_))
          .foreach {
            case lemma: Annotation if lemma.aType == Lemmatizer.aType =>
              println(lemma, document.text.substring(lemma.begin, lemma.end), lemma.metadata.mkString(", "))
            case _ => ()
          }
    }
  }
}
