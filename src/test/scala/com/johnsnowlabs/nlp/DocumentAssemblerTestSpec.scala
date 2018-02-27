package com.johnsnowlabs.nlp

import org.scalatest._
import org.apache.spark.sql.{DataFrame, Row}

import scala.language.reflectiveCalls
import Matchers._

class DocumentAssemblerTestSpec extends FlatSpec {
  def fixture = new {
    val text = ContentProvider.englishPhrase
    val df = AnnotatorBuilder.withDocumentAssembler(DataBuilder.basicDataBuild(text))
    val assembledDoc = df
      .select("document")
      .collect
      .flatMap { _.getSeq[Row](0) }
      .map { Annotation(_) }
  }

  "A DocumentAssembler" should "annotate with the correct indexes" in {
    val f = fixture
    f.text.head should equal (f.text(f.assembledDoc.head.start))
    f.text.last should equal (f.text(f.assembledDoc.head.end))
  }

  it should "index lower bound should be 0" in {
    val df: DataFrame = DataBuilder.multipleDataBuild(Seq[String]("", ".", ";", s"\n", "Someday", "1"))
    val df2 = new DocumentAssembler()
      .setInputCol("text")
      .transform(df)

    df2
      .select("document")
      .collect
      .flatMap { _.getSeq[Row](0) }
      .foreach { a =>
        a.getInt(2) should be >= 0
      }
  }
}