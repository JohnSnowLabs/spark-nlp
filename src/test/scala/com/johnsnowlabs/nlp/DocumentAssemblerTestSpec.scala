package com.johnsnowlabs.nlp

import org.scalatest._
import org.apache.spark.sql.Row
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
    f.text.head should equal (f.text(f.assembledDoc.head.begin))
    f.text.last should equal (f.text(f.assembledDoc.head.end))
  }
}