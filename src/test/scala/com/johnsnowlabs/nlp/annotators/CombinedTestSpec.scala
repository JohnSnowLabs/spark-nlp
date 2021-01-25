package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.AnnotatorType._
import com.johnsnowlabs.nlp._
import com.johnsnowlabs.tags.FastTest
import org.apache.spark.sql.Row
import org.scalatest._

class CombinedTestSpec extends FlatSpec {

  "Simple combined annotators" should "successfully go through all transformations" taggedAs FastTest in {
    val data = DataBuilder.basicDataBuild("This is my first sentence. This is your second list of words")
    val transformation = AnnotatorBuilder.withLemmaTaggedSentences(data)
    transformation
      .collect().foreach {
      row =>
        row.getSeq[Row](1).map(Annotation(_)).foreach { token =>
          // Document annotation
          assert(token.annotatorType == DOCUMENT)
        }
        row.getSeq[Row](2).map(Annotation(_)).foreach { token =>
          // SBD annotation
          assert(token.annotatorType == DOCUMENT)
        }
        row.getSeq[Row](4).map(Annotation(_)).foreach { token =>
          // POS annotation
          assert(token.annotatorType == POS)
        }
    }
  }
}
