package com.jsl.nlp.annotators

import com.jsl.nlp._
import org.apache.spark.sql.Row
import org.scalatest._

/**
  * Created by Saif Addin on 6/16/2017.
  */
class CombinedTestSpec extends FlatSpec {

  "Simple combined annotators" should "successfully go through all transformations" in {

    val data = DataBuilder.basicDataBuild("This is my first sentence. This is your second list of words")
    val transformation = AnnotatorBuilder.withLemmaTaggedSentences(data)
    transformation
      .collect().foreach {
      row =>
        val document = Document(row.getAs[Row](0))
        println(document)
        row.getSeq[Row](1).map(Annotation(_)).foreach {
          token =>
            println(token)
        }
        row.getSeq[Row](2).map(Annotation(_)).foreach {
          token =>
            println(token)
        }
        row.getSeq[Row](4).map(Annotation(_)).foreach {
          token =>
            println(token)
        }
    }

  }

}
