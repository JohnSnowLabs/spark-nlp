package com.jsl.nlp.annotators.clinical.negex

import com.jsl.nlp.{ContentProvider, DataBuilder}
import org.apache.spark.sql.{Dataset, Row}
import org.scalatest._

/**
  * Created by Saif Addin on 5/10/2017.
  */
class NegexTaggerTestSpec extends FlatSpec with NegexTaggerBehaviors {

  val negBodyData: Dataset[Row] = DataBuilder.multipleDataBuild(ContentProvider.negatedSentences)

  "A full NegexTagger pipeline with negated content" should behave like negexTagger(negBodyData)
}
