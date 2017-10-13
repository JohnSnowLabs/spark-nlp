package com.johnsnowlabs.nlp.annotators.ner

import com.johnsnowlabs.nlp.{ContentProvider, DataBuilder}
import org.apache.spark.sql.{Dataset, Row}
import org.scalatest._

/**
 * Created by saif on 02/05/17.
 */
class NERTaggerTestSpec extends FlatSpec with NERTaggerBehaviors {

  val nerSentence: Dataset[Row] = DataBuilder.basicDataBuild(ContentProvider.nerSentence)

  "A full NERTagger pipeline with English content" should behave like fullNERTaggerPipeline(nerSentence)

}
