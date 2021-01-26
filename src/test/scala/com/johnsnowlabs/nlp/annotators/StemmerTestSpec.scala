package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.{AnnotatorType, ContentProvider, DataBuilder}
import com.johnsnowlabs.tags.FastTest
import org.apache.spark.sql.{Dataset, Row}
import org.scalatest._


class StemmerTestSpec extends FlatSpec with StemmerBehaviors {

  val stemmer = new Stemmer
  "a Stemmer" should s"be of type ${AnnotatorType.TOKEN}" taggedAs FastTest in {
    assert(stemmer.outputAnnotatorType == AnnotatorType.TOKEN)
  }

  val englishPhraseData: Dataset[Row] = DataBuilder.basicDataBuild(ContentProvider.englishPhrase)

  "A full Stemmer pipeline with English content" should behave like fullStemmerPipeline(englishPhraseData)


}
