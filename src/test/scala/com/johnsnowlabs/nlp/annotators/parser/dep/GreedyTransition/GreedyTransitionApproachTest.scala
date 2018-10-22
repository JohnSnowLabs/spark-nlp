package com.johnsnowlabs.nlp.annotators.parser.dep.GreedyTransition

import com.johnsnowlabs.nlp.annotators.common.PosTagged
import com.johnsnowlabs.nlp.util.io.{ExternalResource, ReadAs}
import com.johnsnowlabs.nlp.{Annotation, AnnotatorBuilder, ContentProvider, DataBuilder}
import org.scalatest.FlatSpec

import scala.language.reflectiveCalls
import org.scalatest.Matchers._

class GreedyTransitionApproachTest extends FlatSpec {
  def fixture = new {
    val model = new GreedyTransitionApproach
    val df = AnnotatorBuilder.withFullPOSTagger(DataBuilder.basicDataBuild(ContentProvider.depSentence))

    val tokenAnnotations = Annotation.collect(df, "token")
      .flatten
      .sortBy { _.begin }

    val posTagAnnotations = Annotation.collect(df, "pos")
      .flatten
      .sortBy { _.begin }

    val sentenceAnnotation = Annotation.collect(df, "sentence")
      .flatten
      .sortBy { _.begin }

  }

//  "A GreedyTransitionApproach" should "return an array of dependencies" in {
//    val f = fixture
//    val sentences = PosTagged.unpack(f.sentenceAnnotation ++ f.tokenAnnotations ++ f.posTagAnnotations)
//    for (sentence <- sentences) {
//      assert(f.model.parse(sentence, ExternalResource("src/test/resources/models/dep-model.txt", ReadAs.LINE_BY_LINE, Map.empty[String, String])).tokens.length == sentence.indexedTaggedWords.length)
//    }
//  }
}
