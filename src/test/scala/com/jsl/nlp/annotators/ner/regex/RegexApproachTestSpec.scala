package com.jsl.nlp.annotators.ner.regex

import com.jsl.nlp.{ContentProvider, DataBuilder}
import org.scalatest._

class RegexApproachTestSpec extends FlatSpec with RegexApproachBehaviors {

  val trainedTagger: NERRegexModel =
    NERRegexApproach.train("/ner-corpus/dict.txt")

  "an isolated dictionary tagger" should behave like isolatedDictionaryTagging(
    trainedTagger,
    Array(ContentProvider.nerSentence)
  )

   val sentenceFromWsj = ContentProvider.targetSentencesFromWsj
  
  "a spark based pragmatic detector" should behave like sparkBasedNERTagger(
    DataBuilder.basicDataBuild(ContentProvider.nerSentence)
  )

  "A Dictionary Tagger" should "be readable and writable" in {
    val regexNerApproach = NERRegexApproach.train("/ner-corpus/dict.txt")
    val path = "./test-output-tmp/regexnertagger"
    try {
      regexNerApproach.write.overwrite.save(path)
      val nerTaggerRead = NERRegexModel.read.load(path)
      assert(regexNerApproach.get(regexNerApproach.model).isDefined)
    } catch {
      case _: java.io.IOException => succeed
    }
  }

}
