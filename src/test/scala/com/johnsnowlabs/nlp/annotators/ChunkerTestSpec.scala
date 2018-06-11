package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.SparkAccessor
import org.scalatest.FlatSpec

class ChunkerTestSpec extends FlatSpec with ChunkerBehaviors{


  "a chunker using a trained POS" should behave like testChunkingWithTrainedPOS(
  Array(
  "first sentence example",
  "second something going"
  )
  )

  "a chunker using a trained POS" should behave like testChunkingWithTrainedPOSUsingPipeline(
  Array(
  "first sentence example",
  "second something going"
  )
  )

  "a chunker with user input POS tags" should behave like testUserInputPOSTags(
    Array("the little yellow dog barked at the cat"),
    Array(Array("DT", "JJ", "JJ", "NN", "VBD", "IN", "DT", "NN")),
    Array("<DT>?<JJ>*<NN>") //(<DT>)?(<JJ>)*(<NN>)
  )


  "a chunker with a regex parser that does not match" should behave like testRegexDoesNotMatch(
    Array("this should not return a chunk phrase because there is no match"),
    Array(Array("DT", "MD", "RB","VB", "DT", "NN", "NN", "IN", "EX", "VBZ", "DT", "NN")),
    Array("<NNP>+")
  )


  val testingPhrases = List(
    SentenceParams(sentence = "the little yellow dog barked at the cat",
                   POSFormatSentence = Array("DT", "JJ", "JJ", "NN", "VBD", "IN", "DT", "NN"),
                   regexParser = Array("<DT>?<JJ>*<NN>"), //(<DT>)?(<JJ>)*(<NN>)
                   correctChunkPhrases = Array("the little yellow dog", "the cat")),

    SentenceParams(sentence = "Rapunzel let down her long golden hair",
                   POSFormatSentence = Array("NNP", "VBD", "RP", "PP$", "JJ", "JJ", "NN"),
                   regexParser = Array("<DT|PP\\$>?<JJ>*<NN>"), //(<DT>)|(<PP\$>)?(<JJ>)*(<NN>)
                   correctChunkPhrases = Array("her long golden hair")),

    SentenceParams(sentence = "Rapunzel let down her long golden hair",
                   POSFormatSentence = Array("NNP", "VBD", "RP", "PP$", "JJ", "JJ", "NN"),
                   regexParser = Array("<NNP>+"), //(<NNP>)+
                   correctChunkPhrases = Array("Rapunzel")),

    SentenceParams(sentence = "money market fund",
                   POSFormatSentence = Array("NN", "NN", "NN"),
                   regexParser = Array("<NN><NN>"), //(<NN>)(<NN>)
                   correctChunkPhrases = Array("money market")),

    SentenceParams(sentence = "In the emergency room his ultrasound showed mild hydronephrosis with new perinephric fluid surrounding the transplant kidney",
      POSFormatSentence = Array("IN","DT","NN","NN","PRP$","NN","VBD","JJ","NN","IN","JJ","JJ","NN","VBG","DT","NN","NN"),
      regexParser = Array("<NN><NN>"),
      correctChunkPhrases = Array("emergency room", "transplant kidney")),

   SentenceParams(sentence = "this should not return a chunk phrase because there is no match",
                  POSFormatSentence = Array("DT", "MD", "RB","VB", "DT", "NN", "NN", "IN", "EX", "VBZ", "DT", "NN"),
                  regexParser = Array("<NN><NNP>"),
                  correctChunkPhrases = Array())
  )

  "a chunker with several sentences and theirs regex parameter" should behave like testChunksGeneration(testingPhrases)


  import SparkAccessor.spark.implicits._

  val document = Seq(
    ("Rapunzel let down her long golden hair", Array("NNP", "VBD", "RP", "PP$", "JJ", "JJ", "NN")),
    ("money market fund", Array("NN", "NN", "NN"))
  ).toDS.toDF("text", "tags")

  "a chunker with a multiple regex parsers" should behave like testMultipleRegEx(document, Array("<NNP>+", "<DT>?<JJ>*<NN>", "<DT|PP\\$>?<JJ>*<NN>"))

}
