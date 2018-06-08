package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.SparkAccessor
import org.scalatest.FlatSpec

class ChunkerTestSpec extends FlatSpec with ChunkerBehaviors{

  "a trained POS tag" should behave like testPOSForChunking(
  Array(
  "first sentence example",
  "second something going"
  )
  )

  "a chunk using a trained POS" should behave like testChunkingWithTrainedPOS(
  Array(
  "first sentence example",
  "second something going"
  )
  )

  "a chunk with user input tags" should behave like testUserInputPOSTags(
    Array("the little yellow dog barked at the cat"),
    Array(Array("DT", "JJ", "JJ", "NN", "VBD", "IN", "DT", "NN")),
    "<DT>?<JJ>*<NN>"
  )

  "another chunk with user input tags" should behave like testUserInputPOSTags(
    Array("Rapunzel let down her long golden hair"),
    Array(Array("NNP", "VBD", "RP", "PP$", "JJ", "JJ", "NN")),
    "<DT|PP\\$>?<JJ>*<NN>" //(<DT>)|(<PP\$>)?(<JJ>)*(<NN>)
  )

  "other chunk with user input tags" should behave like testUserInputPOSTags(
    Array("Rapunzel let down her long golden hair"),
    Array(Array("NNP", "VBD", "RP", "PP$", "JJ", "JJ", "NN")),
    "<NNP>+"
  )


  "chunk with user input tags with no mathces" should behave like testUserInputPOSTags(
    Array("Rapunzel let down her long golden hair"),
    Array(Array("NNP", "VBD", "RP", "PP$", "JJ", "JJ", "NN")),
    "<NN><NN>"
  )

 val testingPhrases = List(
  Phrases(sentence = "the little yellow dog barked at the cat",
          POSFormatSentence = Array("DT", "JJ", "JJ", "NN", "VBD", "IN", "DT", "NN"),
          regexParser = "<DT>?<JJ>*<NN>",
          correctChunkPhrases = Array("the little yellow dog", "the cat")),

    Phrases(sentence = "Rapunzel let down her long golden hair",
            POSFormatSentence = Array("NNP", "VBD", "RP", "PP$", "JJ", "JJ", "NN"),
            regexParser = "<DT|PP\\$>?<JJ>*<NN>",
            correctChunkPhrases = Array("her long golden hair")),

    Phrases(sentence = "Rapunzel let down her long golden hair",
            POSFormatSentence = Array("NNP", "VBD", "RP", "PP$", "JJ", "JJ", "NN"),
            regexParser = "<NNP>+",
            correctChunkPhrases = Array("Rapunzel")),

    Phrases(sentence = "money market fund",
            POSFormatSentence = Array("NN", "NN", "NN"),
            regexParser = "<NN><NN>",
            correctChunkPhrases = Array("money market")),

    Phrases(sentence = "In the emergency room his ultrasound showed mild hydronephrosis with new perinephric fluid surrounding the transplant kidney",
      POSFormatSentence = Array("IN","DT","NN","NN","PRP$","NN","VBD","JJ","NN","IN","JJ","JJ","NN","VBG","DT","NN","NN"),
      regexParser = "<NN><NN>",
      correctChunkPhrases = Array("emergency room", "transplant kidney")),

   Phrases(sentence = "this should not return a chunk phrase because there is no match",
          POSFormatSentence = Array("DT", "MD", "RB","VB", "DT", "NN", "NN", "IN", "EX", "VBZ", "DT", "NN"),
          regexParser = "<NN><NNP>",
          correctChunkPhrases = Array())

  )

  "several chunks with user input tags" should behave like testSeveralSentences(testingPhrases)

  import SparkAccessor.spark.implicits._

  val document = Seq(
    ("the little yellow dog barked at the cat", Array("DT", "JJ", "JJ", "NN", "VBD", "IN", "DT", "NN")),
    ("Rapunzel let down her long golden hair", Array("NNP", "VBD", "RP", "PP$", "JJ", "JJ", "NN")),
    ("money market fund", Array("NN", "NN", "NN"))
  ).toDS.toDF("text", "tags")

  "chunk with a document" should behave like testDocument(document, "<DT>?<JJ>*<NN>")

}
