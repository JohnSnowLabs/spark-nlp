/*
 * Copyright 2017-2022 John Snow Labs
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.DocumentAssembler
import com.johnsnowlabs.nlp.annotator._
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.tags.FastTest
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.DataFrame
import org.scalatest.flatspec.AnyFlatSpec

class ChunkerTestSpec extends AnyFlatSpec with ChunkerBehaviors {

  "a chunker using a trained POS" should behave like testChunkingWithTrainedPOS(
    Array(
      "first sentence example and with another cat",
      "second something going and with another dog"))

  "a chunker using a trained POS" should behave like testChunkingWithTrainedPOSUsingPipeline(
    Array("first sentence example", "second something going"))

  "a chunker with user input POS tags" should behave like testUserInputPOSTags(
    Array("the little yellow dog barked at the cat"),
    "src/test/resources/anc-pos-corpus-small/testUserInputPOSTags.txt",
    Array("<DT>?<JJ>*<NN>") // (<DT>)?(<JJ>)*(<NN>)
  )

  "a chunker with a regex parser that does not match" should behave like testRegexDoesNotMatch(
    Array("this should not return a chunk phrase because there is no match"),
    "src/test/resources/anc-pos-corpus-small/testRegexDoesNotMatch.txt",
    Array("<NNP>+"))

  val testingPhrases = List(
    SentenceParams(
      sentence = "the little yellow dog barked at the cat",
      POSFormatSentence = Array("DT", "JJ", "JJ", "NN", "VBD", "IN", "DT", "NN"),
      regexParser = Array("<DT>?<JJ>*<NN>"), // (<DT>)?(<JJ>)*(<NN>)
      correctChunkPhrases = Array("the little yellow dog", "the cat")),
    SentenceParams(
      sentence = "Rapunzel let down her long golden hair",
      POSFormatSentence = Array("NNP", "VBD", "RP", "PP$", "JJ", "JJ", "NN"),
      regexParser = Array("<DT|PP\\$>?<JJ>*<NN>"), // (<DT>)|(<PP\$>)?(<JJ>)*(<NN>)
      correctChunkPhrases = Array("her long golden hair")),
    SentenceParams(
      sentence = "Rapunzel let down her long golden hair",
      POSFormatSentence = Array("NNP", "VBD", "RP", "PP$", "JJ", "JJ", "NN"),
      regexParser = Array("<NNP>+"), // (<NNP>)+
      correctChunkPhrases = Array("Rapunzel")),
    SentenceParams(
      sentence = "money market fund",
      POSFormatSentence = Array("NN", "NN", "NN"),
      regexParser = Array("<NN><NN>"), // (<NN>)(<NN>)
      correctChunkPhrases = Array("money market")),
    SentenceParams(
      sentence =
        "In the emergency room his ultrasound showed mild hydronephrosis with new perinephric fluid surrounding the transplant kidney",
      POSFormatSentence = Array(
        "IN",
        "DT",
        "NN",
        "NN",
        "PRP$",
        "NN",
        "VBD",
        "JJ",
        "NN",
        "IN",
        "JJ",
        "JJ",
        "NN",
        "VBG",
        "DT",
        "NN",
        "NN"),
      regexParser = Array("<NN><NN>"),
      correctChunkPhrases = Array("emergency room", "transplant kidney")),
    SentenceParams(
      sentence = "this should not return a chunk phrase because there is no match",
      POSFormatSentence =
        Array("DT", "MD", "RB", "VB", "DT", "NN", "NN", "IN", "EX", "VBZ", "DT", "NN"),
      regexParser = Array("<NN><NNP>"),
      correctChunkPhrases = Array()))

  "a chunker with several sentences and theirs regex parameter" should behave like testChunksGeneration(
    testingPhrases)

  import ResourceHelper.spark.implicits._

  val document: DataFrame = Seq(
    (
      "Rapunzel let down her long golden hair",
      Array("NNP", "VBD", "RP", "PP$", "JJ", "JJ", "NN")),
    ("money market fund", Array("NN", "NN", "NN"))).toDS.toDF("text", "labels")

  "a chunker with a multiple regex parsers" should behave like testMultipleRegEx(
    document,
    Array("<NNP>+", "<DT>?<JJ>*<NN>", "<DT|PP\\$>?<JJ>*<NN>"))

  "Chunker" should "gracefully ignore bad index within dirty inputs" taggedAs FastTest in {

    val text = Seq("""
    '<p id="p0001" num="1">\n12 \nABSTRACT \nThe present disclosure relates to a gutter fixture assembly. In a particular form the present disclosure \nrelates to a gutter hanger or bracket assembly. In one aspect, the rain gutter hanger assembly \ncomprises a base for securement with respect to a supporting structure, and a connector adapted for \n5 securement with respect to a portion of a rain gutter, and to bridge the base and the rain gutter, and \nwherein at least one of either of the base or the connector is adapted to interconnect with the other.  \n0\n\'N"\' \nN N \n\'~ N N&gt; \nN\'\' \nN \n""N \\ NN \n\\ N NNWNN \nN \n\\\\ ~ N ~\' N\' \nN N N N\' \nNA\' ~ \nAt " NV N\' \\ NNA N \nit ~4\\ ~ \nN\\ &gt;\\? N N\\N\' ~\\ ~/ N \nN NN N\' \nNN N NNN \nN" \'N A\' j\'~~ NN itkN N NNN " \nN N\\ N N\' ji\' i\' N *N,\'N \'N NN N *~A NiN~i\' ~ NN\' NN \nN NN / ~\' N N ~N\' \\N N~ \nNN A\' N" N NN ~N \'N~ \'&lt; NN N \nN\'N\' N \n"N"" C" ""\'s \n\'N ir\'\\N"\\ NN~., NN NNN N\'\' ~\' N \n~ \\ N~QN4 N\' N\\ V N\' N\\ *NN~ N \nNN N N NN N ~\' NN ~N N ""\'~\' N \nNN N\\ N N NNN \nN N N N \nNN N N A, \\N N\' \nNN NN N \nNN NTh \nNN \'K \nNN NN N &lt;N \n\'N N N NN \nNN N N NN N N \nN NN \n\'N N\' Ni NN \nN N NN N N \nN N N N N \nNN N N N" \nNN NN \nN N NN NN N N \nNN N\\ N N \nNN N N N \n\\N N N\' \nNN NN N N \nNN N N N\\ N iN N \nNN N NN \nN NN N N 1, \n\'N \nNN /\n</p>'
    """).toDF("text")

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val cleanUpPatterns = Array("<.*?>", "&lt;.*?&gt;")
    val allowed_tags = Array("<JJ.?>?<NN.?>+", "<VB.?>", "<JJ.?>", "<NN.?>+<IN><JJ.?>?<NN.?>+")

    val documentNormalizer = new DocumentNormalizer()
      .setInputCols("document")
      .setOutputCol("normalizedDocument")
      .setAction("clean")
      .setPatterns(cleanUpPatterns)
      .setReplacement(" ")
      .setPolicy("pretty_all")
      .setLowercase(true)

    val sentence = new SentenceDetector()
      .setInputCols("normalizedDocument")
      .setOutputCol("sentence")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("sentence"))
      .setOutputCol("token")

    val posTagger = PerceptronModel
      .pretrained("pos_anc")
      .setInputCols("sentence", "token")
      .setOutputCol("pos")

    val chunker = new Chunker()
      .setInputCols(Array("sentence", "pos"))
      .setOutputCol("chunks")
      .setRegexParsers(allowed_tags)

    val pipeline = new Pipeline()
      .setStages(
        Array(documentAssembler, documentNormalizer, sentence, tokenizer, posTagger, chunker))

    val pipelineDF = pipeline.fit(text).transform(text)
    pipelineDF.select("chunks.result").show(false)

  }

}
