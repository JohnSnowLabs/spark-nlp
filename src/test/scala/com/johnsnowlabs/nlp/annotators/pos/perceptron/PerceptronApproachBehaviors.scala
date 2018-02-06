package com.johnsnowlabs.nlp.annotators.pos.perceptron

import com.johnsnowlabs.nlp.annotators.common.{TaggedSentence, TaggedWord, TokenizedSentence}
import com.johnsnowlabs.nlp.util.io.{ExternalResource, ReadAs, ResourceHelper}
import com.johnsnowlabs.nlp.{Annotation, AnnotatorBuilder, DataBuilder}
import org.apache.spark.sql.{Dataset, Row}
import org.scalatest._

import scala.collection.mutable.{Set => MSet}

/**
  * Created by Saif Addin on 5/18/2017.
  */
trait PerceptronApproachBehaviors { this: FlatSpec =>

  def isolatedPerceptronTraining(trainingSentencesPath: String): Unit = {
    s"Average Perceptron tagger" should "successfully train a provided wsj corpus" in {
      val trainingSentences = ResourceHelper.parseTupleSentences(ExternalResource(trainingSentencesPath, ReadAs.LINE_BY_LINE, Map("delimiter" -> "|")))
      val nIterations = 5
      val tagger = new PerceptronApproach()
        .setCorpus(ExternalResource(trainingSentencesPath, ReadAs.LINE_BY_LINE, Map("delimiter" -> "|")))
        .fit(DataBuilder.basicDataBuild("dummy"))
      val model = tagger.getModel
      val nWords = trainingSentences.map(_.words.length).sum
      assert(
        nWords * nIterations == model.getUpdateIterations,
        s"because Words: $nWords -- nIterations: $nIterations -- multip: ${nWords * nIterations}" +
          s"-- model iters: ${model.getUpdateIterations}"
      )
      val tagSet: MSet[String] = MSet()
      trainingSentences.foreach{s => {
        s.tags.foreach(tagSet.add)
      }}
      assert(tagSet.size == model.getTags.length)
      tagSet.foreach(tag => assert(model.getTags.contains(tag)))
    }
  }

  def isolatedPerceptronTagging(
                                 trainedTagger: PerceptronModel,
                                 targetSentences: Array[TokenizedSentence]
                               ): Unit = {
    s"Average Perceptron tagger" should "successfully tag all word sentences after training" in {
      val result = trainedTagger.tag(targetSentences)
      assert(result.head.words.length == targetSentences.head.tokens.length, "because tagger returned less than" +
        " the amount of appropriate tagged words")
    }
  }

  def isolatedPerceptronTagCheck(
                                  trainedTagger: PerceptronModel,
                                  targetSentence: Array[TokenizedSentence],
                                  expectedTags: Array[String]
                                ): Unit = {
    s"Average Perceptron tagger" should "successfully return expected tags" in {
      val resultTags = trainedTagger.tag(targetSentence).head
      val resultContent = resultTags.taggedWords.zip(expectedTags)
        .filter(rte => rte._1.tag != rte._2)
        .map(rte => (rte._1.word, (rte._1.tag, rte._2)))
      assert(resultTags.words.length == expectedTags.length, s"because tag amount ${resultTags.words.length} differs from" +
        s" expected ${expectedTags.length}")
      assert(resultTags.taggedWords.zip(expectedTags).forall(t => t._1.tag == t._2), s"because expected tags do not match returned" +
        s" tags.\n------------------------\n(word,(result,expected))\n-----------------------\n${resultContent.mkString("\n")}")
    }
  }

  def sparkBasedPOSTagger(dataset: => Dataset[Row]): Unit = {
    "a Perceptron POS tagger Annotator" should s"successfully tag sentences " in {
      val df = AnnotatorBuilder.withFullPOSTagger(dataset)
      df.show(5)
      val posCol = df.select("pos")
      assert(posCol.first.getSeq[Row](0).head.getAs[String](0)  == "pos", "Annotation type should be equal to `pos`")
    }

    it should "tag each word sentence" in {
      val df = AnnotatorBuilder.withFullPOSTagger(dataset)
      val posCol = df.select("pos")
      val tokensCol = df.select("token")
      val tokens: Seq[String] = tokensCol.collect.flatMap(r => r.getSeq[Row](0)).flatMap(a => a.getMap[String, Any](4).get("token")).map(_.toString)
      val taggedWords: Seq[String] = posCol.collect.flatMap(r => r.getSeq[Row](0)).flatMap(a => a.getMap[String, Any](4).get("word")).map(_.toString)
      tokens.foreach { token: String =>
        assert(taggedWords.contains(token), s"Token ${token} should be list of tagged words")
      }
    }

    it should "annotate with the correct word index" in {
      case class IndexedWord(word: String, begin: Int, end:Int) {
        def equals(o: IndexedWord): Boolean = { this.word == o.word && this.begin == o.begin && this.end == o.end }
      }
      val df = AnnotatorBuilder.withFullPOSTagger(dataset)
      val posCol = df.select("pos")
      val tokensCol = df.select("token")
      val tokens: Seq[IndexedWord] = tokensCol.collect.flatMap(r => r.getSeq[Row](0)).map(a => IndexedWord(a.getString(3), a.getInt(1), a.getInt(2)))
      val taggedWords: Seq[IndexedWord] = posCol.collect.flatMap(r => r.getSeq[Row](0)).map(a => IndexedWord(a.getMap[String, String](4)("word"), a.getInt(1), a.getInt(2)))
      taggedWords.foreach { word: IndexedWord =>
        assert(tokens.exists { token => token.equals(word) }, s"Indexed word ${word} should be included in ${tokens.filter(t => t.word == word.word)}")
      }
    }
  }
}
