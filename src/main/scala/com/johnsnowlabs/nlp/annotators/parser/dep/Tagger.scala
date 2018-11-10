package com.johnsnowlabs.nlp.annotators.parser.dep

import com.johnsnowlabs.nlp.annotators.parser.dep.GreedyTransition._

import scala.io.Source


class Tagger(classes: Vector[ClassName], tagDict: Map[Word, ClassNum])  {

  private val getClassNum = classes.zipWithIndex.toMap.withDefaultValue(-1) // -1 => "CLASS-NOT-FOUND"
  val perceptron = new Perceptron(classes.length)

  def getFeatures(word:List[Word], pos:List[ClassName], i:Int): Map[Feature, Score] = {
    val featureSet = Set(
      Feature("bias",       ""),  // It's useful to have a constant feature, which acts sort of like a prior

      Feature("word",       word(i)),
      Feature("w suffix",   word(i).takeRight(3)),
      Feature("w pref1",    word(i).take(1)),

      Feature("tag-1",      pos(i-1)),
      Feature("tag-2",      pos(i-2)),
      Feature("tag-1-2",    s"${pos(i-1)} ${pos(i-2)}"),

      Feature("w,tag-1",    s"${word(i)} ${pos(i-1)}"),

      Feature("w-1",        word(i-1)),
      Feature("w-1 suffix", word(i-1).takeRight(3)),

      Feature("w-2",        word(i-2)),

      Feature("w+1",        word(i+1)),
      Feature("w+1 suffix", word(i+1).takeRight(3)),

      Feature("w+2",        word(i+2))
    )

    // All weights on this set of features are ==1
    featureSet.map(feature => (feature, 1:Score)).toMap
  }

  def train(sentences: List[Sentence], seed: Int): Double = {
    val rand = new util.Random(seed)
    rand.shuffle(sentences).map(sentence => trainOne(sentence)).sum / sentences.length
  }

  def trainOne(sentence:Sentence): Double = goodness(sentence, process(sentence, train = true))

  def goodness(sentence:Sentence, fit:List[ClassName]): Double = {
    val gold = sentence.map( _.pos ).toVector
    val correct = fit.zip( gold ).count( pair => pair._1 == pair._2)  / gold.length.toFloat
    //println(s"Part-of-Speech score : ${pct_fit_fmt_str(correct)}")
    correct
  }

  def process(sentence:Sentence, train:Boolean):List[ClassName] = {
    val wordsNorm = sentence.map( _.norm )
    val words: List[Word] = List("%START%","%PAD%") ::: wordsNorm ::: List("%ROOT%","%END%")
    val goldTags: List[ClassNum] = if(train) List(-1,-1) ::: sentence.map(wd => getClassNum(wd.pos) ) ::: List(-1,-1)
                                   else Nil

    val (_, allTags) =
      wordsNorm.foldLeft( (2:Int, List[ClassName]("%START%","%PAD%")) ) { case ( (i, tags), word_norm ) =>
        val guess = tagDict.getOrElse(word_norm, {   // Don't do the feature scoring if we already 'know' the right PoS
          val features = getFeatures(words, tags, i)
          val score = perceptron.dotProductScore(features, if(train) perceptron.current else perceptron.average)
          val guessed = perceptron.predict( score )

          if(train) {// Update the perceptron
            perceptron.update( goldTags(i), guessed, features.keys)
          }
          guessed // Use the guessed value for next prediction/learning step (rather than the truth...)
        })
        (i+1, tags :+ classes(guess))
      }
    allTags.drop(2)
  }

  def tagSentence(sentence: Sentence): List[ClassName] = process(sentence, train = false)

  def getPerceptronAsArray: Array[String] = {
      //val pruebas = perceptron.toString()
      perceptron.toString().split("\\n")
  }

  def getHarcodedPerceptron: Array[String] = {
    val filename = "src/test/resources/models/dep-model-small.txt"
    val fileContents = Source.fromFile(filename).mkString
    fileContents.split("\\n")
  }

}
