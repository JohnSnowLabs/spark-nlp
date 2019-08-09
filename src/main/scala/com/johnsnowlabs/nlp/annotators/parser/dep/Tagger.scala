package com.johnsnowlabs.nlp.annotators.parser.dep

import com.johnsnowlabs.nlp.annotators.parser.dep.GreedyTransition._

import scala.collection.mutable


class Tagger(classes:Vector[ClassName], tagDict:Map[Word, ClassNum]) extends Serializable {
  private val getClassNum = classes.zipWithIndex.toMap.withDefaultValue(-1) // -1 => "CLASS-NOT-FOUND"

  private val perceptron = new Perceptron(classes.length)

  def getFeatures(word:List[Word], pos:List[ClassName], i:Int):Map[Feature,Score] = {
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
    featureSet.map( f => (f, 1:Score) ).toMap
  }

  def train(sentences:List[Sentence], seed:Int):Float = {
    val rand = new util.Random(seed)
    rand.shuffle(sentences).map( s=>trainSentence(s) ).sum / sentences.length
  }
  def trainSentence(sentence:Sentence):Float = goodness(sentence, process(sentence, train=true))

  def tag(sentence:Sentence):List[ClassName] = process(sentence, train=false)

  def process(sentence:Sentence, train:Boolean):List[ClassName] = {
    val wordsNorm = sentence.map( _.norm )
    val words:List[Word] = List("%START%","%PAD%") ::: wordsNorm ::: List("%ROOT%","%END%")
    val goldTags:List[ClassNum] = if(train) List(-1,-1) ::: sentence.map( wd => getClassNum(wd.pos) ) ::: List(-1,-1) else Nil

    val (_, allTags) =
      wordsNorm.foldLeft( (2:Int, List[ClassName]("%START%","%PAD%")) ) { case ( (i, tags), wordNorm ) => {
        val guess = tagDict.getOrElse(wordNorm, {   // Don't do the feature scoring if we already 'know' the right PoS
          val features = getFeatures(words, tags, i)
          val score = perceptron.score(features, if(train) perceptron.current else perceptron.average)
          val guessed = perceptron.predict( score )

          if(train) {
            perceptron.update( goldTags(i), guessed, features.keys)
          }
          guessed // Use the guessed value for next prediction/learning step (rather than the truth...)
        })
        (i+1, tags :+ classes(guess))
      }}
    allTags.drop(2)
  }

  def goodness(sentence:Sentence, fit:List[ClassName]):Float = {
    val gold = sentence.map( _.pos ).toVector
    val correct = fit.zip( gold ).count(pair => pair._1 == pair._2)  / gold.length.toFloat
    correct
  }

  override def toString: String = {
    classes.mkString("tagger.classes=[","|","]" + System.lineSeparator()) +
      tagDict.map({ case (norm, classnum) => s"$norm=$classnum" }).mkString("tagger.tag_dict=[","|","]" + System.lineSeparator()) +
      System.lineSeparator() +
      perceptron.toString
  }

  def getPerceptronAsIterator: Iterator[String] = {
    perceptron.toString().split(System.lineSeparator()).toIterator
  }

  def getTaggerAsIterator: Iterator[String] = {
    this.toString().split(System.lineSeparator()).toIterator
  }

}
object Tagger {  // Here, tag == Part-of-Speech

  def load(lines:Iterator[String]): Tagger = {
    var (classes, tagDict)=(Array[ClassName](), mutable.Map[Word, ClassNum]())

    val taggerClasses = """tagger.classes=\[(.*)\]""".r
    val taggerTagDict = """tagger.tag_dict=\[(.*)\]""".r
    def parse(lines: Iterator[String]):Unit = lines.next match {
      case taggerClasses(data) if data.nonEmpty => {
        classes = data.split('|')
        parse(lines)
      }
      case taggerTagDict(data) if data.nonEmpty => {
        tagDict ++= data.split('|').map( nc => { val arr = nc.split('='); (arr(0),arr(1).toInt) })  // println(s"Tagger pair : $nc");
        parse(lines)
      }
      case _ => () // line not understood : Finish
    }
    parse(lines)

    val t = new Tagger(classes.toVector, tagDict.toMap)
    t.perceptron.load(lines)
    t
  }

}
