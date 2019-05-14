package com.johnsnowlabs.nlp.annotators.parser.dep

import com.johnsnowlabs.nlp.annotators.parser.dep.GreedyTransition._

import scala.collection.mutable


class Tagger(classes:Vector[ClassName], tag_dict:Map[Word, ClassNum]) {
  //println(s"Tagger.Classes = [${classes.mkString(",")}]")
  val getClassNum = classes.zipWithIndex.toMap.withDefaultValue(-1) // -1 => "CLASS-NOT-FOUND"
  //def getClassNum(class_name: ClassName): ClassNum = classes.indexOf(class_name) // -1 => "CLASS-NOT-FOUND"

  val perceptron = new Perceptron(classes.length)

  def get_features(word:List[Word], pos:List[ClassName], i:Int):Map[Feature,Score] = {
    val feature_set = Set(
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
    feature_set.map( f => (f, 1:Score) ).toMap
  }

  def train(sentences:List[Sentence], seed:Int):Float = {
    val rand = new util.Random(seed)
    rand.shuffle(sentences).map( s=>train_one(s) ).sum / sentences.length
  }
  def train_one(sentence:Sentence):Float = goodness(sentence, process(sentence, true))

  def tag(sentence:Sentence):List[ClassName] = process(sentence, false)

  def process(sentence:Sentence, train:Boolean):List[ClassName] = {
    //val context:List[Word] = ("%START%" :: "%PAD%" :: (sentence.map( _.norm ) :+ "%ROOT%" :+ "%END%"))
    val words_norm = sentence.map( _.norm )
    val words:List[Word] = (List("%START%","%PAD%") ::: words_norm ::: List("%ROOT%","%END%"))
    val gold_tags:List[ClassNum] = if(train) (List(-1,-1) ::: sentence.map( wd => getClassNum(wd.pos) ) ::: List(-1,-1)) else Nil

    val (i_final, all_tags) =
      words_norm.foldLeft( (2:Int, List[ClassName]("%START%","%PAD%")) ) { case ( (i, tags), word_norm ) => {
        val guess = tag_dict.getOrElse(word_norm, {   // Don't do the feature scoring if we already 'know' the right PoS
          val features = get_features(words, tags, i)
          val score = perceptron.score(features, if(train) perceptron.current else perceptron.average)
          val guessed = perceptron.predict( score )

          if(train) {// Update the perceptron
            //println(f"Training '${word_norm}%12s': ${classes(guessed)}%4s -> ${classes(gold_tags(i))}%4s :: ")
            perceptron.update( gold_tags(i), guessed, features.keys)
          }
          guessed // Use the guessed value for next prediction/learning step (rather than the truth...)
        })
        (i+1, tags :+ classes(guess))
      }}
    all_tags.drop(2)
  }

  def goodness(sentence:Sentence, fit:List[ClassName]):Float = {
    val gold = sentence.map( _.pos ).toVector
    val correct = fit.zip( gold ).count( pair => (pair._1 == pair._2))  / gold.length.toFloat
    correct
  }

  override def toString():String = {
    classes.mkString("tagger.classes=[","|","]" + System.lineSeparator()) +
      tag_dict.map({ case (norm, classnum) => s"$norm=$classnum" }).mkString("tagger.tag_dict=[","|","]" + System.lineSeparator()) +
      System.lineSeparator() +
      perceptron.toString
  }

  def getPerceptronAsArray: Iterator[String] = {
    perceptron.toString().split(System.lineSeparator()).toIterator
  }

  def getTaggerAsArray: Iterator[String] = {
    this.toString().split(System.lineSeparator()).toIterator
  }

}
object Tagger {  // Here, tag == Part-of-Speech

  // Make a tag dictionary for single-tag words : So that they can be 'resolved' immediately, as well as the class list
  def classes_and_tagdict(training_sentences: List[Sentence]): (Vector[ClassName], Map[Word, ClassNum])  = {

    def functional_approach(): (Set[ClassName], Map[ Word, Map[ClassName, Int] ]) = {
      // takes 120ms on full training data
      training_sentences.foldLeft( ( Set[ClassName](), Map[ Word, Map[ClassName, Int] ]() ) ) { case ( (classes, map), sentence ) => {
        sentence.foldLeft( (classes, map) ) { case ( (classes, map), word_data) => {
          val count = map.getOrElse(word_data.norm, Map[ClassName, Int]()).getOrElse(word_data.pos, 0:Int)
          (
            classes + word_data.pos,
            map + (( word_data.norm,
              map.getOrElse(word_data.norm, Map[ClassName, Int]())
                + ((word_data.pos, count+1))
            ))
          )
        }}
      }}
    }

    def mutational_approach(): (mutable.Set[ClassName], mutable.Map[ Word, mutable.Map[ClassName, Int] ]) = {
      // takes 60ms on full training data !
      val class_set = mutable.Set[ClassName]()
      val full_map  = mutable.Map[ Word, mutable.Map[ClassName, Int] ]()
      //.withDefault( k => mutable.Map[ClassName, Int]().withDefaultValue(0) )       // FAIL - reuse
      //.withDefaultValue( new mutable.Map[ClassName, Int]().withDefaultValue(0) )   // FAIL - types

      for {
        sentence <- training_sentences
        word_data <- sentence
      } {
        class_set += word_data.pos
        full_map.getOrElseUpdate(word_data.norm, mutable.Map[ClassName, Int]().withDefaultValue(0))(word_data.pos) += 1
        //println(s"making (${word_data.norm})(${word_data.pos})=>(${full_map(word_data.norm)(word_data.pos)})")
      }

      (class_set, full_map)
    }

    // First, get the set of classnames, and the counts for all the words and tags
    val (class_set, full_map) = if(false) functional_approach() else mutational_approach()

    // Convert the set of classes into a nice map, with indexer
    val classes = class_set.toVector.sorted  // This is alphabetical
    val class_map = classes.zipWithIndex.toMap
    println(s"Classes = [${classes.mkString(",")}]")

    val freq_thresh = 20
    val ambiguity_thresh = 0.97

    // Now, go through the full_map, and work out which are worth 'resolving' immediately - and return a suitable tagdict
    val tag_dict = mutable.Map[Word, ClassNum]().withDefaultValue(0)
    for {
      (norm, classes) <- full_map
      if(classes.values.sum >= freq_thresh)  // ignore if not enough samples
      (cl, v) <- classes
      if(v >= classes.values.sum * ambiguity_thresh) // Must be concentrated (in fact, cl must be unique... since >50%)
    } {
      tag_dict(norm) = class_map(cl)
      //println(s"${norm}=${cl}")
    }
    (classes, tag_dict.toMap)
  }

  def load(lines:Iterator[String]):Tagger = {
    var (classes, tag_dict)=(Array[ClassName](), mutable.Map[Word, ClassNum]())

    val tagger_classes = """tagger.classes=\[(.*)\]""".r
    val tagger_tag_dict = """tagger.tag_dict=\[(.*)\]""".r
    def parse(lines: Iterator[String]):Unit = lines.next match {
      case tagger_classes(data) if data.nonEmpty => {
        classes = data.split('|')
        parse(lines)
      }
      case tagger_tag_dict(data) if data.nonEmpty => {
        tag_dict ++= data.split('|').map( nc => { val arr = nc.split('='); (arr(0),arr(1).toInt) })  // println(s"Tagger pair : $nc");
        parse(lines)
      }
      case _ => () // line not understood : Finish
    }
    parse(lines)

    val t = new Tagger(classes.toVector, tag_dict.toMap)
    t.perceptron.load(lines)
    t
  }

}
