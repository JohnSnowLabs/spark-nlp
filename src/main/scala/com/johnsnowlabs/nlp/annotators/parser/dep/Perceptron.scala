package com.johnsnowlabs.nlp.annotators.parser.dep

import com.johnsnowlabs.nlp.annotators.parser.dep.GreedyTransition._

import scala.collection.mutable

class Perceptron(n_classes:Int) {
  // These need not be visible outside the class
  type TimeStamp = Int

  case class WeightLearner(current:Int, total:Int, ts:TimeStamp) {
    def add_change(change:Int) = {
      WeightLearner(current + change, total + current*(seen-ts), seen)
    }
  }

  type ClassToWeightLearner = mutable.Map[ ClassNum,  WeightLearner ]  // tells us the stats of each class (if present)

  // The following are keyed on feature (to keep tally of total numbers into each, and when)(for the TRAINING phase)
  val learning =  mutable.Map.empty[
    String,    // Corresponds to Feature.name
    mutable.Map[
      String,  // Corresponds to Feature.data
      ClassToWeightLearner
      ]
    ] // This is hairy and mutable...

  // Number of instances seen - used to measure how 'old' each total is
  var seen:TimeStamp = 0

  type ClassVector = Vector[Score]

  def predict(classnum_vector : ClassVector) : ClassNum = { // Return best class guess for this vector of weights
    classnum_vector.zipWithIndex.maxBy(_._1)._2   // in vector order (stabilizes) ///NOT : (and alphabetically too)
  }

  def current(w : WeightLearner):Float =  w.current
  def average(w : WeightLearner):Float = (w.current*(seen-w.ts) + w.total) / seen // This is dynamically calculated
  // No need for average_weights() function - it's all done dynamically

  def score(features: Map[Feature, Score], score_method: WeightLearner => Float): ClassVector = { // Return 'dot-product' score for all classes
    if(false) { // This is the functional version : 3023ms for 1 train_all, and 0.57ms for a sentence
      features
        .filter( pair => pair._2 != 0 )  // if the 'score' multiplier is zero, skip
        .foldLeft( Vector.fill(n_classes)(0:Float) ){ case (acc, (Feature(name,data), score)) => {  // Start with a zero classnum->score vector
        learning
          .getOrElse(name, Map[String,ClassToWeightLearner]())   // This is first level of feature access
          .getOrElse(data, Map[ ClassNum,  WeightLearner ]())       // This is second level of feature access and is a Map of ClassNums to Weights (or NOOP if not there)
          .foldLeft( acc ){ (acc_for_feature, cn_wl) => { // Add each of the class->weights onto our score vector
          val classnum:ClassNum = cn_wl._1
          val weight_learner:WeightLearner = cn_wl._2
          acc_for_feature.updated(classnum, acc_for_feature(classnum) + score * score_method(weight_learner))
        }}
      }}
    }
    else { //  This is the mutable version : 2493ms for 1 train_all, and 0.45ms for a sentence
      val scores = (new Array[Score](n_classes)) // All 0?

      features
        .filter( pair => pair._2 != 0 )  // if the 'score' multiplier is zero, skip
        .map{ case (Feature(name,data), score) => {  // Ok, so given a particular feature, and score to weight it by
        if(learning.contains(name) && learning(name).contains(data)) {
          learning(name)(data).foreach { case (classnum, weight_learner) => {
            //println(s"classnum = ${classnum} n_classes=${n_classes}")
            scores(classnum) += score * score_method(weight_learner)
          }}
        }
      }}
      scores.toVector
    }
  }

  def update(truth:ClassNum, guess:ClassNum, features:Iterable[Feature]): Unit = { // Hmmm ..Unit..
    seen += 1
    if(truth != guess) {
      for {
        feature <- features
      } {
        learning.getOrElseUpdate(feature.name, mutable.Map[FeatureData, ClassToWeightLearner]() )
        var this_map = learning(feature.name).getOrElseUpdate(feature.data, mutable.Map[ClassNum, WeightLearner]() )

        //println(s"  update [${feature.name},${feature.data}] : ${learning(feature.name)(feature.data)}")
        if(this_map.contains(guess)) {
          this_map.update(guess, this_map(guess).add_change(-1))
        }
        this_map.update(truth, this_map.getOrElse( truth, WeightLearner(0,0,seen) ).add_change(+1))

        learning(feature.name)(feature.data) = this_map
      }
    }
  }

  override def toString():String = {
    s"perceptron.seen=[$seen]" + System.lineSeparator() +
      learning.map({ case (feature_name, m1) => {
        m1.map({ case (feature_data, cn_feature) => {
          cn_feature.map({ case (cn, feature) => {
            s"$cn:${feature.current},${feature.total},${feature.ts}"
          }}).mkString(s"${feature_data}[","|","]" + System.lineSeparator())
        }}).mkString(s"${feature_name}{" + System.lineSeparator(),"","}" + System.lineSeparator())
      }}).mkString("perceptron.learning={" + System.lineSeparator(),"","}" + System.lineSeparator())
  }

  def load(lines:Iterator[String]):Unit = {
    val perceptron_seen     = """perceptron.seen=\[(.*)\]""".r
    val perceptron_feat_n   = """(.*)\{""".r
    val perceptron_feat_d   = """(.*)\[(.*)\]""".r
    val ilines = lines.toIterator
    def parse(lines: Iterator[String]):Unit = if(ilines.hasNext) ilines.next match {
      case perceptron_seen(data) => {
        seen = data.toInt
        parse(lines)
      }
      case "perceptron.learning={" => {
        parse_featurename(ilines)
        parse(lines)
      }
      case _ => () // line not understood : Finished with perceptron
    }
    def parse_featurename(lines: Iterator[String]):Unit = if(lines.hasNext) lines.next match {
      case perceptron_feat_n(feature_name) => {
        //println(s"Reading FeatureName[$feature_name]")
        learning.getOrElseUpdate(feature_name, mutable.Map[FeatureData, ClassToWeightLearner]() )
        parse_featuredata(feature_name, lines)
        parse_featurename(lines) // Go back for more featurename sections
      }
      case _ => () // line not understood : Finished with featurename
    }
    def parse_featuredata(feature_name:String, lines: Iterator[String]):Unit = if(lines.hasNext) lines.next match {
      case perceptron_feat_d(feature_data, classnum_weight) => {
        //println(s"Reading FeatureData[$feature_data]")
        learning(feature_name).getOrElseUpdate(feature_data, mutable.Map[ClassNum, WeightLearner]() )
        classnum_weight.split('|').map( cw => {
          val cn_wt = cw.split(':').map(_.split(',').map(_.toInt))
          // println(s"Tagger node : $cn_wt");
          learning(feature_name)(feature_data) += (( cn_wt(0)(0), WeightLearner(cn_wt(1)(0), cn_wt(1)(1), cn_wt(1)(2)) ))
        })
        parse_featuredata(feature_name, lines)  // Go back for more featuredata lines
      }
      case _ => () // line not understood : Finished with featuredata
    }
    parse(lines)
  }

}