package com.johnsnowlabs.nlp.annotators.parser.dep

import com.johnsnowlabs.nlp.annotators.parser.dep.GreedyTransition.{ClassNum, FeatureData, FeatureName, Score}

import scala.collection.mutable

case class Feature(name: FeatureName, data: FeatureData)

class Perceptron(numberOfClasses:Int) {
  // These need not be visible outside the class
  type TimeStamp = Int

  case class WeightLearner(current:Int, total:Int, ts:TimeStamp) {
    def addChange(change:Int) = {
      WeightLearner(current + change, total + current*(seen-ts), seen)
    }
  }

  type ClassToWeightLearner = mutable.Map[ClassNum, WeightLearner]  // tells us the stats of each class (if present)

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

  def predict(classnumVector : ClassVector) : ClassNum = { // Return best class guess for this vector of weights
    classnumVector.zipWithIndex.maxBy(_._1)._2   // in vector order (stabilizes) ///NOT : (and alphabetically too)
  }

  def current(w : WeightLearner):Double =  w.current
  def average(w : WeightLearner):Double = (w.current*(seen-w.ts) + w.total) / seen // This is dynamically calculated
  // No need for average_weights() function - it's all done dynamically

  def score(features: Map[Feature, Score], scoreMethod: WeightLearner => Double): ClassVector = { // Return 'dot-product' score for all classes
    if(false) { // This is the functional version : 3023ms for 1 train_all, and 0.57ms for a sentence
      features
        .filter( pair => pair._2 != 0 )  // if the 'score' multiplier is zero, skip
        .foldLeft( Vector.fill(numberOfClasses)(0:Double) ){ case (acc, (Feature(name,data), score)) => {  // Start with a zero classnum->score vector
        learning
          .getOrElse(name, Map[String, ClassToWeightLearner]())   // This is first level of feature access
          .getOrElse(data, Map[ ClassNum,  WeightLearner ]())       // This is second level of feature access and is a Map of ClassNums to Weights (or NOOP if not there)
          .foldLeft( acc ){ (accForFeature, classNumAndWeights) => { // Add each of the class->weights onto our score vector
          val classNum: ClassNum = classNumAndWeights._1
          val weightLearner:WeightLearner = classNumAndWeights._2
          accForFeature.updated(classNum, accForFeature(classNum) + score * scoreMethod(weightLearner))
        }}
      }}
    }
    else { //  This is the mutable version : 2493ms for 1 train_all, and 0.45ms for a sentence
      val scores = new Array[Score](numberOfClasses) // All 0?

      features
        .filter( pair => pair._2 != 0 )  // if the 'score' multiplier is zero, skip
        .foreach{ case (Feature(name,data), score) => {  // Ok, so given a particular feature, and score to weight it by
        if(learning.contains(name) && learning(name).contains(data)) {
          learning(name)(data).foreach { case (classnum, weight_learner) => {
            scores(classnum) += score * scoreMethod(weight_learner)
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
          this_map.update(guess, this_map(guess).addChange(-1))
        }
        this_map.update(truth, this_map.getOrElse( truth, WeightLearner(0,0,seen) ).addChange(+1))

        learning(feature.name)(feature.data) = this_map
      }
    }
  }

  override def toString(): String = {
    s"perceptron.seen=[$seen]\n" +
      learning.map({ case (feature_name, m1) => {
        m1.map({ case (feature_data, cn_feature) => {
          cn_feature.map({ case (cn, feature) => {
            s"$cn:${feature.current},${feature.total},${feature.ts}"
          }}).mkString(s"$feature_data[","|","]\n")
        }}).mkString(s"$feature_name{\n","","}\n")
      }}).mkString("perceptron.learning={\n","","}\n")
  }

  def loadUsedInPerceptron(lines:Iterator[String]):Unit = {
    val perceptronSeen     = """perceptron.seen=\[(.*)\]""".r
    val perceptronFeatN   = """(.*)\{""".r
    val perceptronFeatD   = """(.*)\[(.*)\]""".r
    def parse(lines: Iterator[String]):Unit = if(lines.hasNext) lines.next match {
      case perceptronSeen(data) => {
        seen = data.toInt
        parse(lines)
      }
      case "perceptron.learning={" => {
        parseFeatureName(lines)
        parse(lines)
      }
      case _ => () // line not understood : Finished with perceptron
    }
    def parseFeatureName(lines: Iterator[String]):Unit = if(lines.hasNext) lines.next match {
      case perceptronFeatN(feature_name) => {
        learning.getOrElseUpdate(feature_name, mutable.Map[FeatureData, ClassToWeightLearner]() )
        parseFeatureData(feature_name, lines)
        parseFeatureName(lines) // Go back for more featurename sections
      }
      case _ => () // line not understood : Finished with featurename
    }
    def parseFeatureData(feature_name:String, lines: Iterator[String]):Unit = if(lines.hasNext) lines.next match {
      case perceptronFeatD(feature_data, classnum_weight) => {
        learning(feature_name).getOrElseUpdate(feature_data, mutable.Map[ClassNum, WeightLearner]() )
        classnum_weight.split('|').map( cw => {
          val cn_wt = cw.split(':').map(_.split(',').map(_.toInt))
          learning(feature_name)(feature_data) += (( cn_wt(0)(0), WeightLearner(cn_wt(1)(0), cn_wt(1)(1), cn_wt(1)(2)) ))
        })
        parseFeatureData(feature_name, lines)  // Go back for more featuredata lines
      }
      case _ => () // line not understood : Finished with featuredata
    }
    parse(lines)
  }

}
