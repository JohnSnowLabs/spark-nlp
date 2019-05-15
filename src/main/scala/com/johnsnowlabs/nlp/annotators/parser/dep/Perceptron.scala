package com.johnsnowlabs.nlp.annotators.parser.dep

import com.johnsnowlabs.nlp.annotators.parser.dep.GreedyTransition._

import scala.collection.mutable

class Perceptron(nClasses:Int) {
  // These need not be visible outside the class
  type TimeStamp = Int

  case class WeightLearner(current:Int, total:Int, ts:TimeStamp) {
    def addChange(change:Int): WeightLearner = {
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

  def predict(classnumVector : ClassVector) : ClassNum = { // Return best class guess for this vector of weights
    classnumVector.zipWithIndex.maxBy(_._1)._2   // in vector order (stabilizes) ///NOT : (and alphabetically too)
  }

  def current(w : WeightLearner):Float =  w.current
  def average(w : WeightLearner):Float = (w.current*(seen-w.ts) + w.total) / seen // This is dynamically calculated
  // No need for average_weights() function - it's all done dynamically

  def score(features: Map[Feature, Score], scoreMethod: WeightLearner => Float): ClassVector = { // Return 'dot-product' score for all classes
    if(false) { // This is the functional version : 3023ms for 1 train_all, and 0.57ms for a sentence
      features
        .filter( pair => pair._2 != 0 )  // if the 'score' multiplier is zero, skip
        .foldLeft( Vector.fill(nClasses)(0:Float) ){ case (acc, (Feature(name,data), score)) => {  // Start with a zero classnum->score vector
        learning
          .getOrElse(name, Map[String,ClassToWeightLearner]())   // This is first level of feature access
          .getOrElse(data, Map[ ClassNum,  WeightLearner ]())       // This is second level of feature access and is a Map of ClassNums to Weights (or NOOP if not there)
          .foldLeft( acc ){ (accuracyForFeature, cnWL) => { // Add each of the class->weights onto our score vector
          val classnum:ClassNum = cnWL._1
          val weightLearner: WeightLearner = cnWL._2
          accuracyForFeature.updated(classnum, accuracyForFeature(classnum) + score * scoreMethod(weightLearner))
        }}
      }}
    }
    else { //  This is the mutable version : 2493ms for 1 train_all, and 0.45ms for a sentence
      val scores = new Array[Score](nClasses) // All 0?

      features
        .filter( pair => pair._2 != 0 )  // if the 'score' multiplier is zero, skip
        .foreach{ case (Feature(name,data), score) => {  // Ok, so given a particular feature, and score to weight it by
        if(learning.contains(name) && learning(name).contains(data)) {
          learning(name)(data).foreach { case (classnum, weightLearner) => {
            scores(classnum) += score * scoreMethod(weightLearner)
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
        val thisLearning = learning(feature.name).getOrElseUpdate(feature.data, mutable.Map[ClassNum, WeightLearner]() )

        if(thisLearning.contains(guess)) {
          thisLearning.update(guess, thisLearning(guess).addChange(-1))
        }
        thisLearning.update(truth, thisLearning.getOrElse( truth, WeightLearner(0,0,seen) ).addChange(+1))

        learning(feature.name)(feature.data) = thisLearning
      }
    }
  }

  override def toString: String = {
    s"perceptron.seen=[$seen]" + System.lineSeparator() +
      learning.map({ case (featureName, m1) => {
        m1.map({ case (featureData, cnFeature) => {
          cnFeature.map({ case (cn, feature) => {
            s"$cn:${feature.current},${feature.total},${feature.ts}"
          }}).mkString(s"${featureData}[","|","]" + System.lineSeparator())
        }}).mkString(s"${featureName}{" + System.lineSeparator(),"","}" + System.lineSeparator())
      }}).mkString("perceptron.learning={" + System.lineSeparator(),"","}" + System.lineSeparator())
  }

  def load(lines:Iterator[String]):Unit = {
    val perceptronSeen     = """perceptron.seen=\[(.*)\]""".r
    val perceptronFeatN   = """(.*)\{""".r
    val perceptronFeatD   = """(.*)\[(.*)\]""".r
    val ilines = lines
    def parse(lines: Iterator[String]):Unit = if(ilines.hasNext) ilines.next match {
      case perceptronSeen(data) => {
        seen = data.toInt
        parse(lines)
      }
      case "perceptron.learning={" => {
        parseFeatureName(ilines)
        parse(lines)
      }
      case _ => () // line not understood : Finished with perceptron
    }
    def parseFeatureName(lines: Iterator[String]):Unit = if(lines.hasNext) lines.next match {
      case perceptronFeatN(featureName) => {
        learning.getOrElseUpdate(featureName, mutable.Map[FeatureData, ClassToWeightLearner]() )
        parseFeatureData(featureName, lines)
        parseFeatureName(lines) // Go back for more featurename sections
      }
      case _ => () // line not understood : Finished with featurename
    }
    def parseFeatureData(featureName:String, lines: Iterator[String]):Unit = if(lines.hasNext) lines.next match {
      case perceptronFeatD(featureData, classnumWeight) => {
        learning(featureName).getOrElseUpdate(featureData, mutable.Map[ClassNum, WeightLearner]() )
        classnumWeight.split('|').map( cw => {
          val cnWT = cw.split(':').map(_.split(',').map(_.toInt))
          learning(featureName)(featureData) += (( cnWT(0)(0), WeightLearner(cnWT(1)(0), cnWT(1)(1), cnWT(1)(2)) ))
        })
        parseFeatureData(featureName, lines)  // Go back for more featuredata lines
      }
      case _ => () // line not understood : Finished with featuredata
    }
    parse(lines)
  }

}