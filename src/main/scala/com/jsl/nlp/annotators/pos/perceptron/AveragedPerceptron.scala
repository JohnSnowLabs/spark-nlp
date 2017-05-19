package com.jsl.nlp.annotators.pos.perceptron

import scala.collection.mutable.{Map => MMap}
import scala.util.Random

/**
  * Created by Saif Addin on 5/16/2017.
  */
class AveragedPerceptron {

  val featureWeights: MMap[String, MMap[String, Double]] = MMap()
  var classes: Set[String] = Set()

  private var nIteration: Int = 0
  private val totals: MMap[(String, String), Double] = MMap()
  private val timestamps: MMap[(String, String), Int] = MMap()

  def predict(features: Map[String, Int]): String = {
    val scores: MMap[String, Double] = MMap().withDefaultValue(0.0)
    features
      .filter{case (feature, value) => featureWeights.contains(feature) && value != 0}
      .map{case (feature, value ) => (featureWeights(feature), value)}
      .foreach{case (categoryWeights, value) => {
        categoryWeights.foreach{case (category, weight) => {
          scores(category) += (value * weight)
        }}
      }}
    classes.maxBy((category) => (scores(category),category))
  }

  def averageWeights = {
    featureWeights.foreach{case (feature, weights) => {
      val newWeights: MMap[String, Double] = MMap()
      weights.foreach{case (category, weight) => {
        val param = (feature, category)
        val total = totals(param) + ((nIteration - timestamps(param)) * weight)
        newWeights(category) = total / nIteration.toDouble
        featureWeights(feature) = newWeights
      }}
    }}
  }

  def update(truth: String, guess: String, features: Map[String, Int]): Unit = {
    def updateFeature(feature: String, category: String, weight: Double, value: Double) = {
      val param = (feature, category)
      totals(param) += ((nIteration - timestamps(param)) * weight)
      timestamps(param) = nIteration
      featureWeights(feature)(category) = weight + value
    }
    nIteration += 1
    if (truth != guess) {
      features.foreach{case (category, feature) => {
        val weights = featureWeights.getOrElseUpdate(category, MMap())
        updateFeature(truth, category, weights.getOrElse(truth, 0.0), 1.0)
        updateFeature(guess, category, weights.getOrElse(guess, 0.0), -1.0)
      }}
    }
  }

}
object AveragedPerceptron {
  def train(nIterations: Int, examples: Map[Map[String, Int], String]): AveragedPerceptron = {
    val model = new AveragedPerceptron()
    (1 to nIterations).foreach { _ =>
      Random.shuffle(examples).foreach{case (features, answer) => {
        val guess = model.predict(features)
        if (guess != answer) {
          model.update(answer, guess, features)
        }
      }}
    }
    model.averageWeights
    model
  }
}