package com.jsl.nlp.annotators.pos.perceptron

import scala.collection.mutable.{Map => MMap, Set => MSet}
import scala.util.Random

/**
  * Created by Saif Addin on 5/16/2017.
  */
class AveragedPerceptron {

  val weights: MMap[String, MMap[String, Double]] = MMap()
  val classes: MSet[String] = MSet()

  private var nIteration: Int = 0
  private val totals: MMap[(String, String), Int] = MMap()
  private val timestamps: MMap[(String, String), Int] = MMap()

  def predict(features: Array[String]): String = {
    val scores: MMap[String, Double] = MMap()
    features
      .filter(weights.contains)
      .map(weights.get)
      .foreach{case (category, weight) => scores.getOrElseUpdate(category, 0.0) += weight}
    classes.maxBy((category) => scores(category))
  }

  def averageWeights = {
    weights.foreach{case (feature, featureWeights) => {
      val newWeights: MMap[String, Double] = MMap()
      featureWeights.foreach{case (category, weight) => {
        val param = (feature, category)
        var total = totals(param)
        total += ((nIteration - timestamps(param)) * weight)
        newWeights(category) = total / nIteration.toDouble
        weights(feature) = newWeights
      }}
    }}
  }

  def update(truth: String, guess: String, features: Array[String]): Unit = {
    def updateFeature(feature: String, category: String, weight: Double, value: Double) = {
      val param = (feature, category)
      totals(param) += ((nIteration - timestamps(param)) * weight)
      timestamps(param) = nIteration
      weights(feature)(category) = weight + value
    }
    nIteration += 1
    if (truth != guess) {
      features.foreach(feature => {
        val featureWeights = weights.getOrElseUpdate(feature, MMap())
        updateFeature(truth, feature, featureWeights.getOrElse(truth, 0.0), 1.0)
        updateFeature(guess, feature, featureWeights.getOrElse(guess, 0.0), -1.0)
      })
    }
  }

}
object AveragedPerceptron {
  def train(nIterations: Int, examples: Array[(Array[String], String)]): AveragedPerceptron = {
    val model = new AveragedPerceptron()
    val shuffleExamples = Random.shuffle(examples)
    (1 to nIterations).foreach { _ =>
      shuffleExamples.foreach{case (features, answer) => {
        val guess = model.predict(features)
        if (guess != answer) {
          features.foreach(feature => {
            model.weights(feature)(answer) += 1
            model.weights(feature)(guess) += -1
          })
        }
      }}
    }
    model
  }
}