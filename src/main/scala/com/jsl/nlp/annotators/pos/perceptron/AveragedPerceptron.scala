package com.jsl.nlp.annotators.pos.perceptron

import scala.collection.mutable.{Map => MMap, Set => MSet}
import scala.util.Random

/**
  * Created by Saif Addin on 5/16/2017.
  */
class AveragedPerceptron {

  val featureWeights: MMap[String, MMap[String, Double]] = MMap()
  val classes: MSet[String] = MSet()

  private var nIteration: Int = 0
  private val totals: MMap[(String, String), Int] = MMap()
  private val timestamps: MMap[(String, String), Int] = MMap()

  def predict(features: Map[String, Int]): String = {
    val scores: MMap[String, Double] = MMap().withDefault(_ => 0.0)
    features
      .filter{case (feature, value) => featureWeights.contains(feature) && value != 0}
      .map{case (feature, value ) => (featureWeights.get(feature), value)}
      .foreach{case (categoryWeights, value) => {
        categoryWeights.foreach{case (category, weight) => {
          scores(category) += value * weight
        }}
      }}
    classes.maxBy((category) => scores(category))
  }

  def averageWeights = {
    featureWeights.foreach{case (feature, featureWeights) => {
      val newWeights: MMap[String, Double] = MMap()
      featureWeights.foreach{case (category, weight) => {
        val param = (feature, category)
        val total = totals(param) + ((nIteration - timestamps(param)) * weight)
        newWeights(category) = total / nIteration.toDouble
        featureWeights(feature) = newWeights
      }}
    }}
  }

  def update(truth: String, guess: String, features: Array[String]): Unit = {
    def updateFeature(feature: String, category: String, weight: Double, value: Double) = {
      val param = (feature, category)
      totals(param) += ((nIteration - timestamps(param)) * weight)
      timestamps(param) = nIteration
      featureWeights(feature)(category) = weight + value
    }
    nIteration += 1
    if (truth != guess) {
      features.foreach(feature => {
        val featureWeights = featureWeights.getOrElseUpdate(feature, MMap())
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
            model.featureWeights(feature)(answer) += 1
            model.featureWeights(feature)(guess) += -1
          })
        }
      }}
    }
    model
  }
}