package com.johnsnowlabs.ml.lstm

import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.distribution.UniformDistribution
import org.deeplearning4j.nn.conf.layers.{GravesBidirectionalLSTM, RnnOutputLayer}
import org.nd4j.linalg.factory.Nd4j
import org.deeplearning4j.nn.conf.{NeuralNetConfiguration, Updater}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.{CollectScoresIterationListener, ScoreIterationListener}
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.lossfunctions.LossFunctions

/**
  * Created by jose on 12/02/18.
  */
class BiLSTM(lambda:Double) {


  /* hard coded stuff until we get any good result */
  val innerLayerSize = 94
  val secondLayerSize = 47

  /* TODO hard coded parameters here! */
  val extraFeatSize = 10
  val vectorSize = 200 + extraFeatSize

  Nd4j.getMemoryManager().setAutoGcWindow(5000)

  val conf = new NeuralNetConfiguration.Builder()
    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
    .learningRate(0.22).rmsDecay(0.85).regularization(true)
    .l2(lambda).updater(Updater.RMSPROP)
    .seed(12345).list().pretrain(false)
    .layer(0, new GravesBidirectionalLSTM.Builder()
      .activation(Activation.TANH).nIn(vectorSize).nOut(innerLayerSize).weightInit(WeightInit.DISTRIBUTION)
      .dist(new UniformDistribution(-0.05, 0.05)).build())
    .layer(1, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
      .activation(Activation.SOFTMAX)
      .nIn(innerLayerSize).nOut(6).build())
    .pretrain(false).backprop(true).build()


  val multiLayerConf = new NeuralNetConfiguration.Builder()
    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
    .learningRate(0.22).rmsDecay(0.85).regularization(true)
    .l2(lambda).updater(Updater.RMSPROP)
    .seed(12345).list().pretrain(false)
    .layer(0, new GravesBidirectionalLSTM.Builder()
      .activation(Activation.TANH).nIn(vectorSize).nOut(innerLayerSize).weightInit(WeightInit.DISTRIBUTION)
      .dist(new UniformDistribution(-0.05, 0.05)).build())
    .layer(1, new GravesBidirectionalLSTM.Builder()
      .activation(Activation.TANH).nIn(innerLayerSize).nOut(secondLayerSize).weightInit(WeightInit.DISTRIBUTION)
      .dist(new UniformDistribution(-0.05, 0.05)).build())
    .layer(2, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
      .activation(Activation.SOFTMAX)
      .nIn(secondLayerSize).nOut(6).build())
    .pretrain(false).backprop(true).build()

  val model = new MultiLayerNetwork(multiLayerConf)
  model.init()
  model.setListeners(new ScoreIterationListener(1))
  model.setListeners(new CollectScoresIterationListener)

}
