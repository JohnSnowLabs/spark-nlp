package com.johnsnowlabs.nlp.eval

import com.johnsnowlabs.nlp.annotator.{NerCrfApproach, WordEmbeddings}
import com.johnsnowlabs.nlp.embeddings.WordEmbeddingsFormat
import org.scalatest.FlatSpec

class NerCrfEvalTesSpec extends FlatSpec {

  "A NER CRF Evaluation" should "display accuracy results" in {
    val trainFile = "./eng.train.small"
    val testFile = "./eng.testa"
    val format = ""
    val modelPath = "./ner_crf"

    val glove = new WordEmbeddings()
      .setInputCols("sentence", "token")
      .setOutputCol("glove")
      .setEmbeddingsSource("./glove.6B.100d.txt ",
        100, WordEmbeddingsFormat.TEXT)
      .setCaseSensitive(true)

    val nerTagger = new NerCrfApproach()
      .setInputCols(Array("sentence", "token","pos", "glove"))
      .setLabelColumn("label")
      .setOutputCol("ner")
      .setMaxEpochs(10)
      .setRandomSeed(0)
      .setVerbose(2)

    val nerCrfEvaluation = new NerCrfEvaluation(testFile, format)
    nerCrfEvaluation.computeAccuracyAnnotator(modelPath, trainFile, nerTagger, glove)

  }

}
