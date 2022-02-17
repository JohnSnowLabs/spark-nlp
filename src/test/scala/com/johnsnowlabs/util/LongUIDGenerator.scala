package com.johnsnowlabs.util

import com.johnsnowlabs.tags.FastTest
import org.apache.commons.math3.random.RandomDataGenerator
import org.scalatest.flatspec.AnyFlatSpec

class LongUIDGenerator extends AnyFlatSpec{

  "LongUIDGenerator" should "load default property values" taggedAs FastTest in {
    val leftLimit = 100000000000000L
    val rightLimit = 200000000000000L

        for(i <- 1 to 100) {
          println(new RandomDataGenerator().nextLong(leftLimit, rightLimit))
        }

//    val sheep = new Sheep("Stef")
//    println(java.io.ObjectStreamClass.lookup(sheep.getClass()).getSerialVersionUID())
  }
}


@SerialVersionUID(136630359430337L)
class Sheep(val name: String) extends Serializable {
  override def toString = name
  val greet: String = {
    s"Hello, $name"
  }
}

class UIDMappingManager{
  def getMapping(): Unit ={
    val mapping = Map[Long, String](
      112462048007662L -> "ClassifierDatasetEncoder",
      109232503554387L -> "NerDatasetEncoder",
      109139394247273L -> "TensorflowAlbert",
      119528022731187L -> "TensorflowAlbertClassification",
      127176388137330L -> "TensorflowBert",
      157443390643185L -> "TensorflowBertClassification",
      188830945871683L -> "TensorflowClassifier",
      151859637336068L -> "TensorflowDistilBert",
      177105061382704L -> "TensorflowDistilBertClassification",
      126173323244180L -> "TensorflowElmo",
      120274873769475L -> "TensorflowGPT2",
      182448962886290L -> "TensorflowLD",
      102775212797036L -> "TensorflowMarian",
      196684486821142L -> "TensorflowMultiClassifier",
      123505493299093L -> "TensorflowNer",
      108465285596706L -> "TensorflowRoBerta",
      113564387504819L -> "TensorflowRoBertaClassification",
      110621548029802L -> "TensorflowSentenceDetectorDL",
      189421277280130L -> "TensorflowSentiment",
      153107425878292L -> "TensorflowT5",
      174931581545407L -> "TensorflowUSE",
      186020885468857L -> "TensorflowWrapper",
      111350083808341L -> "TensorflowXlmRoberta",
      131219453547632L -> "TensorflowXlmRoBertaClassification",
      126804502508656L -> "TensorflowXlnet",
      199003965729496L -> "TensorflowXlnetClassification",
      148402594829421L -> "SentencePieceWrapper",
      114051250246689L -> "DateMatcherTranslator",
      170011195874373L -> "EntityRulerFeatures",
      131872618660943L -> "WritableAnnotatorComponent",
      107582264381823L -> "Perceptron",
      177023163447644L -> "Tagger",
      164285316273142L -> "DependencyMaker",
      169521053390383L -> "AveragedPerceptron",
      104108120972206L -> "TrainingPerceptronLegacy",
      139419481789513L -> "PragmaticScorer",
      112986025194248L -> "SentenceDetectorDLEncoder",
      111258924299110L -> "SerializableClass",
      195511164260220L -> "Feature",
      196155433597231L -> "RegexRule",
      105014511018936L -> "Database"
    )
  }
}