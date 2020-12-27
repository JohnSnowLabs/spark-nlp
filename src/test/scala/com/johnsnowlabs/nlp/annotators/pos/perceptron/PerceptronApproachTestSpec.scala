package com.johnsnowlabs.nlp.annotators.pos.perceptron

import com.johnsnowlabs.nlp.annotators.Tokenizer
import com.johnsnowlabs.nlp.annotators.common.Sentence
import com.johnsnowlabs.nlp.training.POS
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.nlp.{ContentProvider, DataBuilder, SparkNLP}
import org.apache.spark.sql.DataFrame
import org.scalatest._
import org.spark_project.dmg.pmml.False


class PerceptronApproachTestSpec extends FlatSpec with PerceptronApproachBehaviors {

  "an isolated perceptron tagger" should behave like isolatedPerceptronTraining(
    "src/test/resources/anc-pos-corpus-small/test-training.txt"
  )

  val trainingPerceptronDF: DataFrame = POS().readDataset(ResourceHelper.spark, "src/test/resources/anc-pos-corpus-small/test-training.txt", "|", "tags")

  val trainedTagger: PerceptronModel =
    new PerceptronApproach()
      .setPosColumn("tags")
      .setNIterations(3)
      .fit(trainingPerceptronDF)

  // Works with high iterations only
  val targetSentencesFromWsjResult = Array("NNP", "NNP", "CD", "JJ", "NNP", "CD", "JJ", "NNP", "CD", "JJ", "NNP", "CD",
    "IN", "DT", "IN", ".", "NN", ".", "NN", ".", "DT", "JJ", "NNP", "CD", "JJ", "NNP", "CD", "NNP", ",", "CD", ".",
    "JJ", "NNP", ".")

  val tokenizedSentenceFromWsj = {
    var length = 0
    val sentences = ContentProvider.targetSentencesFromWsj.map { text =>
      val sentence = Sentence(text, length, length + text.length - 1, 0)
      length += text.length + 1
      sentence
    }
    new Tokenizer().fit(trainingPerceptronDF).tag(sentences).toArray
  }


  "an isolated perceptron tagger" should behave like isolatedPerceptronTagging(
    trainedTagger,
    tokenizedSentenceFromWsj
  )

  "an isolated perceptron tagger" should behave like isolatedPerceptronTagCheck(
    new PerceptronApproach()
      .setPosColumn("tags")
      .setNIterations(3)
      .fit(POS().readDataset(ResourceHelper.spark, "src/test/resources/anc-pos-corpus-small/test-training.txt", "|", "tags")),
    tokenizedSentenceFromWsj,
    targetSentencesFromWsjResult
  )

  "a spark based pos detector" should behave like sparkBasedPOSTagger(
    DataBuilder.basicDataBuild(ContentProvider.sbdTestParagraph)
  )

  "a spark trained pos detector" should behave like sparkBasedPOSTraining(
    path="src/test/resources/anc-pos-corpus-small/test-training.txt",
    test="src/test/resources/test.txt"
  )
  "A Perceptron Tagger" should "Reload weights onto new object and have same preds" in {
    print("SPARK VERSION : ", SparkNLP.start().sparkContext.version)
    import com.johnsnowlabs.nlp._
    import com.johnsnowlabs.nlp.annotators._
    import org.apache.spark.ml.Pipeline

    val spark = SparkNLP.start()
    import spark.implicits._

    val df = List(
      "Hello, this is an example sentence",
      "And this is a second sentence").toDF("text")

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val token_assembler = new Tokenizer()
      .setInputCols(Array("document"))
      .setOutputCol("token")

    val perceptronTagger = PerceptronModel.pretrained()


    val pipeline = new Pipeline().
      setStages(Array(
        documentAssembler,
        token_assembler,
        perceptronTagger
      ))


    val preDf = pipeline.fit(df).transform(df)
    print("PRE DF ")
    preDf.select("pos").show(false)


 //    tags: Array[String],
//    taggedWordBook: Map[String, String],
//    featuresWeight: Map[String, Map[String, Double]]
    val tags = perceptronTagger.getModel.getTags
    val taggedWordBook = perceptronTagger.getModel.getTaggedBook
    val featuresWeight = perceptronTagger.getModel.getWeights

    val finalModel = AveragedPerceptron(tags, taggedWordBook, featuresWeight)
    val perM = new  PerceptronModel().setModel(finalModel)
    perM.setInputCols("document","token").setOutputCol("pos")

    val pipeline_post = new Pipeline().
      setStages(Array(
        documentAssembler,
        token_assembler,
        perM
      ))

    val postDf = pipeline_post.fit(df).transform(df)
    print("POST DF ")
    postDf.select("pos").show(false)


  }




  "A Perceptron Tagger" should "write its weights to TXT's" in {
    print("SPARK VERSION : ", SparkNLP.start().sparkContext.version)
    import com.johnsnowlabs.nlp._
    import com.johnsnowlabs.nlp.annotators._
    import org.apache.spark.ml.Pipeline

    val spark = SparkNLP.start()
    import spark.implicits._

    val df = List(
      "Hello, this is an example sentence",
      "And this is a second sentence").toDF("text")

//    val documentAssembler = new DocumentAssembler()
//      .setInputCol("text")
//      .setOutputCol("document")
//
//    val token_assembler = new Tokenizer()
//      .setInputCols(Array("document"))
//      .setOutputCol("token")

    val perceptronTagger = PerceptronModel.pretrained()

//
//    val pipeline = new Pipeline().
//      setStages(Array(
//        documentAssembler,
//        token_assembler,
//        perceptronTagger
//      ))


//    val preDf = pipeline.fit(df).transform(df)
//    print("PRE DF ")
//    preDf.select("pos").show(false)

//
//    //    tags: Array[String],
//    //    taggedWordBook: Map[String, String],
//    //    featuresWeight: Map[String, Map[String, Double]]
    val tags = perceptronTagger.getModel.getTags
    val taggedWordBook = perceptronTagger.getModel.getTaggedBook
    val featuresWeight = perceptronTagger.getModel.getWeights
//
//    val finalModel = AveragedPerceptron(tags, taggedWordBook, featuresWeight)
//    val perM = new  PerceptronModel().setModel(finalModel)
//    perM.setInputCols("document","token").setOutputCol("pos")
//
//    val pipeline_post = new Pipeline().
//      setStages(Array(
//        documentAssembler,
//        token_assembler,
//        perM
//      ))
//
//    val postDf = pipeline_post.fit(df).transform(df)
//    print("POST DF ")
//    postDf.select("pos").show(false)

    import com.johnsnowlabs.nlp.base._
    import java.io.PrintWriter
    import java.io.File
    import java.io.PrintWriter
    import scala.io.Source



    def tagsToTxt(tags:Array[String]) : Unit = {
      //Write POS tag to txt    tagToTxt(perceptronTagger.getModel.getTags)
      // Save Tags seperated by ~
      val writer = new PrintWriter(new File("./pos.txt"))
      writer.write(tags.mkString("~"))
      writer.close()


    }

    def taggedWordBookToTxt(taggedWordBook:Map[String, String]) : Unit = {
      //Write POS tag to txt    tagToTxt(perceptronTagger.getModel.getTags)
      // Save key/value seperated by ~! followed by \n
      val writer = new PrintWriter(new File("./taggedWordBook.txt"))

      taggedWordBook.map{x =>
        // value -> key
        // value~!key
        writer.write(x._1 + "~!" + x._2 + "\n")
      }
      writer.close()


    }
    def txtToTaggedWordBook (txtPath : String) : Map[String,String] = {
      // read TaggedWordBook from file
      val src = Source.fromFile(txtPath)

      val taggedWordBook = src.
        getLines
            .map( l => l.split("~!"))
            .map{case Array(key, value) => key -> value}
            .toMap

      src.close()

      taggedWordBook

    }


    def txtToTags(tagsPath : String):Array[String] = {
      // Load tags and split on ~
      val src = Source.fromFile(tagsPath)
      val res = src.mkString("").split("~")
      src.close()
      res
    }


    def featuresWeightToTxt(featuresWeight: Map[String, Map[String, Double]]) :Unit = {
      // First Level map always has just 1 Element, which is a Map[String,Double]
      // The inner map can have multiple elements, but there are no deeper maps
      // So we write one line for each outer map.
      // 1 Map = KEY->INNERMAP
      // 2. INNERMAP->[STR->DOUBLE, STR->DOUBLE, ..]
      // so one element is
      // Map = KEY->[STR->DOUBLE, STR->DOUBLE, ..]
      // use ~ instead of ->
      val writer = new PrintWriter(new File("./featuresWeight.txt"))

      featuresWeight.map{ l1Map =>
        val l1MapKey = l1Map._1

        // All key/val pairs of l1MapValues are list(key ~! value)
        val l1MapValues = l1Map._2.map{ l2Map =>
          val l2MapKey = l2Map._1
          val l2MapValue = l2Map._2
          l2MapKey + "<M2ARROW>" + l2MapValue.toString
        }
        // l1Mapkey ~@ l1MapValue
        // L1MapValue = l1mv1~?l1mv2~?1mv2~?...
        // l1mv1 = l2mk1~!l2mv1 ~? l2mk2~!l2mv2 ~?
        writer.write(l1MapKey + "<M1ARROW>" + l1MapValues.mkString("<ENDM1>") + "\n")
      }
      writer.close()

    }

    def txtToFeaturesWeight (txtPath : String) : Map[String, Map[String, Double]] = {
      // read TaggedWordBook from file
      val src = Source.fromFile(txtPath)

      val taggedWordBook = src.
        getLines
        .map{ l => // l1MapKey~@l1MapValue
          val l1m = l.split("<M1ARROW>")
          val l1MapKey = l1m(0)
          val l2MapVal = l1m(1).split("<ENDM1>").map{ l2m => // ~? seperates elements of L1MapValue which are also maps
                    val l2Map = l2m.split("<M2ARROW>") // l2 Maps look like  l2v ~! l2k
                    val l2MapKey = l2Map(0)
                    val l2MapVal = l2Map(1)

                    Tuple2(l2MapKey, l2MapVal.toDouble)
          }.toMap//[String,Double]// Map[String,Double]
          Tuple2(l1MapKey,l2MapVal)
        }.toMap  //Map[String, Map[String, Double]]
//        .map{case Array(key, value) => key -> value}

      src.close()

      taggedWordBook

    }



    //write and load tags
    tagsToTxt(tags)
    val tagsBack = txtToTags("./pos.txt")

    //write and load WordBook
    taggedWordBookToTxt(taggedWordBook)
    val taggedWordBookBack = txtToTaggedWordBook("./taggedWordBook.txt")

    //write and load featuresWeight
    featuresWeightToTxt(featuresWeight)
    val featuresWeightBack = txtToFeaturesWeight("./featuresWeight.txt")



    // CHECK ALL RELOADED OBJECT FOR EQUALITY

    //check features weight
    val equalCheckWeights = featuresWeight.keys.map{ k1 =>
      featuresWeight(k1).keys.map{k2=>
        val res = featuresWeight(k1)(k2) == featuresWeightBack(k1)(k2)
        if (res == false) { print("BAD BOI")}
        res
      }
    }
    print(equalCheckWeights)

  //check taggedWordBook map
  val equalCheckTaggedBook = taggedWordBook.keys.map { key =>
    val res = taggedWordBook(key) == taggedWordBookBack(key)
    if (res == false) {print("BAD BOI")}
    res
  }
    print(equalCheckTaggedBook)

  // check tags
    val equalCheckTags = tags.zip(tagsBack).map{ e =>
      val res = e._1 == e._2
      if (res == false) {print("BAD BOI")}
      res
  }

  print(equalCheckTags)

  }



//BAD BOIchecking K1=i+1 word artemis K2=JJ

























  "A Perceptron Tagger" should "be readable and writable" in {
    val trainingPerceptronDF = POS().readDataset(ResourceHelper.spark, "src/test/resources/anc-pos-corpus-small/", "|", "tags")

    val perceptronTagger = new PerceptronApproach()
      .setPosColumn("tags")
      .setNIterations(1)
      .fit(trainingPerceptronDF)
    val path = "./test-output-tmp/perceptrontagger"
    try {
      perceptronTagger.write.overwrite.save(path)
      val perceptronTaggerRead = PerceptronModel.read.load(path)
      assert(perceptronTagger.tag(perceptronTagger.getModel, tokenizedSentenceFromWsj).head.tags.head ==
        perceptronTaggerRead.tag(perceptronTagger.getModel, tokenizedSentenceFromWsj).head.tags.head)
    } catch {
      case _: java.io.IOException => succeed
    }
  }
  /*
  * Testing POS() class
  * Making sure it only extracts good token_labels
  *
  * */
  val originalFrenchLabels: List[(String, Int)] = List(
    ("DET",9), ("ADP",12), ("AUX",2),
    ("CCONJ",2), ("NOUN",12), ("ADJ",3),
    ("NUM",9), ("PRON",1),
    ("PROPN",2), ("PUNCT",10),
    ("SYM",2), ("VERB",2), ("X",2)
  )

  "French readDataset in POS() class" should behave like readDatasetInPOS(
    path="src/test/resources/universal-dependency/UD_French-GSD/UD_French-test.txt",
    originalFrenchLabels
  )

  //  /*
  //  * Test ReouceHelper to convert token|tag to DataFrame with POS annotation as a column
  //  *
  //  * */
  //  val posTrainingDataFrame: DataFrame = ResourceHelper.annotateTokenTagTextFiles(path = "src/test/resources/anc-pos-corpus-small", delimiter = "|")
  //  posTrainingDataFrame.show(1,truncate = false)
}