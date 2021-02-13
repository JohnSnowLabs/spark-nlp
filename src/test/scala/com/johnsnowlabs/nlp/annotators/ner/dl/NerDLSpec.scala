package com.johnsnowlabs.nlp.annotators.ner.dl

import com.johnsnowlabs.nlp._
import com.johnsnowlabs.nlp.training.CoNLL
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.tags.{FastTest, SlowTest}
import com.johnsnowlabs.util.FileHelper
import org.scalatest.FlatSpec

class NerDLSpec extends FlatSpec {


  "NerDLApproach" should "correctly annotate" taggedAs SlowTest in {
    val nerSentence = DataBuilder.buildNerDataset(ContentProvider.nerCorpus)
    //    System.out.println(s"number of sentences in dataset ${nerSentence.count()}")

    // Dataset ready for NER tagger
    val nerInputDataset = AnnotatorBuilder.withGlove(nerSentence)
    //    System.out.println(s"number of sentences in dataset ${nerInputDataset.count()}")

    val nerModel = AnnotatorBuilder.getNerDLModel(nerSentence)


    val tagged = nerModel.transform(nerInputDataset)
    val annotations = Annotation.collect(tagged, "ner").flatten.toSeq
    val labels = Annotation.collect(tagged, "label").flatten.toSeq

    assert(annotations.length == labels.length)
    for ((annotation, label) <- annotations.zip(labels)) {
      assert(annotation.begin == label.begin)
      assert(annotation.end == label.end)
      assert(annotation.annotatorType == AnnotatorType.NAMED_ENTITY)
      assert(annotation.result == label.result)
      assert(annotation.metadata.contains("word"))
    }
  }

  "NerDLApproach" should "correctly tag sentences" taggedAs SlowTest in {
    val nerSentence = DataBuilder.buildNerDataset(ContentProvider.nerCorpus)
    System.out.println(s"number of sentences in dataset ${nerSentence.count()}")

    // Dataset ready for NER tagger
    val nerInputDataset = AnnotatorBuilder.withGlove(nerSentence)
    System.out.println(s"number of sentences in dataset ${nerInputDataset.count()}")

    val nerModel = AnnotatorBuilder.getNerDLModel(nerSentence)

    val tagged = nerModel.transform(nerInputDataset)
    val annotations = Annotation.collect(tagged, "ner").flatten

    val tags = annotations.map(a => a.result).toSeq
    assert(tags.toList == Seq("PER", "PER", "O", "O", "ORG", "LOC", "O"))
  }

  "NerDLModel" should "correctly train using dataset from file" taggedAs SlowTest in {
    val spark = SparkAccessor.spark
    val nerSentence = DataBuilder.buildNerDataset(ContentProvider.nerCorpus)
    System.out.println(s"number of sentences in dataset ${nerSentence.count()}")

    // Dataset ready for NER tagger
    val nerInputDataset = AnnotatorBuilder.withGlove(nerSentence)
    System.out.println(s"number of sentences in dataset ${nerInputDataset.count()}")

    val tagged = AnnotatorBuilder.withNerDLTagger(nerInputDataset)
    val annotations = Annotation.collect(tagged, "ner").flatten

    val tags = annotations.map(a => a.result).toSeq
    assert(tags.toList == Seq("PER", "PER", "O", "O", "ORG", "LOC", "O"))
  }

  "NerDLApproach" should "be serializable and deserializable correctly" taggedAs SlowTest in {

    val nerSentence = DataBuilder.buildNerDataset(ContentProvider.nerCorpus)
    System.out.println(s"number of sentences in dataset ${nerSentence.count()}")

    // Dataset ready for NER tagger
    val nerInputDataset = AnnotatorBuilder.withGlove(nerSentence)
    System.out.println(s"number of sentences in dataset ${nerInputDataset.count()}")

    val nerModel = AnnotatorBuilder.getNerDLModel(nerSentence)


    nerModel.write.overwrite.save("./test_ner_dl")
    val loadedNer = NerDLModel.read.load("./test_ner_dl")
    FileHelper.delete("./test_ner_dl")

    // Test that params of loaded model are the same
    assert(loadedNer.datasetParams.getOrDefault == nerModel.datasetParams.getOrDefault)

    // Test that loaded model do the same predictions
    val tokenized = AnnotatorBuilder.withTokenizer(nerInputDataset)
    val tagged = loadedNer.transform(tokenized)
    val annotations = Annotation.collect(tagged, "ner").flatten

    val tags = annotations.map(a => a.result).toSeq
    assert(tags.toList == Seq("PER", "PER", "O", "O", "ORG", "LOC", "O"))
  }

  "NerDLApproach" should "correct search for suitable graphs" taggedAs FastTest in {
    val smallGraphFile = NerDLApproach.searchForSuitableGraph(10, 100, 120)
    assert(smallGraphFile.endsWith("blstm_10_100_128_120.pb"))

    val bigGraphFile = NerDLApproach.searchForSuitableGraph(25, 300, 120)
    assert(bigGraphFile.endsWith("blstm_38_300_128_200.pb"))

    assertThrows[IllegalArgumentException](NerDLApproach.searchForSuitableGraph(31, 101, 100))
    assertThrows[IllegalArgumentException](NerDLApproach.searchForSuitableGraph(50, 300, 601))
    assertThrows[IllegalArgumentException](NerDLApproach.searchForSuitableGraph(40, 512, 101))
  }

  "NerDLApproach" should "validate against part of the training dataset" taggedAs SlowTest in {

    val conll = CoNLL()
    val training_data = conll.readDataset(ResourceHelper.spark, "src/test/resources/ner-corpus/test_ner_dataset.txt")

    val embeddings = AnnotatorBuilder.getGLoveEmbeddings(training_data.toDF())

    val trainData = embeddings.transform(training_data)

    val ner = new NerDLApproach()
      .setInputCols("sentence", "token", "embeddings")
      .setOutputCol("ner")
      .setLabelColumn("label")
      .setOutputCol("ner")
      .setLr(1e-1f) //0.1
      .setPo(5e-3f) //0.005
      .setDropout(5e-1f) //0.5
      .setMaxEpochs(1)
      .setRandomSeed(0)
      .setVerbose(0)
      .setEvaluationLogExtended(true)
      .setGraphFolder("src/test/resources/graph/")
      .fit(trainData)

    ner.write.overwrite()save("./tmp_ner_dl_tf115")
  }

  "NerDLModel" should "successfully load saved model" taggedAs SlowTest in {

    val conll = CoNLL()
    val test_data = conll.readDataset(ResourceHelper.spark, "src/test/resources/conll2003/eng.testb")

    val embeddings = AnnotatorBuilder.getGLoveEmbeddings(test_data.toDF())

    val testData = embeddings.transform(test_data)

    val nerModel = NerDLModel.load("./tmp_ner_dl_tf115")
      .setInputCols("sentence", "token", "embeddings")
      .setOutputCol("ner")
      .transform(testData)

  }

  "NerDLModel" should "successfully download a pretrained model" taggedAs FastTest in {

    val nerModel = NerDLModel.pretrained()
      .setInputCols("sentence", "token", "embeddings")
      .setOutputCol("ner")

    nerModel.getClasses.foreach(x=>println(x))

  }

}

