package com.johnsnowlabs.nlp.annotators.ner.dl

import com.johnsnowlabs.nlp._
import com.johnsnowlabs.util.FileHelper
import org.scalatest.FlatSpec
import com.johnsnowlabs.nlp.training.CoNLL
import com.johnsnowlabs.nlp.util.io.ResourceHelper

class NerDLSpec extends FlatSpec {


  "NerDLApproach" should "correctly annotate" in {
    val nerSentence = DataBuilder.buildNerDataset(ContentProvider.nerCorpus)
    System.out.println(s"number of sentences in dataset ${nerSentence.count()}")

    // Dataset ready for NER tagger
    val nerInputDataset = AnnotatorBuilder.withGlove(nerSentence)
    System.out.println(s"number of sentences in dataset ${nerInputDataset.count()}")

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

  "NerDLApproach" should "correctly tag sentences" in {
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

  "NerDLModel" should "correctly train using dataset from file" in {
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

  "NerDLApproach" should "be serializable and deserializable correctly" in {

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

  "NerDLApproach" should "correct search for suitable graphs" in {
    val smallGraphFile = NerDLApproach.searchForSuitableGraph(10, 100, 120)
    assert(smallGraphFile.endsWith("blstm_10_100_128_120.pb"))

    val bigGraphFile = NerDLApproach.searchForSuitableGraph(25, 300, 120)
    assert(bigGraphFile.endsWith("blstm_30_300_128_600.pb"))

    assertThrows[IllegalArgumentException](NerDLApproach.searchForSuitableGraph(31, 101, 100))
    assertThrows[IllegalArgumentException](NerDLApproach.searchForSuitableGraph(20, 768, 601))
    assertThrows[IllegalArgumentException](NerDLApproach.searchForSuitableGraph(31, 100, 101))
  }

  "NerDL Approach" should "validate against part of the training dataset" in {

    val conll = CoNLL()
    val training_data = conll.readDataset(ResourceHelper.spark, "src/test/resources/conll2003/eng.testa")
    val test_data = conll.readDataset(ResourceHelper.spark, "src/test/resources/conll2003/eng.testb")

    val embeddings = AnnotatorBuilder.getGLoveEmbeddings(training_data.toDF())

    val trainData = embeddings.transform(training_data)
    val testData = embeddings.transform(test_data)
    testData.write.mode("overwrite").parquet("./tmp_conll_validate")

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
      .setValidationSplit(0.1f)
      .setEvaluationLogExtended(true)
      .setTestDataset("./tmp_conll_validate/")
      .setGraphFolder("src/test/resources/graph/")
      .fit(trainData)

    ner.write.overwrite()save("./tmp_ner_dl_tf115")
  }

  "NerDLModel" should "successfully download pretrained and predict" ignore {

    val conll = CoNLL()
    val test_data = conll.readDataset(ResourceHelper.spark, "src/test/resources/conll2003/eng.testb")

    val embeddings = AnnotatorBuilder.getGLoveEmbeddings(test_data.toDF())

    val testData = embeddings.transform(test_data)

    val nerModel = NerDLModel.load("./tmp_ner_dl_tf115")
      .setInputCols("sentence", "token", "embeddings")
      .setOutputCol("ner")
      .transform(testData)

    nerModel.show()
  }

}

