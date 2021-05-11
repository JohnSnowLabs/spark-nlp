/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.johnsnowlabs.nlp.annotators.ner.crf

import com.johnsnowlabs.nlp._
import com.johnsnowlabs.tags.{FastTest, SlowTest}

import org.apache.spark.sql.{Dataset, Row, SparkSession}
import org.scalatest.FlatSpec

class NerCrfApproachTestSpec extends FlatSpec {
  val spark: SparkSession = SparkAccessor.spark

  val nerSentence: Dataset[Row] = DataBuilder.buildNerDataset(ContentProvider.nerCorpus)
  //  System.out.println(s"number of sentences in dataset ${nerSentence.count()}")

  // Dataset ready for NER tagger
  val nerInputDataset: Dataset[Row] = AnnotatorBuilder.withGlove(nerSentence)
  //  System.out.println(s"number of sentences in dataset ${nerInputDataset.count()}")
  val nerModel: NerCrfModel = AnnotatorBuilder.getNerCrfModel(nerSentence)

  "NerCrfApproach" should "be serializable and deserializable correctly" taggedAs SlowTest in {
    nerModel.write.overwrite.save("./test_crf_pipeline")
    val loadedNer = NerCrfModel.read.load("./test_crf_pipeline")

    assert(nerModel.model.getOrDefault.serialize == loadedNer.model.getOrDefault.serialize)
    assert(nerModel.dictionaryFeatures.getOrDefault == loadedNer.dictionaryFeatures.getOrDefault)
  }


  it should "have correct set of labels" taggedAs FastTest in {
    assert(nerModel.model.isSet)
    val metadata = nerModel.model.getOrDefault.metadata
    assert(metadata.labels.toSeq == Seq("@#Start", "PER", "O", "ORG", "LOC"))
  }


  it should "correctly store annotations" taggedAs FastTest in {
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


  it should "correctly tag sentences" taggedAs FastTest in {
    val tagged = nerModel.transform(nerInputDataset)
    val annotations = Annotation.collect(tagged, "ner").flatten

    val tags = annotations.map(a => a.result).toSeq
    assert(tags.toList == Seq("PER", "PER", "O", "O", "ORG", "LOC", "O"))
  }


  "NerCrfModel" should "correctly train using dataset from file" taggedAs SlowTest in {
    val tagged = AnnotatorBuilder.withNerCrfTagger(nerInputDataset)
    val annotations = Annotation.collect(tagged, "ner").flatten

    val tags = annotations.map(a => a.result).toSeq
    assert(tags.toList == Seq("PER", "PER", "O", "O", "ORG", "LOC", "O"))
  }


  it should "correctly handle entities param" taggedAs FastTest in {

    val restrictedModel = new NerCrfModel()
      .setEntities(Array("PER", "LOC"))
      .setModel(nerModel.model.getOrDefault)
      .setOutputCol(nerModel.getOutputCol)
      .setInputCols(nerModel.getInputCols)
      .setStorageRef("embeddings_ner_100")

    val tagged = restrictedModel.transform(nerInputDataset)
    val annotations = Annotation.collect(tagged, "ner").flatten
    val tags = annotations.map(a => a.result).toSeq

    assert(tags == Seq("PER", "PER", "LOC"))
  }

}
