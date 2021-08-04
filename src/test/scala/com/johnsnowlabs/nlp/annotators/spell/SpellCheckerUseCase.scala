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

package com.johnsnowlabs.nlp.annotators.spell

import com.johnsnowlabs.nlp.annotator.RecursiveTokenizer
import com.johnsnowlabs.nlp.{DocumentAssembler, SparkAccessor}
import com.johnsnowlabs.nlp.annotators.spell.context.{ContextSpellCheckerApproach, ContextSpellCheckerModel}
import org.apache.spark.ml.Pipeline

object SpellCheckerUseCase extends App {

  // ==================================
  // Getting and cleaning all the data
  // ==================================

  // Let's use the Paisa corpus
  // https://clarin.eurac.edu/repository/xmlui/bitstream/handle/20.500.12124/3/paisa.raw.utf8.gz
  // (update with the path you downloaded the file to)
  val paisaCorpusPath = "/tmp/paisa/paisa.raw.utf8"

  // do some brief DS exploration, and preparation to get clean text
  val df = SparkAccessor.spark.read.text(paisaCorpusPath)

  val dataset = df.filter(!df("value").contains("</text")).
    filter(!df("value").contains("<text")).
    filter(!df("value").startsWith("#")).
    limit(10000)

  dataset.show(truncate = false)

  val names = List("Achille", "Achillea", "Achilleo", "Achillina", "Achiropita", "Acilia", "Acilio", "Acquisto",
    "Acrisio", "Ada", "Adalberta", "Adalberto", "Adalciso", "Adalgerio", "Adalgisa")

  import scala.collection.JavaConverters._

  val javaNames = new java.util.ArrayList[String](names.asJava)

  // ==================================
  // all the pipeline & training
  // ==================================
  val assembler = new DocumentAssembler()
    .setInputCol("value")
    .setOutputCol("document")

  val tokenizer = new RecursiveTokenizer()
    .setInputCols("document")
    .setOutputCol("token")
    .setPrefixes(Array("\"", "“", "(", "[", "\n", ".", "l’", "dell’", "nell’", "sull’", "all’", "d’", "un’"))
    .setSuffixes(Array("\"", "”", ".", ",", "?", ")", "]", "!", ";", ":"))

  val spellCheckerModel = new ContextSpellCheckerApproach()
    .setInputCols("token")
    .setOutputCol("checked")
    .addVocabClass("_NAME_", javaNames)
    .setLanguageModelClasses(1650)
    .setWordMaxDistance(3)
    .setEpochs(2)

  val pipelineTok = new Pipeline().setStages(Array(assembler, tokenizer)).fit(dataset)
  val tokenized = pipelineTok.transform(dataset)
  spellCheckerModel.fit(tokenized).write.save("./tmp_contextSpellCheckerModel")
  val spellChecker = ContextSpellCheckerModel.load("./tmp_contextSpellCheckerModel")

}
