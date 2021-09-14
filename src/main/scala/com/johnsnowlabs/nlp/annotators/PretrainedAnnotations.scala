/*
 * Copyright 2017-2021 John Snow Labs
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.Annotation
import com.johnsnowlabs.nlp.annotators.parser.dep.DependencyParserModel
import com.johnsnowlabs.nlp.annotators.parser.typdep.TypedDependencyParserModel
import com.johnsnowlabs.nlp.annotators.pos.perceptron.PerceptronModel

object PretrainedAnnotations {

  def getPos(annotations: Seq[Annotation], posModelCoordinates: Array[String]): Seq[Annotation] = {
    val pretrainedPosModel = posModelCoordinates.length match {
      case 2 =>
        PerceptronModel.pretrained(name = posModelCoordinates.head, lang = posModelCoordinates(1))
      case 3 =>
        PerceptronModel.pretrained(name = posModelCoordinates.head, lang = posModelCoordinates(1), remoteLoc = posModelCoordinates(2))
      case _ => PerceptronModel.pretrained()
    }
    val averagePerceptron = pretrainedPosModel.model.getOrDefault
    val posTagger = new PerceptronModel().setModel(averagePerceptron)
    posTagger.annotate(annotations)
  }

  def getDependencyParser(annotations: Seq[Annotation], dependencyParserModelCoordinates: Array[String]):
  Seq[Annotation] = {
    val dependencyMaker = dependencyParserModelCoordinates.length match {
      case 2 =>
        DependencyParserModel.pretrained(name = dependencyParserModelCoordinates.head,
          lang = dependencyParserModelCoordinates(1))
      case 3 =>
        DependencyParserModel.pretrained(name = dependencyParserModelCoordinates.head,
          lang = dependencyParserModelCoordinates(1), remoteLoc = dependencyParserModelCoordinates(2))
      case _ => DependencyParserModel.pretrained()
    }
    val dependencyParser = new DependencyParserModel().setPerceptron(dependencyMaker.perceptron.getOrDefault)
    val dependencyParserAnnotations = dependencyParser.annotate(annotations)
    dependencyParserAnnotations
  }

  def getTypedDependencyParser(annotations: Seq[Annotation],
                               typedDependencyParserModelCoordinates: Array[String]): Seq[Annotation] = {
    val pretrainedModel = typedDependencyParserModelCoordinates.length match {
      case 2 =>
        TypedDependencyParserModel.pretrained(name = typedDependencyParserModelCoordinates.head,
          lang = typedDependencyParserModelCoordinates(1))
      case 3 =>
        TypedDependencyParserModel.pretrained(name = typedDependencyParserModelCoordinates.head,
          lang = typedDependencyParserModelCoordinates(1), remoteLoc = typedDependencyParserModelCoordinates(2))
      case _ => TypedDependencyParserModel.pretrained()
    }
    val dependencyPipe = pretrainedModel.trainDependencyPipe.getOrDefault
    val trainOptions = pretrainedModel.trainOptions.getOrDefault
    val typedDependencyParser = new TypedDependencyParserModel()
      .setDependencyPipe(dependencyPipe)
      .setOptions(trainOptions)
      .setConllFormat("2009")
    typedDependencyParser.annotate(annotations)
  }

}