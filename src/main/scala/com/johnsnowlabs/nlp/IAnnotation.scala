/*
 * Copyright 2017-2022 John Snow Labs
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

package com.johnsnowlabs.nlp

/** IAnnotation trait is used to abstract the annotator's output for each NLP tasks available in
  * Spark NLP.
  *
  * Currently Spark NLP supports three types of outputs:
  *   - Text Output: [[com.johnsnowlabs.nlp.Annotation]]
  *   - Image Output: [[com.johnsnowlabs.nlp.AnnotationImage]]
  *   - Audio Output: [[com.johnsnowlabs.nlp.AnnotationAudio]]
  *
  * LightPipeline models in Java/Scala returns an IAnnotation collection. All of these outputs are
  * structs with the required data types to represent Text, Image and Audio.
  *
  * If one wants to access the data as Annotation, AnnotationImage or AnnotationAudio, one just
  * needs casting to the desired output.
  * ==Example==
  * {{{
  * import com.johnsnowlabs.nlp.annotators.cv.ViTForImageClassification
  * import org.apache.spark.ml.Pipeline
  * import com.johnsnowlabs.nlp.annotators.Tokenizer
  * import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector
  * import com.johnsnowlabs.nlp.ImageAssembler
  * import com.johnsnowlabs.nlp.LightPipeline
  * import com.johnsnowlabs.util.PipelineModels
  *
  * val imageDf = spark.read
  *  .format("image")
  *  .option("dropInvalid", value = true)
  *  .load("./images")
  *
  * val imageAssembler = new ImageAssembler()
  *   .setInputCol("image")
  *   .setOutputCol("image_assembler")
  *
  * val imageClassifier = ViTForImageClassification
  * .pretrained()
  * .setInputCols("image_assembler")
  * .setOutputCol("class")
  *
  * val pipeline: Pipeline = new Pipeline().setStages(Array(imageAssembler, imageClassifier))
  *
  * val vitModel = pipeline.fit(imageDf)
  * val lightPipeline = new LightPipeline(vitModel)
  * val predictions = lightPipeline.fullAnnotate("./images/hen.JPEG")
  *
  * val result = predictions.flatMap(prediction => prediction._2.map {
  *     case annotationText: Annotation =>
  *       annotationText
  *     case annotationImage: AnnotationImage =>
  *       annotationImage
  * })
  *
  * }}}
  *
  * @param annotatorType
  */

trait IAnnotation {

  def annotatorType: String

}
