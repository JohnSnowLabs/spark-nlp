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

package com.johnsnowlabs.nlp.annotators.spell.context

import com.github.liblevenshtein.transducer.{Candidate, ITransducer}
import com.johnsnowlabs.nlp.HasFeatures
import com.johnsnowlabs.nlp.annotators.spell.context.parser.{SpecialClassParser, TransducerSeqFeature, VocabParser}
import com.johnsnowlabs.nlp.serialization.TransducerFeature

trait HasTransducerFeatures extends HasFeatures {

  protected def set(feature: TransducerFeature, value: VocabParser): this.type = {feature.setValue(Some(value)); this}

  protected def set(feature: TransducerSeqFeature, value: Seq[SpecialClassParser]): this.type = {feature.setValue(Some(value)); this}

  protected def setDefault(feature: TransducerFeature, value: () => VocabParser): this.type = {feature.setFallback(Some(value)); this}

  protected def setDefault(feature: TransducerSeqFeature, value: () => Seq[SpecialClassParser]): this.type = {feature.setFallback(Some(value)); this}

  protected def get(feature: TransducerFeature): Option[VocabParser] = feature.get

  protected def get(feature: TransducerSeqFeature): Option[Seq[SpecialClassParser]] = feature.get

  protected def $$(feature: TransducerFeature): VocabParser = feature.getOrDefault

  protected def $$(feature: TransducerSeqFeature): Seq[SpecialClassParser] = feature.getOrDefault

}
