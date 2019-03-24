package com.johnsnowlabs.nlp.annotators.spell.context

import com.github.liblevenshtein.transducer.{Candidate, ITransducer}
import com.johnsnowlabs.nlp.HasFeatures
import com.johnsnowlabs.nlp.annotators.spell.context.parser.SpecialClassParser
import com.johnsnowlabs.nlp.serialization.{TransducerFeature, TransducerSeqFeature}

trait HasTransducerFeatures extends HasFeatures {

  protected def set(feature: TransducerFeature, value: ITransducer[Candidate]): this.type = {feature.setValue(Some(value)); this}

  protected def set(feature: TransducerSeqFeature, value: Seq[SpecialClassParser]): this.type = {feature.setValue(Some(value)); this}

  protected def setDefault(feature: TransducerFeature, value: () => ITransducer[Candidate]): this.type = {feature.setFallback(Some(value)); this}

  protected def setDefault(feature: TransducerSeqFeature, value: () => Seq[SpecialClassParser]): this.type = {feature.setFallback(Some(value)); this}

  protected def get(feature: TransducerFeature): Option[ITransducer[Candidate]] = feature.get

  protected def get(feature: TransducerSeqFeature): Option[Seq[SpecialClassParser]] = feature.get

  protected def $$(feature: TransducerFeature): ITransducer[Candidate] = feature.getOrDefault

  protected def $$(feature: TransducerSeqFeature): Seq[SpecialClassParser] = feature.getOrDefault

}
