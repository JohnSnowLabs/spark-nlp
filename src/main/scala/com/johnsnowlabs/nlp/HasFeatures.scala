package com.johnsnowlabs.nlp

import com.johnsnowlabs.nlp.serialization._

import scala.collection.mutable.ArrayBuffer

trait HasFeatures {

  val features: ArrayBuffer[Feature[_, _, _]] = ArrayBuffer.empty

  protected def set[T](feature: ArrayFeature[T], value: Array[T]): this.type = {feature.setValue(Some(value)); this}

  protected def set[T](feature: SetFeature[T], value: Set[T]): this.type = {feature.setValue(Some(value)); this}

  protected def set[K, V](feature: MapFeature[K, V], value: Map[K, V]): this.type = {feature.setValue(Some(value)); this}

  protected def set[T](feature: StructFeature[T], value: T): this.type = {feature.setValue(Some(value)); this}

  protected def setDefault[T](feature: ArrayFeature[T], value: () => Array[T]): this.type = {feature.setFallback(Some(value)); this}

  protected def setDefault[T](feature: SetFeature[T], value: () => Set[T]): this.type = {feature.setFallback(Some(value)); this}

  protected def setDefault[K, V](feature: MapFeature[K, V], value: () => Map[K, V]): this.type = {feature.setFallback(Some(value)); this}

  protected def setDefault[T](feature: StructFeature[T], value: () => T): this.type = {feature.setFallback(Some(value)); this}

  protected def get[T](feature: ArrayFeature[T]): Option[Array[T]] = feature.get

  protected def get[T](feature: SetFeature[T]): Option[Set[T]] = feature.get

  protected def get[K, V](feature: MapFeature[K, V]): Option[Map[K, V]] = feature.get

  protected def get[T](feature: StructFeature[T]): Option[T] = feature.get

  protected def $$[T](feature: ArrayFeature[T]): Array[T] = feature.getOrDefault

  protected def $$[T](feature: SetFeature[T]): Set[T] = feature.getOrDefault

  protected def $$[K, V](feature: MapFeature[K, V]): Map[K, V] = feature.getOrDefault

  protected def $$[T](feature: StructFeature[T]): T = feature.getOrDefault

}
