#  Copyright 2017-2025 John Snow Labs
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
from pyspark.ml import Transformer
from pyspark.ml.param.shared import *
from pyspark.sql import DataFrame
from typing import Any

# Custom converter for string-to-string dictionaries
def toStringDict(value):
    if not isinstance(value, dict):
        raise TypeError("Expected a dictionary of strings.")
    return {str(k): str(v) for k, v in value.items()}

class DataFrameOptimizer(Transformer):
    """
    Optimizes a Spark DataFrame by repartitioning, optionally caching, and persisting it to disk.

    This transformer is intended to improve performance for Spark NLP pipelines or when preparing
    data for export. It allows partition tuning via `numPartitions` directly, or indirectly using
    `executorCores` and `numWorkers`. The DataFrame can also be persisted in a specified format
    (`csv`, `json`, or `parquet`) with additional writer options.

    Parameters
    ----------
    executorCores : int, optional
        Number of cores per Spark executor (used to compute number of partitions if `numPartitions` is not set).

    numWorkers : int, optional
        Number of executor nodes (used to compute number of partitions if `numPartitions` is not set).

    numPartitions : int, optional
        Target number of partitions for the DataFrame (overrides calculation via cores × workers).

    doCache : bool, default False
        Whether to cache the DataFrame after repartitioning.

    persistPath : str, optional
        Path to save the DataFrame output (if persistence is enabled).

    persistFormat : str, optional
        Format to persist the DataFrame in: one of `'csv'`, `'json'`, or `'parquet'`.

    outputOptions : dict, optional
        Dictionary of options for the DataFrameWriter (e.g., `{"compression": "snappy"}` for parquet).

    Examples
    --------
    >>> optimizer = DataFrameOptimizer() \\
    ...     .setExecutorCores(4) \\
    ...     .setNumWorkers(5) \\
    ...     .setDoCache(True) \\
    ...     .setPersistPath("/tmp/out") \\
    ...     .setPersistFormat("parquet") \\
    ...     .setOutputOptions({"compression": "snappy"})

    >>> optimized_df = optimizer.transform(input_df)

    Notes
    -----
    - You must specify either `numPartitions`, or both `executorCores` and `numWorkers`.
    - Schema is preserved; no columns are modified or removed.
    """

    executorCores = Param(
        Params._dummy(),
        "executorCores",
        "Number of cores per executor",
        typeConverter = TypeConverters.toInt
    )
    numWorkers = Param(
        Params._dummy(),
        "numWorkers",
        "Number of Spark workers",
        typeConverter = TypeConverters.toInt
    )
    numPartitions = Param(
        Params._dummy(),
        "numPartitions",
        "Total number of partitions (overrides executorCores * numWorkers)",
        typeConverter = TypeConverters.toInt
    )
    doCache = Param(
        Params._dummy(),
        "doCache",
        "Whether to cache the DataFrame",
        typeConverter = TypeConverters.toBoolean
    )

    persistPath = Param(
        Params._dummy(),
        "persistPath",
        "Optional path to persist the DataFrame",
        typeConverter = TypeConverters.toString
    )
    persistFormat = Param(
        Params._dummy(),
        "persistFormat",
        "Format to persist: parquet, json, csv",
        typeConverter = TypeConverters.toString
    )

    outputOptions = Param(
        Params._dummy(),
        "outputOptions",
        "Additional writer options",
        typeConverter=toStringDict
    )

    def __init__(self):
        super().__init__()
        self._setDefault(
            doCache=False,
            persistFormat="none",
            numPartitions=1,
            executorCores=1,
            numWorkers=1
        )

    # Parameter setters
    def setExecutorCores(self, value: int):
        """Set the number of executor cores."""
        return self._set(executorCores=value)

    def setNumWorkers(self, value: int):
        """Set the number of Spark workers."""
        return self._set(numWorkers=value)

    def setNumPartitions(self, value: int):
        """Set the total number of partitions (overrides cores * workers)."""
        return self._set(numPartitions=value)

    def setDoCache(self, value: bool):
        """Set whether to cache the DataFrame."""
        return self._set(doCache=value)

    def setPersistPath(self, value: str):
        """Set the path where the DataFrame should be persisted."""
        return self._set(persistPath=value)

    def setPersistFormat(self, value: str):
        """Set the format to persist the DataFrame (parquet, json, csv)."""
        return self._set(persistFormat=value)

    def setOutputOptions(self, value: dict):
        """Set additional writer options (e.g. for csv headers)."""
        return self._set(outputOptions=value)

    # Optional bulk setter
    def setParams(self, **kwargs: Any):
        for param, value in kwargs.items():
            self._set(**{param: value})
        return self

    def _transform(self, dataset: DataFrame) -> DataFrame:
        self._validate_params()
        part_count = self.getOrDefault(self.numPartitions)
        cores = self.getOrDefault(self.executorCores)
        workers = self.getOrDefault(self.numWorkers)
        if cores is None or workers is None:
            raise ValueError("Provide either numPartitions or both executorCores and numWorkers")
        if part_count == 1:
            part_count = cores * workers

        optimized_df = dataset.repartition(part_count)

        if self.getOrDefault(self.doCache):
            optimized_df = optimized_df.cache()

        format = self.getOrDefault(self.persistFormat).lower()
        if format != "none":
            path = self.getOrDefault(self.persistPath)
            if not path:
                raise ValueError("persistPath must be set when persistFormat is not 'none'")
            writer = optimized_df.write.mode("overwrite")
            if self.isDefined(self.outputOptions):
                writer = writer.options(**self.getOrDefault(self.outputOptions))
            if format == "parquet":
                writer.parquet(path)
            elif format == "json":
                writer.json(path)
            elif format == "csv":
                writer.csv(path)
            else:
                raise ValueError(f"Unsupported format: {format}")

        return optimized_df

    def _validate_params(self):
        if self.isDefined(self.executorCores):
            val = self.getOrDefault(self.executorCores)
            if val <= 0:
                raise ValueError("executorCores must be > 0")

        if self.isDefined(self.numWorkers):
            val = self.getOrDefault(self.numWorkers)
            if val <= 0:
                raise ValueError("numWorkers must be > 0")

        if self.isDefined(self.numPartitions):
            val = self.getOrDefault(self.numPartitions)
            if val <= 0:
                raise ValueError("numPartitions must be > 0")

        if self.isDefined(self.persistPath) and not self.isDefined(self.persistFormat):
            raise ValueError("persistFormat must be defined when persistPath is set")