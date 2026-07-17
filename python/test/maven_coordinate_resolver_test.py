#  Copyright 2017-2026 John Snow Labs
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

import inspect
from types import SimpleNamespace

import pytest
import sparknlp


SPARK_NLP_MAVEN_VERSION = sparknlp.__version__.split("-")[0].split("+")[0]


class FakeHadoopConfiguration:
    def set(self, key, value):
        pass


class FakeJsc:
    def hadoopConfiguration(self):
        return FakeHadoopConfiguration()


class FakeSparkContext:
    def __init__(self, gateway=None):
        self.gateway = gateway
        self._jsc = FakeJsc()


class FakeSparkSessionInstance:
    def __init__(self, spark_context=None):
        self.sparkContext = spark_context or FakeSparkContext()


class FakeBuilder:
    def __init__(self):
        self.settings = {}

    def appName(self, value):
        self.settings["spark.app.name"] = value
        return self

    def master(self, value):
        self.settings["spark.master"] = value
        return self

    def config(self, key, value):
        self.settings[key] = value
        return self

    def getOrCreate(self):
        return FakeSparkSessionInstance()


class FakeSparkSession:
    _instantiatedSession = None
    builder = FakeBuilder()

    def __new__(cls, spark_context=None):
        return FakeSparkSessionInstance(spark_context)


class FakeSparkConf:
    latest = None

    def __init__(self):
        self.settings = {}
        type(self).latest = self

    def setAppName(self, value):
        self.settings["spark.app.name"] = value
        return self

    def setMaster(self, value):
        self.settings["spark.master"] = value
        return self

    def set(self, key, value):
        self.settings[key] = value
        return self


class FakeThread:
    def __init__(self, target):
        self.target = target

    def start(self):
        pass

    def join(self):
        pass


def resolve_coordinate(*args, **kwargs):
    from sparknlp._maven import resolve_spark_nlp_coordinate

    return resolve_spark_nlp_coordinate(*args, **kwargs)


@pytest.mark.fast
@pytest.mark.parametrize(
    "spark_version,expected_artifact",
    [
        ("3.0.0", "spark-nlp_2.12"),
        ("3.5.6", "spark-nlp_2.12"),
        ("4.0.0", "spark-nlp-spark400_2.13"),
        ("4.0.1", "spark-nlp_2.13"),
        ("4.1.0", "spark-nlp_2.13"),
        ("4.1.1", "spark-nlp_2.13"),
        ("4.1.2", "spark-nlp_2.13"),
    ],
)
def test_resolves_cpu_artifact_from_pyspark_version(spark_version, expected_artifact):
    coordinate = resolve_coordinate(spark_version, SPARK_NLP_MAVEN_VERSION)

    assert coordinate == (
        f"com.johnsnowlabs.nlp:{expected_artifact}:{SPARK_NLP_MAVEN_VERSION}"
    )


@pytest.mark.fast
@pytest.mark.parametrize(
    "variant_flags,expected_base_artifact",
    [
        ({}, "spark-nlp"),
        ({"gpu": True}, "spark-nlp-gpu"),
        ({"apple_silicon": True}, "spark-nlp-silicon"),
        ({"aarch64": True}, "spark-nlp-aarch64"),
    ],
)
def test_resolves_all_spark3_hardware_variants(variant_flags, expected_base_artifact):
    coordinate = resolve_coordinate("3.5.8", SPARK_NLP_MAVEN_VERSION, **variant_flags)

    assert coordinate == (
        f"com.johnsnowlabs.nlp:{expected_base_artifact}_2.12:"
        f"{SPARK_NLP_MAVEN_VERSION}"
    )


@pytest.mark.fast
@pytest.mark.parametrize(
    "variant_flags,expected_base_artifact",
    [
        ({}, "spark-nlp"),
        ({"gpu": True}, "spark-nlp-gpu"),
        ({"apple_silicon": True}, "spark-nlp-silicon"),
        ({"aarch64": True}, "spark-nlp-aarch64"),
    ],
)
def test_resolves_all_spark400_hardware_variants(variant_flags, expected_base_artifact):
    coordinate = resolve_coordinate("4.0.0", SPARK_NLP_MAVEN_VERSION, **variant_flags)

    assert coordinate == (
        f"com.johnsnowlabs.nlp:{expected_base_artifact}-spark400_2.13:"
        f"{SPARK_NLP_MAVEN_VERSION}"
    )


@pytest.mark.fast
@pytest.mark.parametrize(
    "variant_flags,expected_base_artifact",
    [
        ({}, "spark-nlp"),
        ({"gpu": True}, "spark-nlp-gpu"),
        ({"apple_silicon": True}, "spark-nlp-silicon"),
        ({"aarch64": True}, "spark-nlp-aarch64"),
    ],
)
def test_resolves_all_forward_spark4_hardware_variants(
    variant_flags, expected_base_artifact
):
    coordinate = resolve_coordinate("4.1.2", SPARK_NLP_MAVEN_VERSION, **variant_flags)

    assert coordinate == (
        f"com.johnsnowlabs.nlp:{expected_base_artifact}_2.13:"
        f"{SPARK_NLP_MAVEN_VERSION}"
    )


@pytest.mark.fast
@pytest.mark.parametrize(
    "spark_version",
    ["2.4.8", "4.0.2", "4.2.0", "5.0.0", "4.0.0.dev1", "invalid"],
)
def test_rejects_unsupported_or_unvalidated_pyspark_versions(spark_version):
    with pytest.raises(ValueError, match="Unsupported PySpark version"):
        resolve_coordinate(spark_version, SPARK_NLP_MAVEN_VERSION)


@pytest.mark.fast
def test_normalizes_version_like_objects_before_comparison():
    class VersionLike:
        def __str__(self):
            return "4.0.0"

    assert resolve_coordinate(VersionLike(), SPARK_NLP_MAVEN_VERSION) == (
        "com.johnsnowlabs.nlp:spark-nlp-spark400_2.13:"
        f"{SPARK_NLP_MAVEN_VERSION}"
    )


@pytest.mark.fast
def test_start_does_not_expose_manual_scala_selector():
    assert "scala213" not in inspect.signature(sparknlp.start).parameters


@pytest.mark.fast
def test_start_uses_resolver_coordinate_for_standard_session(monkeypatch):
    coordinate = (
        "com.johnsnowlabs.nlp:spark-nlp-spark400_2.13:"
        f"{SPARK_NLP_MAVEN_VERSION}"
    )
    calls = []

    def fake_resolver(*args, **kwargs):
        calls.append((args, kwargs))
        return coordinate

    FakeSparkSession.builder = FakeBuilder()
    monkeypatch.setattr(sparknlp, "pyspark_version", "4.0.0")
    monkeypatch.setattr(sparknlp, "resolve_spark_nlp_coordinate", fake_resolver)
    monkeypatch.setattr(sparknlp, "SparkSession", FakeSparkSession)

    sparknlp.start()

    assert calls == [
        (
            ("4.0.0", SPARK_NLP_MAVEN_VERSION),
            {
                "gpu": False,
                "apple_silicon": False,
                "aarch64": False,
            },
        )
    ]
    assert FakeSparkSession.builder.settings["spark.jars.packages"] == coordinate


@pytest.mark.fast
def test_start_uses_resolver_coordinate_for_realtime_session(monkeypatch):
    coordinate = (
        "com.johnsnowlabs.nlp:spark-nlp-gpu_2.13:"
        f"{SPARK_NLP_MAVEN_VERSION}"
    )
    calls = []

    def fake_resolver(*args, **kwargs):
        calls.append((args, kwargs))
        return coordinate

    process = SimpleNamespace(
        stdout=SimpleNamespace(readline=lambda: b""),
        stderr=SimpleNamespace(readline=lambda: b""),
    )
    gateway = SimpleNamespace(proc=process)

    monkeypatch.setattr(sparknlp, "pyspark_version", "4.1.2")
    monkeypatch.setattr(sparknlp, "resolve_spark_nlp_coordinate", fake_resolver)
    monkeypatch.setattr(sparknlp, "SparkConf", FakeSparkConf)
    monkeypatch.setattr(sparknlp, "SparkContext", FakeSparkContext)
    monkeypatch.setattr(sparknlp, "SparkSession", FakeSparkSession)
    monkeypatch.setattr(sparknlp, "launch_gateway", lambda **kwargs: gateway)
    monkeypatch.setattr(sparknlp.threading, "Thread", FakeThread)

    session = sparknlp.start(gpu=True, real_time_output=True)

    assert calls == [
        (
            ("4.1.2", SPARK_NLP_MAVEN_VERSION),
            {
                "gpu": True,
                "apple_silicon": False,
                "aarch64": False,
            },
        )
    ]
    assert isinstance(session, FakeSparkSessionInstance)
    assert FakeSparkConf.latest is not None
    assert FakeSparkConf.latest.settings["spark.jars.packages"] == coordinate


@pytest.mark.fast
def test_skip_maven_bypasses_runtime_resolution(monkeypatch):
    def unexpected_resolver(*args, **kwargs):
        raise AssertionError("Resolver must not run when Maven injection is disabled")

    FakeSparkSession.builder = FakeBuilder()
    monkeypatch.setattr(sparknlp, "resolve_spark_nlp_coordinate", unexpected_resolver)
    monkeypatch.setattr(sparknlp, "SparkSession", FakeSparkSession)

    sparknlp.start(
        params={"spark.jars": "/tmp/sparknlp.jar"},
        skip_sparknlp_maven=True,
    )

    assert "spark.jars.packages" not in FakeSparkSession.builder.settings
    assert FakeSparkSession.builder.settings["spark.jars"] == "/tmp/sparknlp.jar"


@pytest.mark.fast
def test_skip_maven_param_bypasses_runtime_resolution(monkeypatch):
    def unexpected_resolver(*args, **kwargs):
        raise AssertionError("Resolver must not run when Maven injection is disabled")

    FakeSparkSession.builder = FakeBuilder()
    monkeypatch.setattr(sparknlp, "resolve_spark_nlp_coordinate", unexpected_resolver)
    monkeypatch.setattr(sparknlp, "SparkSession", FakeSparkSession)

    sparknlp.start(
        params={
            "spark.jars": "/tmp/sparknlp.jar",
            "skip_sparknlp_maven": "true",
        }
    )

    assert "spark.jars.packages" not in FakeSparkSession.builder.settings
    assert FakeSparkSession.builder.settings["spark.jars"] == "/tmp/sparknlp.jar"
    assert "skip_sparknlp_maven" not in FakeSparkSession.builder.settings


@pytest.mark.fast
def test_start_merges_resolved_coordinate_with_caller_packages(monkeypatch):
    coordinate = f"com.johnsnowlabs.nlp:spark-nlp_2.13:{SPARK_NLP_MAVEN_VERSION}"
    caller_package = "org.example:extra-package_2.13:1.0.0"

    FakeSparkSession.builder = FakeBuilder()
    monkeypatch.setattr(sparknlp, "pyspark_version", "4.0.1")
    monkeypatch.setattr(
        sparknlp,
        "resolve_spark_nlp_coordinate",
        lambda *args, **kwargs: coordinate,
    )
    monkeypatch.setattr(sparknlp, "SparkSession", FakeSparkSession)

    sparknlp.start(params={"spark.jars.packages": caller_package})

    assert FakeSparkSession.builder.settings["spark.jars.packages"] == (
        coordinate + "," + caller_package
    )


@pytest.mark.fast
def test_rejects_conflicting_hardware_variants():
    with pytest.raises(ValueError, match="Only one Spark NLP hardware variant"):
        resolve_coordinate(
            "4.0.1", SPARK_NLP_MAVEN_VERSION, gpu=True, apple_silicon=True
        )
