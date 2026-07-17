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

import re


SUPPORTED_FORWARD_SPARK4_VERSIONS = frozenset(
    {"4.0.1", "4.1.0", "4.1.1", "4.1.2"}
)

_VARIANT_ARTIFACTS = {
    "cpu": "spark-nlp",
    "gpu": "spark-nlp-gpu",
    "silicon": "spark-nlp-silicon",
    "aarch64": "spark-nlp-aarch64",
}

_RELEASE_VERSION_PATTERN = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")


def resolve_spark_nlp_coordinate(
    pyspark_version,
    spark_nlp_version,
    gpu=False,
    apple_silicon=False,
    aarch64=False,
):
    """Resolve the Spark NLP Maven coordinate for a supported PySpark runtime."""
    selected_variants = [
        name
        for name, enabled in (
            ("gpu", gpu),
            ("silicon", apple_silicon),
            ("aarch64", aarch64),
        )
        if enabled
    ]
    if len(selected_variants) > 1:
        raise ValueError(
            "Only one Spark NLP hardware variant can be enabled: "
            "gpu, apple_silicon, or aarch64."
        )

    normalized_version = str(pyspark_version)
    version_match = _RELEASE_VERSION_PATTERN.fullmatch(normalized_version)
    if version_match is None:
        raise _unsupported_pyspark_version(normalized_version)

    spark_major = int(version_match.group(1))
    if spark_major == 3:
        scala_binary_version = "2.12"
        profile_suffix = ""
    elif normalized_version == "4.0.0":
        scala_binary_version = "2.13"
        profile_suffix = "-spark400"
    elif normalized_version in SUPPORTED_FORWARD_SPARK4_VERSIONS:
        scala_binary_version = "2.13"
        profile_suffix = ""
    else:
        raise _unsupported_pyspark_version(normalized_version)

    variant = selected_variants[0] if selected_variants else "cpu"
    artifact = f"{_VARIANT_ARTIFACTS[variant]}{profile_suffix}_{scala_binary_version}"
    return f"com.johnsnowlabs.nlp:{artifact}:{spark_nlp_version}"


def _unsupported_pyspark_version(pyspark_version):
    supported_spark4_versions = ", ".join(
        ["4.0.0"] + sorted(SUPPORTED_FORWARD_SPARK4_VERSIONS)
    )
    return ValueError(
        f"Unsupported PySpark version '{pyspark_version}'. "
        "Spark NLP supports Spark 3.x with Scala 2.12 and the validated "
        f"Spark 4 versions {supported_spark4_versions} with Scala 2.13."
    )
