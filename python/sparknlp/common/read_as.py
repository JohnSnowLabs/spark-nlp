#  Copyright 2017-2022 John Snow Labs
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
"""Contains utility classes for reading resources."""


class ReadAs(object):
    """Object that contains constants for how to read Spark Resources.

    Possible values are:

    ================= =======================================
    Value             Description
    ================= =======================================
    ``ReadAs.TEXT``   Read the resource as text.
    ``ReadAs.SPARK``  Read the resource as a Spark DataFrame.
    ``ReadAs.BINARY`` Read the resource as a binary file.
    ================= =======================================
    """
    TEXT = "TEXT"
    SPARK = "SPARK"
    BINARY = "BINARY"

