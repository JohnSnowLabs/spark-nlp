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

package com.johnsnowlabs.nlp.annotators.parser.typdep.util;

public class ScoreCollector implements Collector {

    private float[] weights;
    private float score;

    public float getScore() {
        return score;
    }

    public ScoreCollector(float[] w) {
        weights = w;
        score = 0;
    }

    @Override
    public void addEntry(int x) {
        score += weights[x];
    }

    @Override
    public void addEntry(int x, float va) {
        score += weights[x] * va;
    }

}
