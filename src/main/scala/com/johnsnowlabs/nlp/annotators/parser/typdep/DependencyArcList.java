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

package com.johnsnowlabs.nlp.annotators.parser.typdep;

public class DependencyArcList {
    private int headsSize;
    private int[] st;
    private int[] edges;

    DependencyArcList(int[] heads) {
        headsSize = heads.length;
        st = new int[headsSize];
        edges = new int[headsSize];
        constructDepTreeArcList(heads);
    }

    int startIndex(int i) {
        return st[i];
    }

    int endIndex(int i) {
        return (i >= headsSize - 1) ? headsSize - 1 : st[i + 1];
    }

    public int get(int i) {
        return edges[i];
    }

    private void constructDepTreeArcList(int[] heads) {

        for (int i = 0; i < headsSize; ++i)
            st[i] = 0;

        for (int i = 1; i < headsSize; ++i) {
            int j = heads[i];
            ++st[j];
        }

        for (int i = 1; i < headsSize; ++i)
            st[i] += st[i - 1];

        for (int i = headsSize - 1; i > 0; --i) {
            int j = heads[i];
            --st[j];
            edges[st[j]] = i;
        }
    }

}
