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

package com.johnsnowlabs.nlp.annotators.parser.typdep.util;

public class DependencyLabel {

    private String dependency;
    private String label;
    private int head;
    private int begin;
    private int end;

    public String getDependency() {
        return dependency;
    }

    public String getLabel() {
        return label;
    }

    public int getHead() {
        return head;
    }

    public int getBegin() {
        return begin;
    }

    public int getEnd() {
        return end;
    }

    public DependencyLabel(String dependency, String label, int head, int begin, int end) {
        this.dependency = dependency;
        this.label = label;
        this.head = head;
        this.begin = begin;
        this.end = end;
    }
}
