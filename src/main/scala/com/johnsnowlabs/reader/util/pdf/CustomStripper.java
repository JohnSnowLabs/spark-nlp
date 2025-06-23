/*
 * Copyright 2017-2025 John Snow Labs
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
package com.johnsnowlabs.reader.util.pdf;

import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.pdmodel.PDPage;
import org.apache.pdfbox.text.PDFTextStripper;
import org.apache.pdfbox.text.TextPosition;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class CustomStripper extends PDFTextStripper  {

    public CustomStripper() throws IOException {

    }

    @Override
    protected void startPage(PDPage page) throws IOException
    {
        startOfLine = true;
        super.startPage(page);
    }

    @Override
    protected void writeLineSeparator() throws IOException
    {
        startOfLine = true;
        super.writeLineSeparator();
    }

    @Override
    public String getText(PDDocument doc) throws IOException
    {
        lines = new ArrayList<>();
        return super.getText(doc);
    }

    @Override
    protected void writeWordSeparator() throws IOException
    {
        CustomTextLine tmpline;

        tmpline = lines.get(lines.size() - 1);
        tmpline.text += getWordSeparator();

        super.writeWordSeparator();
    }


    @Override
    protected void writeString(String text, List<TextPosition> textPositions) throws IOException
    {
        CustomTextLine tmpline = null;

        if (startOfLine) {
            tmpline = new CustomTextLine();
            tmpline.text = text;
            tmpline.textPositions = textPositions;
            lines.add(tmpline);
        } else {
            tmpline = lines.get(lines.size() - 1);
            tmpline.text += text;
            tmpline.textPositions.addAll(textPositions);
        }

        if (startOfLine)
        {
            startOfLine = false;
        }
        super.writeString(text, textPositions);
    }

    boolean startOfLine = true;
    public ArrayList<CustomTextLine> lines = null;

}
