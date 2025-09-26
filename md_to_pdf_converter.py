#!/usr/bin/env python3
"""
Markdown to PDF Converter with Image Preservation
Converts IEEE research paper from Markdown to PDF format while preserving images and formatting
"""

import os
import re
import markdown
import weasyprint
from pathlib import Path

def convert_md_to_html(md_file_path, output_html_path=None):
    """Convert Markdown file to HTML with proper styling"""

    # Read the markdown file
    with open(md_file_path, 'r', encoding='utf-8') as f:
        md_content = f.read()

    # Convert markdown to HTML
    html_content = markdown.markdown(
        md_content,
        extensions=[
            'markdown.extensions.tables',
            'markdown.extensions.fenced_code',
            'markdown.extensions.toc',
            'markdown.extensions.footnotes'
        ]
    )

    # IEEE-style CSS
    ieee_css = """
    <style>
    @page {
        size: A4;
        margin: 1in;
        @bottom-center {
            content: counter(page);
            font-family: Times, serif;
            font-size: 10pt;
        }
    }

    body {
        font-family: Times, "Times New Roman", serif;
        font-size: 11pt;
        line-height: 1.2;
        color: black;
        margin: 0;
        padding: 0;
        text-align: justify;
    }

    h1 {
        font-size: 18pt;
        font-weight: bold;
        text-align: center;
        margin: 0 0 0.5em 0;
        padding: 0;
    }

    h2 {
        font-size: 12pt;
        font-weight: bold;
        margin: 1em 0 0.5em 0;
        text-transform: uppercase;
        letter-spacing: 0.5pt;
    }

    h3 {
        font-size: 11pt;
        font-weight: bold;
        margin: 0.8em 0 0.4em 0;
        font-style: italic;
    }

    p {
        margin: 0 0 0.5em 0;
        text-indent: 0.2in;
    }

    .abstract p:first-child {
        text-indent: 0;
        font-weight: bold;
    }

    strong {
        font-weight: bold;
    }

    em {
        font-style: italic;
    }

    ul, ol {
        margin: 0.5em 0;
        padding-left: 1.5em;
    }

    li {
        margin: 0.2em 0;
    }

    table {
        border-collapse: collapse;
        margin: 1em auto;
        font-size: 10pt;
        width: 90%;
    }

    th, td {
        border: 1px solid black;
        padding: 4px 6px;
        text-align: center;
    }

    th {
        background-color: #f0f0f0;
        font-weight: bold;
    }

    img {
        max-width: 100%;
        height: auto;
        display: block;
        margin: 1em auto;
        page-break-inside: avoid;
    }

    figure {
        text-align: center;
        margin: 1em 0;
        page-break-inside: avoid;
    }

    figcaption {
        font-size: 10pt;
        font-style: italic;
        margin-top: 0.5em;
        text-align: center;
    }

    .equation {
        text-align: center;
        margin: 1em 0;
        font-style: italic;
    }

    code {
        font-family: "Courier New", monospace;
        font-size: 10pt;
        background-color: #f5f5f5;
        padding: 2px 4px;
    }

    pre {
        font-family: "Courier New", monospace;
        font-size: 9pt;
        background-color: #f5f5f5;
        padding: 0.5em;
        margin: 1em 0;
        white-space: pre-wrap;
        border: 1px solid #ddd;
    }

    .references {
        font-size: 10pt;
    }

    .references ol {
        padding-left: 1em;
    }

    .references li {
        margin: 0.3em 0;
        text-indent: -0.5em;
        margin-left: 0.5em;
    }

    .page-break {
        page-break-before: always;
    }

    .no-break {
        page-break-inside: avoid;
    }

    /* Table styling */
    .table-caption {
        font-size: 10pt;
        font-weight: bold;
        text-align: center;
        margin: 0.5em 0;
    }

    /* Author section */
    .authors {
        font-size: 10pt;
        margin-top: 2em;
    }

    .authors strong {
        font-weight: bold;
    }
    </style>
    """

    # Process the HTML content to improve formatting
    html_content = process_html_content(html_content)

    # Create complete HTML document
    full_html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Multi-Sensor Fusion for Predictive Maintenance - IEEE Research Paper</title>
        {ieee_css}
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """

    # Save HTML file if path provided
    if output_html_path:
        with open(output_html_path, 'w', encoding='utf-8') as f:
            f.write(full_html)

    return full_html

def process_html_content(html_content):
    """Process HTML content to improve formatting for IEEE style"""

    # Fix image paths and add figure elements
    html_content = re.sub(
        r'<p><img src="([^"]+)" alt="([^"]*)"[^>]*></p>',
        r'<figure class="no-break"><img src="\1" alt="\2"><figcaption>\2</figcaption></figure>',
        html_content
    )

    # Add table captions
    html_content = re.sub(
        r'<p>(TABLE [IVX]+)<br>\s*([^<]+)</p>\s*<table>',
        r'<div class="table-caption">\1<br>\2</div><table>',
        html_content
    )

    # Format equations
    html_content = re.sub(
        r'<p>\$\$([^$]+)\$\$</p>',
        r'<div class="equation">$$\1$$</div>',
        html_content
    )

    # Add class to references section
    html_content = re.sub(
        r'<h2>REFERENCES</h2>',
        r'<h2>REFERENCES</h2><div class="references">',
        html_content
    )

    # Close references div at end of document
    if 'REFERENCES' in html_content:
        html_content = html_content + '</div>'

    # Add class to authors section
    html_content = re.sub(
        r'<p><strong>Authors:</strong></p>',
        r'<div class="authors"><strong>Authors:</strong>',
        html_content
    )

    # Close authors div
    if '<div class="authors">' in html_content:
        html_content = html_content + '</div>'

    return html_content

def convert_html_to_pdf(html_content, output_pdf_path, base_path=None):
    """Convert HTML content to PDF using WeasyPrint"""

    if base_path is None:
        base_path = Path.cwd()

    # Create WeasyPrint HTML document
    html_doc = weasyprint.HTML(
        string=html_content,
        base_url=str(base_path)
    )

    # Convert to PDF
    html_doc.write_pdf(output_pdf_path)

def main():
    """Main function to convert MD to PDF"""

    # File paths
    md_file = "ieee-research-paper.md"
    html_file = "ieee-research-paper.html"
    pdf_file = "IEEE_Research_Paper_Multi_Sensor_Fusion_Predictive_Maintenance.pdf"

    # Get current directory
    base_path = Path.cwd()

    print("Converting Markdown to HTML...")
    html_content = convert_md_to_html(md_file, html_file)
    print(f"HTML file saved: {html_file}")

    print("Converting HTML to PDF...")
    convert_html_to_pdf(html_content, pdf_file, base_path)
    print(f"PDF file saved: {pdf_file}")

    print("Conversion completed successfully!")
    print(f"Output files:")
    print(f"  - HTML: {html_file}")
    print(f"  - PDF: {pdf_file}")

if __name__ == "__main__":
    main()