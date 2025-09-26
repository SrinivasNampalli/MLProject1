#!/usr/bin/env python3
"""
Simple Markdown to PDF Converter
Alternative approach using markdown2 and creating styled HTML for printing
"""

import os
import re
import markdown2
from pathlib import Path

def convert_md_to_styled_html(md_file_path, output_html_path):
    """Convert Markdown file to HTML with print-friendly IEEE styling"""

    # Read the markdown file
    with open(md_file_path, 'r', encoding='utf-8') as f:
        md_content = f.read()

    # Convert markdown to HTML with extras
    html_content = markdown2.markdown(
        md_content,
        extras=[
            'tables',
            'fenced-code-blocks',
            'footnotes',
            'header-ids',
            'toc'
        ]
    )

    # IEEE-style CSS optimized for browser printing
    ieee_css = """
    <style>
    @media print {
        @page {
            size: A4;
            margin: 0.75in;
        }

        body {
            font-family: "Times New Roman", Times, serif !important;
            font-size: 11pt !important;
            line-height: 1.15 !important;
            color: black !important;
            background: white !important;
        }

        .no-print { display: none !important; }

        h1, h2, h3, h4, h5, h6 {
            page-break-after: avoid;
        }

        table, figure, img {
            page-break-inside: avoid;
        }

        p, li {
            orphans: 3;
            widows: 3;
        }
    }

    @media screen {
        body {
            max-width: 8.5in;
            margin: 0 auto;
            padding: 1in;
            background: white;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
    }

    body {
        font-family: "Times New Roman", Times, serif;
        font-size: 11pt;
        line-height: 1.2;
        color: black;
        text-align: justify;
        margin: 0;
        padding: 0;
    }

    h1 {
        font-size: 16pt;
        font-weight: bold;
        text-align: center;
        margin: 0 0 0.5em 0;
        padding: 0;
        page-break-after: avoid;
    }

    h2 {
        font-size: 12pt;
        font-weight: bold;
        margin: 1.2em 0 0.6em 0;
        text-transform: uppercase;
        letter-spacing: 0.5pt;
        page-break-after: avoid;
    }

    h3 {
        font-size: 11pt;
        font-weight: bold;
        margin: 1em 0 0.5em 0;
        font-style: italic;
        page-break-after: avoid;
    }

    p {
        margin: 0 0 0.6em 0;
        text-indent: 0.2in;
    }

    p:first-child,
    h1 + p,
    h2 + p,
    h3 + p {
        text-indent: 0;
    }

    .abstract {
        font-size: 11pt;
        margin: 1em 0;
    }

    .abstract strong {
        font-weight: bold;
    }

    strong {
        font-weight: bold;
    }

    em {
        font-style: italic;
    }

    ul, ol {
        margin: 0.6em 0;
        padding-left: 1.5em;
    }

    li {
        margin: 0.3em 0;
    }

    table {
        border-collapse: collapse;
        margin: 1em auto;
        font-size: 10pt;
        width: 95%;
        page-break-inside: avoid;
    }

    th, td {
        border: 1px solid black;
        padding: 4px 8px;
        text-align: center;
        vertical-align: middle;
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

    .figure {
        text-align: center;
        margin: 1em 0;
        page-break-inside: avoid;
    }

    .figure-caption {
        font-size: 10pt;
        font-style: italic;
        margin-top: 0.5em;
        text-align: center;
        font-weight: normal;
    }

    .equation {
        text-align: center;
        margin: 1em 0;
        font-style: italic;
    }

    code {
        font-family: "Courier New", Courier, monospace;
        font-size: 10pt;
        background-color: #f8f8f8;
        padding: 2px 4px;
        border: 1px solid #ddd;
    }

    pre {
        font-family: "Courier New", Courier, monospace;
        font-size: 9pt;
        background-color: #f8f8f8;
        padding: 0.5em;
        margin: 1em 0;
        white-space: pre-wrap;
        border: 1px solid #ddd;
        page-break-inside: avoid;
    }

    .references {
        font-size: 10pt;
    }

    .references ol {
        padding-left: 1em;
    }

    .references li {
        margin: 0.4em 0;
        text-indent: -0.5em;
        margin-left: 0.5em;
        text-align: left;
    }

    .table-title {
        font-size: 10pt;
        font-weight: bold;
        text-align: center;
        margin: 1em 0 0.5em 0;
    }

    .authors {
        font-size: 10pt;
        margin-top: 2em;
        page-break-inside: avoid;
    }

    .authors strong {
        font-weight: bold;
    }

    .index-terms {
        font-size: 10pt;
        margin: 1em 0;
        font-style: italic;
    }

    .index-terms strong {
        font-weight: bold;
        font-style: normal;
    }

    /* Print button for screen viewing */
    .print-button {
        position: fixed;
        top: 20px;
        right: 20px;
        background: #007ACC;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        cursor: pointer;
        font-size: 14pt;
        z-index: 1000;
    }

    .print-button:hover {
        background: #005999;
    }

    @media print {
        .print-button { display: none; }
    }

    /* Math formatting */
    .math {
        font-style: italic;
        text-align: center;
        margin: 1em 0;
    }
    </style>
    """

    # Process the HTML content for better formatting
    html_content = process_html_for_ieee(html_content)

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
        <button class="print-button no-print" onclick="window.print()">Print to PDF</button>
        {html_content}

        <script>
        // Auto-open print dialog for easy PDF conversion
        document.addEventListener('DOMContentLoaded', function() {{
            // Uncomment the line below to auto-open print dialog
            // setTimeout(() => window.print(), 1000);
        }});
        </script>
    </body>
    </html>
    """

    # Save HTML file
    with open(output_html_path, 'w', encoding='utf-8') as f:
        f.write(full_html)

    return full_html

def process_html_for_ieee(html_content):
    """Process HTML content for IEEE formatting"""

    # Process abstract
    html_content = re.sub(
        r'<p><strong>Abstract—</strong>([^<]+)</p>',
        r'<div class="abstract"><p><strong>Abstract—</strong>\1</p></div>',
        html_content
    )

    # Process index terms
    html_content = re.sub(
        r'<p><strong>Index Terms—</strong>([^<]+)</p>',
        r'<div class="index-terms"><p><strong>Index Terms—</strong>\1</p></div>',
        html_content
    )

    # Fix image references and add figure captions
    html_content = re.sub(
        r'<p><img src="([^"]+)" alt="([^"]*)"[^>]*></p>',
        r'<div class="figure"><img src="\1" alt="\2"><div class="figure-caption">Fig. \2</div></div>',
        html_content
    )

    # Process table titles
    html_content = re.sub(
        r'<p>(TABLE [IVX]+)<br>\s*([^<]+)</p>\s*<table>',
        r'<div class="table-title">\1<br>\2</div><table>',
        html_content
    )

    # Process equations (simple math formatting)
    html_content = re.sub(
        r'<p>\$\$([^$]+)\$\$</p>',
        r'<div class="equation">$$\1$$</div>',
        html_content
    )

    # Add references class
    html_content = re.sub(
        r'<h2>REFERENCES</h2>',
        r'<h2>REFERENCES</h2><div class="references">',
        html_content
    )

    # Close references div
    if '<div class="references">' in html_content:
        html_content = html_content.rsplit('</body>', 1)[0] + '</div></body>'

    # Process authors section
    html_content = re.sub(
        r'<p><strong>Authors:</strong></p>',
        r'<div class="authors"><p><strong>Authors:</strong></p>',
        html_content
    )

    # Close authors div
    if '<div class="authors">' in html_content:
        # Find the last occurrence and close it properly
        html_content = html_content.rsplit('</body>', 1)[0] + '</div></body>'

    return html_content

def main():
    """Main function"""

    # File paths
    md_file = "ieee-research-paper.md"
    html_file = "IEEE_Research_Paper_Printable.html"

    print("Converting Markdown to print-ready HTML...")
    html_content = convert_md_to_styled_html(md_file, html_file)
    print(f"HTML file created: {html_file}")

    print("\nTo convert to PDF:")
    print("1. Open the HTML file in Chrome/Edge")
    print("2. Press Ctrl+P (or click the 'Print to PDF' button)")
    print("3. Select 'Save as PDF' as destination")
    print("4. Adjust settings if needed:")
    print("   - Paper size: A4")
    print("   - Margins: Default")
    print("   - Include headers/footers: No")
    print("   - Background graphics: Yes (to preserve styling)")
    print("5. Click 'Save'")

    print(f"\nFiles created:")
    print(f"  - Printable HTML: {html_file}")
    print(f"  - Original Markdown (updated): {md_file}")

if __name__ == "__main__":
    main()