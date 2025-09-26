#!/usr/bin/env python3
"""
HTML to PDF converter using Playwright (browser automation)
This provides a more reliable PDF conversion with proper rendering
"""

import asyncio
import sys
from pathlib import Path

try:
    from playwright.async_api import async_playwright
except ImportError:
    print("Playwright not installed. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "playwright"])
    subprocess.check_call([sys.executable, "-m", "playwright", "install", "chromium"])
    from playwright.async_api import async_playwright

async def html_to_pdf(html_file_path, pdf_file_path):
    """Convert HTML to PDF using Playwright"""

    async with async_playwright() as p:
        # Launch browser
        browser = await p.chromium.launch()
        page = await browser.new_page()

        # Load the HTML file
        html_path = Path(html_file_path).resolve()
        await page.goto(f"file://{html_path}")

        # Wait for any potential content to load
        await page.wait_for_timeout(2000)

        # Generate PDF with proper settings
        await page.pdf(
            path=pdf_file_path,
            format='A4',
            margin={
                'top': '0.75in',
                'right': '0.75in',
                'bottom': '0.75in',
                'left': '0.75in'
            },
            print_background=True,
            prefer_css_page_size=True
        )

        await browser.close()

def main():
    """Main conversion function"""

    html_file = "IEEE_Research_Paper_Printable.html"
    pdf_file = "IEEE_Research_Paper_Multi_Sensor_Fusion_Predictive_Maintenance.pdf"

    if not Path(html_file).exists():
        print(f"HTML file {html_file} not found. Please run simple_md_to_pdf.py first.")
        return

    print(f"Converting {html_file} to {pdf_file}...")

    # Run the async conversion
    asyncio.run(html_to_pdf(html_file, pdf_file))

    print(f"PDF conversion completed: {pdf_file}")

if __name__ == "__main__":
    main()