import asyncio
import sys
import os

async def convert():
    from playwright.async_api import async_playwright

    html_path = r"c:\Users\USER\Desktop\lg_vibe_ml\customer_satisfaction_ver2\reports\customer_satisfaction_dashboard.html"
    pdf_path  = r"c:\Users\USER\Desktop\lg_vibe_ml\customer_satisfaction_ver2\reports\customer_satisfaction_report.pdf"

    file_url = "file:///" + html_path.replace("\\", "/")

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()

        print("Loading page...")
        await page.goto(file_url, wait_until="networkidle", timeout=30000)

        # Chart.js 렌더링 대기
        await page.wait_for_timeout(3000)

        print("Generating PDF...")
        await page.pdf(
            path=pdf_path,
            format="A4",
            print_background=True,
            margin={"top": "10mm", "bottom": "10mm", "left": "8mm", "right": "8mm"},
        )

        await browser.close()

    size = os.path.getsize(pdf_path)
    print(f"PDF saved: {pdf_path}")
    print(f"Size: {size / 1024:.1f} KB")

asyncio.run(convert())
