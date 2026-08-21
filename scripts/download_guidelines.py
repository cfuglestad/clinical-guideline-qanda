"""Download publicly available clinical practice guidelines for testing.

USPSTF recommendation statements are published as open-access PDFs via JAMA.
This script downloads a curated set of guidelines into data/raw/.

Usage:
    python scripts/download_guidelines.py
"""

import urllib.request
from pathlib import Path

# Each entry: (filename, URL, description)
# USPSTF guidelines are published open-access via JAMA Network
GUIDELINES = [
    (
        "uspstf-colorectal-cancer-screening-2021.pdf",
        "https://jamanetwork.com/journals/jama/fullarticle/2779985",
        "USPSTF: Screening for Colorectal Cancer (2021)",
    ),
    (
        "uspstf-lung-cancer-screening-2021.pdf",
        "https://jamanetwork.com/journals/jama/fullarticle/2777244",
        "USPSTF: Screening for Lung Cancer (2021)",
    ),
    (
        "uspstf-depression-screening-adults-2023.pdf",
        "https://jamanetwork.com/journals/jama/fullarticle/2806389",
        "USPSTF: Screening for Depression and Suicide Risk in Adults (2023)",
    ),
]

# Direct PDF links for USPSTF final recommendation statements
# These are the actual PDF downloads from the USPSTF website
USPSTF_DIRECT = [
    (
        "uspstf-statin-use-prevention-2022.pdf",
        (
            "https://www.uspreventiveservicestaskforce.org/uspstf/"
            "document/RecommendationStatementFinal/"
            "statin-use-for-the-primary-prevention-of-cardiovascular-"
            "disease-in-adults-preventive-medication"
        ),
        "USPSTF: Statin Use for Primary Prevention of CVD (2022)",
    ),
]


def download_guidelines(output_dir: Path) -> None:
    """Download guideline PDFs to the specified directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading guidelines to {output_dir}/\n")
    print(
        "NOTE: JAMA URLs serve HTML by default. For the initial test,\n"
        "please manually download PDFs from the USPSTF website:\n"
        "  https://www.uspreventiveservicestaskforce.org/uspstf/\n"
        "        recommendation-topics\n\n"
        "Steps:\n"
        "  1. Search for a topic (e.g., 'Colorectal Cancer')\n"
        "  2. Click the recommendation\n"
        "  3. Look for 'PDF' download link on the page\n"
        "  4. Save to data/raw/\n\n"
        "Alternatively, download directly from JAMA (requires \n"
        "clicking the PDF icon on the article page).\n"
    )

    print("Recommended starter guidelines:")
    for filename, url, desc in GUIDELINES + USPSTF_DIRECT:
        target = output_dir / filename
        if target.exists():
            print(f"  [exists] {desc}")
        else:
            print(f"  [needed] {desc}")
            print(f"           -> {url}")
            print(f"           -> save as: {target}")
        print()


if __name__ == "__main__":
    data_dir = Path("data/raw")
    download_guidelines(data_dir)
