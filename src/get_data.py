import argparse
import csv
import time
import xml.etree.ElementTree as ET
from typing import Callable, Optional

import requests


def get_gse_ids(pmid: str) -> list[str]:
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi?dbfrom=pubmed&db=gds&linkname=pubmed_gds&id={pmid}&retmode=xml"
    response = requests.get(url)
    root = ET.fromstring(response.text)

    gse_ids = [link.text for link in root.findall(".//LinkSetDb/Link/Id")]

    return gse_ids


def get_geo_details(gse_id: str) -> dict[str, str]:
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=gds&id={gse_id}&retmode=xml"
    response = requests.get(url)
    root = ET.fromstring(response.text)

    dataset_info = {
        "GSE_ID": gse_id,
        "Title": "N/A",
        "Experiment Type": "N/A",
        "Summary": "N/A",
        "Organism": "N/A",
        "Overall Design": "N/A",
    }

    for item in root.findall(".//DocSum"):
        dataset_info["Title"] = item.findtext(".//Item[@Name='title']", "N/A")
        dataset_info["Experiment Type"] = item.findtext(
            ".//Item[@Name='gdsType']", "N/A"
        )
        dataset_info["Summary"] = item.findtext(".//Item[@Name='summary']", "N/A")
        dataset_info["Organism"] = item.findtext(".//Item[@Name='taxon']", "N/A")
        dataset_info["Overall Design"] = item.findtext(".//Item[@Name='design']", "N/A")

    return dataset_info


def process_pmids(
    pmids: list[str],
    output_file: str,
    progress_bar: Optional[Callable] = None,
    progress_placeholder: Optional[Callable] = None,
) -> None:
    results = []

    for idx, pmid in enumerate(pmids):
        if progress_placeholder is not None:
            progress_placeholder.write(
                f"Parsing data from PMID {pmid}. Left to parse: {len(pmids) -idx} articles"
            )
        gse_ids = get_gse_ids(pmid)
        for gse_id in gse_ids:
            geo_details = get_geo_details(gse_id)
            geo_details["PMID"] = pmid
            results.append(geo_details)

        time.sleep(1)
        if progress_bar is not None:
            progress_bar.progress(idx + 1)

    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "PMID",
            "GSE_ID",
            "Title",
            "Experiment Type",
            "Summary",
            "Organism",
            "Overall Design",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(results)


def main():
    parser = argparse.ArgumentParser(
        description="Fetch GEO datasets for given PMIDs and save to CSV."
    )
    parser.add_argument(
        "input_file", help="Path to input file containing PMIDs (one per line)"
    )
    parser.add_argument("output_file", help="Path to output CSV file")

    args = parser.parse_args()
    process_pmids(args.input_file, args.output_file, None, None, True)


if __name__ == "__main__":
    main()
