#!/usr/bin/env python3
"""
Align Bangla and English Wikipedia articles for bilingual training.

Usage:
    python scripts/align_bilingual_wiki.py \
        --bn datasets/wikipedia/raw/bnwiki-latest-pages-articles.xml.bz2 \
        --en datasets/wikipedia/raw/enwiki-latest-pages-articles.xml.bz2 \
        --output datasets/wikipedia/bilingual
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class BilingualAligner:
    """Align Bangla and English Wikipedia articles."""

    def __init__(self):
        self.bn_articles = {}  # title -> content
        self.en_articles = {}  # title -> content
        self.interwiki_links = {}  # bn_title -> en_title

    def extract_interwiki_links(self, text: str, source_lang: str = "bn") -> Optional[str]:
        """
        Extract interwiki links from Wikipedia text.

        Args:
            text: Wikipedia article text
            source_lang: Source language code

        Returns:
            Target language title if found
        """
        target_lang = "en" if source_lang == "bn" else "bn"

        # Pattern for interwiki links: [[en:Title]] or [[bn:শিরোনাম]]
        pattern = rf"\[\[{target_lang}:([^\]]+)\]\]"
        matches = re.findall(pattern, text, re.IGNORECASE)

        if matches:
            return matches[0].strip()

        return None

    def parse_wikipedia_xml(self, xml_path: Path, lang: str) -> Dict[str, str]:
        """
        Parse Wikipedia XML dump and extract articles.

        Args:
            xml_path: Path to XML dump
            lang: Language code

        Returns:
            Dictionary of title -> content
        """
        logger.info(f"Parsing {lang} Wikipedia XML: {xml_path}")

        articles = {}

        # Note: For large dumps, use streaming parser or WikiExtractor
        # This is a simplified version for demonstration

        try:
            # Try to parse with streaming
            import bz2

            if xml_path.suffix == ".bz2":
                file_handle = bz2.open(xml_path, "rt", encoding="utf-8")
            else:
                file_handle = open(xml_path, "r", encoding="utf-8")

            # Simple extraction (for demo purposes)
            # In production, use WikiExtractor or similar
            current_title = None
            current_text = []
            in_text = False

            for line in file_handle:
                if "<title>" in line:
                    current_title = re.search(r"<title>(.*?)</title>", line)
                    if current_title:
                        current_title = current_title.group(1)

                elif "<text" in line:
                    in_text = True
                    # Extract text from same line if present
                    text_match = re.search(r"<text[^>]*>(.*)", line)
                    if text_match:
                        current_text.append(text_match.group(1))

                elif "</text>" in line:
                    in_text = False
                    # Extract remaining text
                    text_match = re.search(r"(.*)</text>", line)
                    if text_match:
                        current_text.append(text_match.group(1))

                    # Store article
                    if current_title and current_text:
                        articles[current_title] = "\n".join(current_text)

                    current_title = None
                    current_text = []

                elif in_text:
                    current_text.append(line)

                # Limit for demo (remove in production)
                if len(articles) >= 10000:
                    logger.info("Reached limit of 10000 articles for demo")
                    break

            file_handle.close()

        except Exception as e:
            logger.error(f"Failed to parse XML: {e}")
            logger.info("Consider using WikiExtractor for large dumps")
            raise

        logger.info(f"Extracted {len(articles)} articles from {lang} Wikipedia")
        return articles

    def align_articles(
        self,
        bn_articles: Dict[str, str],
        en_articles: Dict[str, str],
    ) -> List[Tuple[str, str, str, str]]:
        """
        Align Bangla and English articles using interwiki links.

        Args:
            bn_articles: Bangla articles (title -> content)
            en_articles: English articles (title -> content)

        Returns:
            List of aligned articles: (bn_title, bn_content, en_title, en_content)
        """
        logger.info("Aligning articles using interwiki links...")

        aligned = []

        # Build interwiki mapping from Bangla to English
        for bn_title, bn_content in bn_articles.items():
            en_title = self.extract_interwiki_links(bn_content, "bn")

            if en_title and en_title in en_articles:
                en_content = en_articles[en_title]
                aligned.append((bn_title, bn_content, en_title, en_content))

        logger.info(f"Aligned {len(aligned)} article pairs")
        return aligned

    def save_aligned_corpus(
        self,
        aligned_articles: List[Tuple[str, str, str, str]],
        output_dir: Path,
    ) -> None:
        """
        Save aligned corpus in various formats.

        Args:
            aligned_articles: List of aligned articles
            output_dir: Output directory
        """
        logger.info(f"Saving aligned corpus to {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save as JSON
        json_path = output_dir / "aligned_articles.json"
        with open(json_path, "w", encoding="utf-8") as f:
            data = [
                {
                    "bn_title": bn_title,
                    "bn_content": bn_content,
                    "en_title": en_title,
                    "en_content": en_content,
                }
                for bn_title, bn_content, en_title, en_content in aligned_articles
            ]
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved JSON: {json_path}")

        # Save as parallel text files
        bn_path = output_dir / "bangla.txt"
        en_path = output_dir / "english.txt"

        with open(bn_path, "w", encoding="utf-8") as bn_f, open(
            en_path, "w", encoding="utf-8"
        ) as en_f:

            for bn_title, bn_content, en_title, en_content in aligned_articles:
                # Write titles and content
                bn_f.write(f"{bn_title}\n{bn_content}\n\n")
                en_f.write(f"{en_title}\n{en_content}\n\n")

        logger.info(f"Saved parallel texts: {bn_path}, {en_path}")

        # Save metadata
        metadata = {
            "total_pairs": len(aligned_articles),
            "bn_file": str(bn_path),
            "en_file": str(en_path),
            "json_file": str(json_path),
        }

        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved metadata: {metadata_path}")


def align_using_extracted_files(
    bn_extract_dir: Path,
    en_extract_dir: Path,
    output_dir: Path,
) -> None:
    """
    Align articles from already extracted Wikipedia files.

    Args:
        bn_extract_dir: Directory with extracted Bangla articles
        en_extract_dir: Directory with extracted English articles
        output_dir: Output directory for aligned corpus
    """
    logger.info("Aligning from extracted files...")

    # Load extracted articles
    bn_articles = {}
    en_articles = {}

    # Load Bangla articles
    for file_path in bn_extract_dir.rglob("wiki_*"):
        if not file_path.is_file():
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    title = data.get("title", "")
                    text = data.get("text", "")
                    if title and text:
                        bn_articles[title] = text
                except json.JSONDecodeError:
                    continue

    # Load English articles
    for file_path in en_extract_dir.rglob("wiki_*"):
        if not file_path.is_file():
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    title = data.get("title", "")
                    text = data.get("text", "")
                    if title and text:
                        en_articles[title] = text
                except json.JSONDecodeError:
                    continue

    logger.info(f"Loaded {len(bn_articles)} Bangla articles")
    logger.info(f"Loaded {len(en_articles)} English articles")

    # Align
    aligner = BilingualAligner()
    aligned = aligner.align_articles(bn_articles, en_articles)

    # Save
    aligner.save_aligned_corpus(aligned, output_dir)


def main():
    parser = argparse.ArgumentParser(description="Align Bangla and English Wikipedia articles")
    parser.add_argument(
        "--bn",
        type=str,
        help="Path to Bangla Wikipedia dump or extracted directory",
    )
    parser.add_argument(
        "--en",
        type=str,
        help="Path to English Wikipedia dump or extracted directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="datasets/wikipedia/bilingual",
        help="Output directory for aligned corpus",
    )
    parser.add_argument(
        "--use-extracted",
        action="store_true",
        help="Use already extracted files instead of XML dumps",
    )

    args = parser.parse_args()

    if not args.bn or not args.en:
        logger.error("Please specify both --bn and --en paths")
        sys.exit(1)

    bn_path = Path(args.bn)
    en_path = Path(args.en)
    output_dir = Path(args.output)

    try:
        if args.use_extracted:
            # Use extracted files
            align_using_extracted_files(bn_path, en_path, output_dir)
        else:
            # Parse XML dumps
            aligner = BilingualAligner()

            bn_articles = aligner.parse_wikipedia_xml(bn_path, "bn")
            en_articles = aligner.parse_wikipedia_xml(en_path, "en")

            aligned = aligner.align_articles(bn_articles, en_articles)
            aligner.save_aligned_corpus(aligned, output_dir)

        logger.info("Bilingual alignment complete!")

    except Exception as e:
        logger.error(f"Alignment failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
