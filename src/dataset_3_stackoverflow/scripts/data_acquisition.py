"""
Data Acquisition Module
========================
Fetches data from StackOverflow API with fallback to local CSV files.
Includes rate limiting, batching, and comprehensive error handling.
"""

import json
import csv
import re
import time
import logging
import sys
from pathlib import Path
from typing import Optional
from html import unescape
from datetime import datetime

import requests

from config import config, ERROR_PATTERNS, INFRA_PATTERNS

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.LOGS_DIR / 'data_acquisition.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class RateLimitedRequester:
    """Handles API requests with rate limiting and exponential backoff."""
    
    def __init__(self):
        self.last_request_time = 0
        self.consecutive_failures = 0
        self.total_requests = 0
        self.failed_requests = 0
    
    def wait_for_rate_limit(self):
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self.last_request_time
        wait_time = config.API_DELAY
        
        if self.consecutive_failures > 0:
            wait_time = min(
                config.API_DELAY * (config.BACKOFF_MULTIPLIER ** self.consecutive_failures),
                config.MAX_BACKOFF
            )
        
        if elapsed < wait_time:
            sleep_time = wait_time - elapsed
            logger.debug(f"Rate limiting: waiting {sleep_time:.1f}s")
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def get(self, url: str, params: dict) -> Optional[dict]:
        """Make GET request with retry logic."""
        if config.STACKOVERFLOW_API_KEY:
            params["key"] = config.STACKOVERFLOW_API_KEY
        
        for attempt in range(config.MAX_RETRIES):
            self.wait_for_rate_limit()
            self.total_requests += 1
            
            try:
                response = requests.get(url, params=params, timeout=30)
                
                if response.status_code == 429:
                    self.consecutive_failures += 1
                    self.failed_requests += 1
                    backoff = min(
                        config.API_DELAY * (config.BACKOFF_MULTIPLIER ** self.consecutive_failures),
                        config.MAX_BACKOFF
                    )
                    logger.warning(f"Rate limited! Waiting {backoff:.0f}s (attempt {attempt + 1})")
                    time.sleep(backoff)
                    continue
                
                response.raise_for_status()
                self.consecutive_failures = 0
                
                data = response.json()

                if "backoff" in data:
                    logger.info("API requested backoff of %ss", data["backoff"])
                    time.sleep(data["backoff"])

                remaining = data.get("quota_remaining", "unknown")
                if remaining != "unknown":
                    remaining = int(remaining)
                    if remaining == 0:
                        raise RuntimeError(
                            "StackExchange API daily quota exhausted (0 remaining). "
                            "Set STACKOVERFLOW_API_KEY for a higher quota, or retry tomorrow."
                        )
                    if remaining < 100:
                        logger.warning("Low API quota: %d remaining", remaining)
                
                return data
                
            except requests.exceptions.Timeout:
                logger.error(f"Request timeout (attempt {attempt + 1})")
                self.failed_requests += 1
            except requests.exceptions.ConnectionError as e:
                logger.error(f"Connection error: {e}")
                self.failed_requests += 1
            except requests.exceptions.RequestException as e:
                logger.error(f"Request failed: {e}")
                self.failed_requests += 1
            
            if attempt < config.MAX_RETRIES - 1:
                time.sleep(config.API_DELAY * (config.BACKOFF_MULTIPLIER ** attempt))
        
        return None
    
    def get_stats(self) -> dict:
        """Return request statistics."""
        return {
            "total_requests": self.total_requests,
            "failed_requests": self.failed_requests,
            "success_rate": (self.total_requests - self.failed_requests) / max(self.total_requests, 1)
        }


def clean_html(text: str) -> str:
    """Remove HTML tags and decode entities."""
    if not text:
        return ""
    text = re.sub(r'<[^>]+>', ' ', text)
    text = unescape(text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def extract_error_signatures(text: str) -> list:
    """Extract error patterns from text."""
    found = []
    for name, pattern in ERROR_PATTERNS.items():
        if re.search(pattern, text, re.IGNORECASE):
            found.append(name)
    return found


def extract_infra_components(text: str) -> list:
    """Identify infrastructure components."""
    found = []
    for name, pattern in INFRA_PATTERNS.items():
        if re.search(pattern, text, re.IGNORECASE):
            found.append(name)
    return found


def load_questions_from_csv() -> list:
    """Load questions from local CSV file."""
    logger.info(f"Loading questions from CSV: {config.CSV_QUESTIONS_PATH}")
    
    if not config.CSV_QUESTIONS_PATH.exists():
        raise FileNotFoundError(f"CSV file not found: {config.CSV_QUESTIONS_PATH}")
    
    questions = []
    with open(config.CSV_QUESTIONS_PATH, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                score = int(row.get("Score", 0) or 0)
                accepted_id = row.get("AcceptedAnswerId", "").strip()
                
                if score >= config.MIN_SCORE and accepted_id:
                    tags_raw = row.get("Tags", "")
                    tags = re.findall(r'<([^>]+)>', tags_raw) if tags_raw else []
                    
                    questions.append({
                        "question_id": int(row["Id"]),
                        "title": clean_html(row.get("Title", "")),
                        "body": clean_html(row.get("Body", "")),
                        "score": score,
                        "accepted_answer_id": int(accepted_id),
                        "tags": tags,
                        "view_count": int(row.get("ViewCount", 0) or 0),
                        "creation_date": row.get("CreationDate", ""),
                    })
            except (ValueError, KeyError) as e:
                logger.debug(f"Skipping malformed row: {e}")
                continue
    
    logger.info(f"Loaded {len(questions)} questions from CSV")
    return questions


def load_answers_from_csv() -> dict:
    """Load answers from local CSV into lookup dict."""
    logger.info(f"Loading answers from CSV: {config.CSV_ANSWERS_PATH}")
    
    if not config.CSV_ANSWERS_PATH.exists():
        raise FileNotFoundError(f"CSV file not found: {config.CSV_ANSWERS_PATH}")
    
    answers = {}
    with open(config.CSV_ANSWERS_PATH, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                answer_id = int(row.get("AnswerId", 0) or 0)
                question_id = int(row.get("QuestionId", 0) or 0)
                
                if answer_id and question_id:
                    answers[question_id] = {
                        "answer_id": answer_id,
                        "body": clean_html(row.get("AnswerBody", "")),
                        "score": int(row.get("AnswerScore", 0) or 0),
                    }
            except (ValueError, KeyError) as e:
                logger.debug(f"Skipping malformed answer row: {e}")
                continue
    
    logger.info(f"Loaded {len(answers)} answers from CSV")
    return answers


def fetch_questions_from_api(requester: RateLimitedRequester, max_questions: int = 0) -> list:
    """Fetch questions from StackOverflow API with pagination.

    Args:
        requester: Rate-limited HTTP client.
        max_questions: Stop after collecting this many questions.
            0 means no cap (fetch all available pages for every tag).
    """
    all_questions = []
    seen_ids = set()
    page_size = 100

    for tag in config.TAGS:
        if max_questions and len(all_questions) >= max_questions:
            break

        page = 1
        logger.info(f"Fetching questions for tag: {tag}")

        while True:
            data = requester.get(
                "https://api.stackexchange.com/2.3/questions",
                {
                    "order": "desc",
                    "sort": "votes",
                    "tagged": tag,
                    "site": "stackoverflow",
                    "pagesize": page_size,
                    "page": page,
                    "filter": "withbody",
                },
            )

            if not data:
                logger.warning(f"Failed to fetch page {page} for {tag}")
                break

            items = data.get("items", [])
            for item in items:
                qid = item["question_id"]
                if qid not in seen_ids and item.get("accepted_answer_id"):
                    seen_ids.add(qid)
                    all_questions.append(item)

            logger.info(f"  Tag={tag} page={page}: {len(items)} items, total unique={len(all_questions)}")

            if max_questions and len(all_questions) >= max_questions:
                break
            if not data.get("has_more", False):
                break

            page += 1

    if max_questions and len(all_questions) > max_questions:
        all_questions = all_questions[:max_questions]

    logger.info(f"Total unique questions fetched: {len(all_questions)}")
    return all_questions


def fetch_answers_batch(requester: RateLimitedRequester, answer_ids: list) -> dict:
    """Fetch answers in batches from API."""
    fetched = {}
    
    for i in range(0, len(answer_ids), config.BATCH_SIZE):
        batch = answer_ids[i:i + config.BATCH_SIZE]
        batch_str = ";".join(str(aid) for aid in batch)
        
        logger.info(f"Fetching batch {i // config.BATCH_SIZE + 1} ({len(batch)} answers)")
        
        data = requester.get(
            f"https://api.stackexchange.com/2.3/answers/{batch_str}",
            {
                "order": "desc",
                "sort": "votes",
                "site": "stackoverflow",
                "filter": "withbody"
            }
        )
        
        if data and "items" in data:
            for item in data["items"]:
                fetched[item["answer_id"]] = {
                    "body": clean_html(item.get("body", "")),
                    "score": item.get("score", 0)
                }
            logger.info(f"Got {len(data['items'])} answers")
        else:
            logger.warning("Batch request failed")
    
    return fetched


def run_acquisition(use_csv: bool = False, max_questions: int = 0) -> dict:
    """
    Main acquisition function - entry point for Airflow task.

    Args:
        use_csv: If True, load from local CSV instead of API.
        max_questions: Cap on questions to fetch via API (0 = no cap).

    Returns:
        dict: Statistics about the acquisition process
    """
    logger.info("=" * 60)
    logger.info("STARTING DATA ACQUISITION")
    logger.info(f"Mode: {'CSV' if use_csv else 'API'}")
    logger.info("=" * 60)
    
    start_time = datetime.now()
    stats = {"source": "csv" if use_csv else "api", "start_time": start_time.isoformat()}
    
    try:
        if use_csv:
            questions = load_questions_from_csv()
            csv_answers = load_answers_from_csv()
            
            combined = []
            for q in questions:
                qid = q["question_id"]
                if qid in csv_answers:
                    combined.append({
                        "question_id": qid,
                        "title": q["title"],
                        "question_body": q["body"],
                        "tags": q["tags"],
                        "score": q["score"],
                        "view_count": q["view_count"],
                        "answer_body": csv_answers[qid]["body"],
                        "answer_score": csv_answers[qid]["score"],
                    })
            
            stats["questions_loaded"] = len(questions)
            stats["answers_matched"] = len(combined)
            
        else:
            requester = RateLimitedRequester()
            
            api_questions = fetch_questions_from_api(requester, max_questions=max_questions)
            
            with open(config.raw_dir / "questions_raw.json", "w", encoding="utf-8") as f:
                json.dump({"items": api_questions}, f, indent=2)
            
            answer_ids = [q["accepted_answer_id"] for q in api_questions]
            fetched_answers = fetch_answers_batch(requester, answer_ids)
            
            combined = []
            for q in api_questions:
                aid = q.get("accepted_answer_id")
                if aid and aid in fetched_answers:
                    combined.append({
                        "question_id": q["question_id"],
                        "title": clean_html(q.get("title", "")),
                        "question_body": clean_html(q.get("body", "")),
                        "tags": q.get("tags", []),
                        "score": q.get("score", 0),
                        "view_count": q.get("view_count", 0),
                        "answer_body": fetched_answers[aid]["body"],
                        "answer_score": fetched_answers[aid]["score"],
                    })
            
            api_stats = requester.get_stats()
            stats.update(api_stats)
            stats["questions_fetched"] = len(api_questions)
            stats["answers_matched"] = len(combined)
        
        output_path = config.raw_dir / "qa_pairs_raw.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(combined, f, indent=2)
        
        stats["output_file"] = str(output_path)
        stats["total_pairs"] = len(combined)
        stats["status"] = "success"
        
        logger.info(f"Successfully saved {len(combined)} Q&A pairs to {output_path}")
        
    except Exception as e:
        logger.error(f"Acquisition failed: {e}")
        stats["status"] = "failed"
        stats["error"] = str(e)
        raise
    
    finally:
        stats["end_time"] = datetime.now().isoformat()
        stats["duration_seconds"] = (datetime.now() - start_time).total_seconds()
        
        stats_path = config.LOGS_DIR / "acquisition_stats.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Acquisition completed in {stats['duration_seconds']:.1f}s")
    
    return stats


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-csv", action="store_true")
    args = parser.parse_args()
    
    run_acquisition(use_csv=args.use_csv)
