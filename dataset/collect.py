import os
import re
import json
import time
import logging
import requests
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

SPECIES_MAP = {
    "dragons": "dragon",
    "ferrets": "ferret",
    "canines": "canid",
    "felines": "feline",
    "foxes": "fox"
}

RATING_MAP = {
    "safe": "rating:s",
    "questionable": "rating:q",
    "explicit": "rating:e"
}

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

HTML_SEARCH_URL = "https://e621.net/posts"
POSTS_URL = "https://e621.net/posts.json"
HEADERS = {
    "User-Agent": "E6TransCNNCollector/1.0 (contact: CodieTheDragonOne)"
}
TARGET_DIR = Path("raw")

def download_image(url, path):
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        if response.status_code == 200:
            with open(path, "wb") as f:
                f.write(response.content)
            return True
        else:
            logging.error(f"Failed to download image from {url} - Status: {response.status_code}")
    except Exception as e:
        logging.error(f"Error downloading image {url}: {e}")
    return False

def get_active_count(query):
    try:
        params = {"tags": query}
        response = requests.get(HTML_SEARCH_URL, headers=HEADERS, params=params, timeout=10)
        if response.status_code == 200:
            html = response.text
            
            match = re.search(r'data-total-posts="(\d+)"', html)
            if match:
                return int(match.group(1))
                
            match = re.search(r'data-total="(\d+)"', html)
            if match:
                return int(match.group(1))
                
            match = re.search(r'(\d[\d,]*)\s+posts?\s+found', html, re.IGNORECASE)
            if match:
                return int(match.group(1).replace(",", ""))
                
            match = re.search(r'class="post-count">([\d,]+)', html)
            if match:
                return int(match.group(1).replace(",", ""))
        else:
            logging.error(f"Failed to load HTML search page - Status: {response.status_code}")
    except Exception as e:
        logging.error(f"Error querying HTML count for '{query}': {e}")
    return 0

def collect_dataset():
    logging.info("Starting dataset collection pipeline...")
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    for species_name, species_tag in SPECIES_MAP.items():
        logging.info(f"Scanning counts for species: {species_name}")
        counts = {}
        for rating_name, rating_tag in RATING_MAP.items():
            query = f"{species_tag} {rating_tag}"
            count = get_active_count(query)
            if count == 0:
                logging.warning(f"Could not retrieve exact count for '{query}'. Falling back to 1000.")
                count = 1000
            counts[rating_name] = count
            logging.info(f"Query '{query}' resolved to: {count} posts")
            time.sleep(1.0)
        
        total_sum = sum(counts.values())
        average_limit = int(total_sum / 3)
        logging.info(f"Calculated average limit for {species_name}: {average_limit}")
        
        for rating_name, rating_tag in RATING_MAP.items():
            category_dir = TARGET_DIR / species_name / rating_name
            category_dir.mkdir(parents=True, exist_ok=True)
            
            target_limit = min(average_limit, counts[rating_name])
            logging.info(f"Target limit for {species_name} ({rating_name}): {target_limit}")
            if target_limit == 0:
                continue
                
            downloaded_count = 0
            page = 1
            
            while downloaded_count < target_limit:
                query = f"{species_tag} {rating_tag}"
                params = {
                    "tags": query,
                    "limit": 100,
                    "page": page
                }
                try:
                    logging.info(f"Fetching posts for {species_name} ({rating_name}) - Page {page}")
                    response = requests.get(POSTS_URL, headers=HEADERS, params=params, timeout=15)
                    if response.status_code == 429:
                        logging.warning("Rate limit hit (429). Sleeping for 10 seconds...")
                        time.sleep(10)
                        continue
                    elif response.status_code != 200:
                        logging.error(f"Unexpected status code {response.status_code} while fetching posts page {page}")
                        break
                    
                    data = response.json()
                    posts = data.get("posts", [])
                    if not posts:
                        logging.info(f"No more posts returned on page {page}")
                        break
                    
                    for post in posts:
                        if downloaded_count >= target_limit:
                            break
                        
                        post_id = post.get("id")
                        file_ext = post.get("file", {}).get("ext", "").lower()
                        
                        if file_ext not in ALLOWED_EXTENSIONS:
                            logging.info(f"Skipping post {post_id} - unsupported format: {file_ext}")
                            continue
                            
                        file_url = post.get("file", {}).get("url")
                        if not file_url:
                            continue
                            
                        image_path = category_dir / f"{post_id}.{file_ext}"
                        json_path = category_dir / f"{post_id}.json"
                        
                        if image_path.exists() and json_path.exists():
                            downloaded_count += 1
                            continue
                        
                        logging.info(f"Downloading post {post_id} ({downloaded_count + 1}/{target_limit})")
                        if download_image(file_url, image_path):
                            with open(json_path, "w") as f:
                                json.dump(post, f, indent=4)
                            downloaded_count += 1
                            time.sleep(1.0)
                    
                    page += 1
                    time.sleep(1.0)
                except Exception as e:
                    logging.error(f"Error on pagination loop: {e}")
                    time.sleep(5)
                    continue

if __name__ == "__main__":
    collect_dataset()