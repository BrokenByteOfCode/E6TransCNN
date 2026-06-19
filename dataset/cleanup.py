import os
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".PNG", ".JPG", ".JPEG", ".WEBP"}

def run_cleanup(data_dir="raw"):
    data_dir = Path(data_dir)
    if not data_dir.exists() or not data_dir.is_dir():
        logging.error(f"Directory '{data_dir}' does not exist.")
        return

    logging.info("Starting dataset sanitation and cleanup process...")
    
    deleted_media_count = 0
    deleted_orphans_count = 0
    
    all_files = list(data_dir.glob("**/*"))
    files_to_check = [f for f in all_files if f.is_file()]
    
    for file_path in files_to_check:
        ext = file_path.suffix
        
        if ext == ".json":
            continue
            
        if ext not in ALLOWED_EXTENSIONS:
            logging.info(f"Removing unsupported media file: {file_path.name}")
            json_path = file_path.with_suffix(".json")
            
            if file_path.exists():
                file_path.unlink()
                deleted_media_count += 1
                
            if json_path.exists():
                json_path.unlink()
                
    remaining_files = list(data_dir.glob("**/*"))
    active_files = [f for f in remaining_files if f.is_file()]
    
    for file_path in active_files:
        ext = file_path.suffix
        
        if ext == ".json":
            img_found = False
            for allowed_ext in ALLOWED_EXTENSIONS:
                img_path = file_path.with_suffix(allowed_ext)
                if img_path.exists():
                    img_found = True
                    break
            if not img_found:
                logging.info(f"Removing orphan metadata file: {file_path.name}")
                file_path.unlink()
                deleted_orphans_count += 1
                
        elif ext in ALLOWED_EXTENSIONS:
            json_path = file_path.with_suffix(".json")
            if not json_path.exists():
                logging.info(f"Removing orphan image file: {file_path.name}")
                file_path.unlink()
                deleted_orphans_count += 1

    logging.info("Cleanup process complete.")
    logging.info(f"Removed {deleted_media_count} unsupported media files with their JSONs.")
    logging.info(f"Cleaned up {deleted_orphans_count} orphan/incomplete files.")

if __name__ == "__main__":
    import sys
    target_directory = "raw"
    if len(sys.argv) > 1:
        target_directory = sys.argv[1]
    run_cleanup(target_directory)