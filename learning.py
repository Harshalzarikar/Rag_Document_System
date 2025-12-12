import hashlib
import os

# 1. Simulate our "Database" (This usually lives in ChromaDB)
# We store the filenames we have already processed.
processed_files_db = {
    "report_2023.txt": "hash_of_content_123",
    "notes.txt": "hash_of_content_456"
}

def get_file_hash(content):
    """Creates a unique fingerprint for the file content"""
    return hashlib.md5(content.encode()).hexdigest()

def process_document(filename, content):
    """Simulates the heavy work of embedding and indexing"""
    print(f"⚙️ PROCESSING: {filename} (Embedding & Indexing...)")
    # In real life, this is where index.insert() happens

def upload_manager(filename, content):
    print(f"\n--- Attempting to upload: {filename} ---")
    
    # Step 1: Generate Hash of the new content
    new_hash = get_file_hash(content)
    
    # Step 2: Check if file exists in our DB
    if filename in processed_files_db:
        existing_hash = processed_files_db[filename]
        
        # Step 3: Check if the content is exactly the same
        if new_hash == existing_hash:
            print(f"⏭️ SKIPPING: {filename} (Already exists and hasn't changed)")
            return
        else:
            print(f"📝 UPDATING: {filename} (File exists but content changed)")
    
    # Step 4: If new or changed, process it
    process_document(filename, content)
    
    # Step 5: Update the DB
    processed_files_db[filename] = new_hash
    print("✅ DATABASE UPDATED")

# --- TEST RUN ---

# Case 1: Upload a file that already exists (Skip)
upload_manager("report_2023.txt", "This is the content of report 2023") 
# (Note: In a real app, we'd read the file content from disk)

# Case 2: Upload a BRAND NEW file (Process)
upload_manager("new_strategy.pdf", "This is brand new content")

# Case 3: Upload an EXISTING file but with CHANGED content (Update)
upload_manager("notes.txt", "This is the EDITED content of notes")