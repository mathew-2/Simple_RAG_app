# test_csv.py
import csv

with open("embeddings/chunks.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    
    print("=" * 60)
    print("CHECKING CSV FILE")
    print("=" * 60)
    
    for i, row in enumerate(reader):
        if i < 3:  # Show first 3 rows
            print(f"\n--- Row {i} ---")
            print(f"Chunk ID: {row['chunk_id']}")
            print(f"Page: {row['page_number']}")
            print(f"Text length: {len(row['text'])} chars")
            print(f"Text preview: {row['text'][:200]}...")
            print(f"Has embedding: {len(row['embedding']) > 0}")
        else:
            break