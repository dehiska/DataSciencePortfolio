# download_uszipcode_db.py
"""
import os # Import os module to remove potentially problematic files

# Define the expected database path
uszipcode_dir = os.path.join(os.path.expanduser("~"), ".uszipcode")
simple_db_path = os.path.join(uszipcode_dir, "simple_db.sqlite")

print(f"Checking for existing database at: {simple_db_path}")

# Optional: Remove existing (potentially corrupt/empty) database files
# This ensures a fresh download attempt
if os.path.exists(simple_db_path):
    print(f"Found existing simple_db.sqlite. Removing to force fresh download...")
    os.remove(simple_db_path)
    # Also remove the comprehensive one if it exists, to be thorough
    comp_db_path = os.path.join(uszipcode_dir, "comprehensive_db.sqlite")
    if os.path.exists(comp_db_path):
        print(f"Found existing comprehensive_db.sqlite. Removing...")
        os.remove(comp_db_path)


print("Attempting to initialize SearchEngine and download database...")
try:
    # Initialize SearchEngine without the 'simple_zipcode' argument
    # This should trigger the default behavior, which includes downloading if needed
    search = SearchEngine()
    print("SearchEngine initialized successfully. Database should be downloaded.")

    # You can optionally test a search to confirm the database is working
    res = search.by_zipcode("90210")
    if res:
        print(f"Test search for 90210 successful: {res.major_city}, {res.state}")
    else:
        print("Test search for 90210 failed (zipcode not found or database issue).")

except Exception as e:
    print(f"An error occurred during database initialization: {e}")

"""