Before you start
1. If using wsl (Windows Subsystem for Linux), you may need to change the wsl version to 1 as authentication may fail on version 2:
    e.g wsl --set-version Ubuntu 1
    This can take 30 minutes+ to update.
2. Create python venv and pip install -r requirements.txt

Script Execution Steps

1. Configuration & Setup
    Load credentials for .env file:
        Catalyst Center URL
        Username/Password
        Max concurrent requests (default: 10)
        Spreadsheet path
        Test mode flag. Setting to true will execute the first site in the spreadsheet only.
    Update area_mapping.xlsx under the dnac-rename-sites folder. 
    Remove examples and add Name mappings in Excel/CSV spreadsheet 
    (Column A = current name, Column B = new name). Add one row for testing.
    Python code to run: area-name-update.py

2. Authentication

    Connect to Catalyst Center
    Obtain authentication token using Basic Auth
    Store token in headers for subsequent API calls


3. Fetch All Areas

    Retrieve complete site hierarchy from Catalyst Center
    Parse flat list response to extract:
        Area ID (UUID)
        Area name
        Full hierarchy path (e.g., "Global/Region/Area1")
    Filter to include only areas (exclude buildings/floors)
    Include "Global" in list (needed for parent ID lookups)


4. Match & Filter Areas

    Compare Catalyst Center areas against spreadsheet mappings
    Build list of areas to update
    Skip areas where:
        Name is unchanged
        Name not in spreadsheet
        Missing hierarchy information
        Area is "Global" (cannot be renamed)


5. Test Mode (Optional)

    If TEST_MODE=true:
        Update only the first matching area
        Validate API request/response
        Exit after test


6. Bulk Update (Async)

For each area to update:

    Extract parent hierarchy path from full hierarchy (remove everything after last /)
    Look up parent's UUID by matching hierarchy path
    Build PUT request payload with:
        name: new area name
        parentId: parent area UUID
    Send async PUT request with:
        Custom headers (__runsync, __timeout, __persistbapioutput)
        Semaphore to control concurrency (max 10 concurrent)
        0.5 second rate limiting between requests


7. Progress Tracking

    Log each update attempt (success/failure)
    Show progress every 10 areas
    Track statistics:
        Total areas retrieved
        Areas in mapping
        Successful updates
        Failed updates
        Skipped areas


8. Summary Report

    Display execution summary
    Log warnings for failures
    Output to console and log file



API Endpoints Used

1. Authentication



Copy Code

POST /dna/system/api/v1/auth/token


    Purpose: Obtain authentication token
    Auth: Basic Authentication (username/password)
    Response: Returns token for subsequent API calls


2. Get Site Hierarchy



Copy Code

GET /dna/intent/api/v1/site/site-hierarchy


    Purpose: Retrieve all sites/areas with hierarchy information
    Auth: Token-based (X-Auth-Token header)
    Response: Flat list of all sites with:
        id: Site/Area UUID
        name: Site/Area name
        siteNameHierarchy: Full hierarchy path
        additionalInfo: Contains type (area/building/floor)


3. Update Area



Copy Code

PUT /dna/intent/api/v1/areas/{area_id}


    Purpose: Rename an area
    Auth: Token-based (X-Auth-Token header)
    Headers:
        Content-Type: application/json
        __runsync: false (async execution)
        __timeout: 30
        __persistbapioutput: true
    Request Body:
    json
    Copy Code

    {
      "name": "new-area-name",
      "parentId": "parent-uuid"
    }

    Response: 200/202 for success, 400 for validation errors



Key Design Patterns

Async Concurrency

    Uses aiohttp for async HTTP requests
    asyncio.Semaphore limits concurrent requests (default: 10)
    asyncio.as_completed() processes responses as they arrive


Parent ID Resolution

    Builds lookup table: hierarchy_path → area_id
    Strips last segment from hierarchy to get parent path
    Matches parent path to find parent UUID
    Special handling for "Global" parent


Error Handling

    Comprehensive try/catch blocks
    Detailed logging of request/response
    Graceful degradation on failures
    Individual area failures don't stop batch


Rate Limiting

    0.5 second delay between requests
    Prevents API throttling
    Configurable via code modification



Data Flow



Copy Code

1. Load .env + spreadsheet
   ↓
2. POST /auth/token → Get token
   ↓
3. GET /site/site-hierarchy → Get all areas
   ↓
4. Parse & filter areas → Match with spreadsheet
   ↓
5. For each area (async):
   - Extract parent hierarchy
   - Lookup parent ID
   - PUT /areas/{id} with {name, parentId}
   ↓
6. Aggregate results → Display summary



Performance

    Sequential: ~2 seconds per area (100 areas = ~200 seconds)
    Async (10 concurrent): ~0.5 seconds per batch (100 areas = ~50 seconds)
    Speedup: ~4-5x faster with async



Security Features

    Credentials stored in .env (not in code)
    SSL verification disabled (for self-signed certs)
    Token-based authentication
    No credentials logged



Files

    Input: .env, area_mapping.xlsx
    Output: catalyst_center_rename.log
    Script: Single Python file: area-name-update.py
