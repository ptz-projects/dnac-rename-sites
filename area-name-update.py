import aiohttp
import asyncio
import pandas as pd
import logging
import os
import json
from typing import Dict, List, Optional
from dotenv import load_dotenv
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('catalyst_center_rename.log'),
        logging.StreamHandler()
    ]
)


class RateLimiter:
    """Rate limiter to ensure we don't exceed Catalyst Center API limits"""
    
    def __init__(self, max_calls_per_minute: int = 80):
        """
        Initialize rate limiter
        
        Args:
            max_calls_per_minute: Maximum API calls per minute (default 80 for 20% safety margin)
        """
        self.max_calls = max_calls_per_minute
        self.calls = []
        self.lock = asyncio.Lock()
    
    async def acquire(self):
        """Wait if necessary to stay within rate limits"""
        async with self.lock:
            now = time.time()
            
            # Remove calls older than 1 minute
            self.calls = [call_time for call_time in self.calls if now - call_time < 60]
            
            # If at limit, wait until oldest call expires
            if len(self.calls) >= self.max_calls:
                sleep_time = 60 - (now - self.calls[0]) + 0.1  # Add 100ms buffer
                logging.warning(f"Rate limit reached. Waiting {sleep_time:.1f} seconds...")
                await asyncio.sleep(sleep_time)
                
                # Clean up again after waiting
                now = time.time()
                self.calls = [call_time for call_time in self.calls if now - call_time < 60]
            
            # Record this call
            self.calls.append(now)
            logging.debug(f"Rate limiter: {len(self.calls)}/{self.max_calls} calls in last minute")


class CatalystCenterAsyncClient:
    def __init__(self, base_url: str, username: str, password: str, max_concurrent: int = 5):
        """
        Initialize Catalyst Center async API client
        
        Args:
            base_url: Base URL of Catalyst Center
            username: API username
            password: API password
            max_concurrent: Maximum number of concurrent requests (default: 5)
        """
        self.base_url = base_url.rstrip('/')
        self.username = username
        self.password = password
        self.token = None
        self.headers = {'Content-Type': 'application/json'}
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.rate_limiter = RateLimiter(max_calls_per_minute=80)  # 80 calls/min for safety
        self.retry_attempts = 3
        self.base_retry_delay = 2  # seconds
        
    async def authenticate(self) -> bool:
        """Authenticate and get access token"""
        try:
            auth_url = f"{self.base_url}/dna/system/api/v1/auth/token"
            
            # Create basic auth
            auth = aiohttp.BasicAuth(self.username, self.password)
            
            # Acquire rate limit token
            await self.rate_limiter.acquire()
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    auth_url,
                    auth=auth,
                    headers=self.headers,
                    ssl=False,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    response.raise_for_status()
                    data = await response.json()
                    
                    self.token = data['Token']
                    self.headers['X-Auth-Token'] = self.token
                    logging.info("✓ Successfully authenticated to Catalyst Center")
                    return True
                    
        except aiohttp.ClientError as e:
            logging.error(f"✗ Authentication failed: {e}")
            return False
        except Exception as e:
            logging.error(f"✗ Unexpected error during authentication: {e}")
            return False
    
    def get_parent_hierarchy(self, site_name_hierarchy: str) -> str:
        """
        Extract parent hierarchy by removing everything after the last /
        
        Args:
            site_name_hierarchy: Full hierarchy path (e.g., "Global/Area1/Area2/AreaName")
            
        Returns:
            Parent hierarchy path (e.g., "Global/Area1/Area2")
        """
        if not site_name_hierarchy:
            logging.warning("Empty hierarchy provided, defaulting to 'Global'")
            return "Global"
        
        site_name_hierarchy = site_name_hierarchy.strip()
        
        # Find the last occurrence of /
        last_slash_index = site_name_hierarchy.rfind('/')
        
        if last_slash_index == -1:
            # No slash found, this is a top-level area under Global
            logging.debug(f"No slash found in hierarchy: {site_name_hierarchy}")
            return "Global"
        
        # Extract everything before the last slash
        parent_hierarchy = site_name_hierarchy[:last_slash_index]
        
        if not parent_hierarchy:
            parent_hierarchy = "Global"
        
        logging.debug(f"Parent: '{parent_hierarchy}' from '{site_name_hierarchy}'")
        return parent_hierarchy
    
    def get_parent_id_from_hierarchy(self, all_areas: List[Dict], parent_hierarchy: str) -> Optional[str]:
        """
        Get parent area ID by matching the parent hierarchy path
        
        Args:
            all_areas: List of all areas with their hierarchies
            parent_hierarchy: Parent hierarchy path (e.g., "Global/Area1/Area2")
            
        Returns:
            Parent area ID or None if not found
        """
        # Special case: if parent is "Global", find Global's ID
        if parent_hierarchy == "Global":
            for area in all_areas:
                if area['name'] == 'Global' and area['nameHierarchy'] == 'Global':
                    logging.debug(f"Found Global parent ID: {area['id']}")
                    return area['id']
            logging.warning("Global parent requested but Global ID not found")
            return None
        
        # Find the area that matches the parent hierarchy
        for area in all_areas:
            if area['nameHierarchy'] == parent_hierarchy:
                logging.debug(f"Found parent ID for '{parent_hierarchy}': {area['id']}")
                return area['id']
        
        logging.error(f"Could not find parent ID for hierarchy: {parent_hierarchy}")
        return None
    
    def is_area(self, site: Dict) -> bool:
        """
        Check if a site is an area by examining additionalInfo
        
        Args:
            site: Site dictionary from API response
            
        Returns:
            True if site is an area, False otherwise
        """
        # Check additionalInfo for type = "area"
        if 'additionalInfo' in site:
            for info in site.get('additionalInfo', []):
                if info.get('nameSpace') == 'Location':
                    attributes = info.get('attributes', {})
                    if attributes.get('type') == 'area':
                        return True
        
        # Fallback: if additionalInfo is empty, check if it's not Global
        if not site.get('additionalInfo') and site.get('name') != 'Global':
            site_hierarchy = site.get('siteNameHierarchy', '')
            if '/' in site_hierarchy:
                return True
        
        return False
    
    async def get_all_areas_from_hierarchy(self, session: aiohttp.ClientSession) -> List[Dict]:
        """
        Get all areas using the site-hierarchy endpoint
        
        Args:
            session: aiohttp ClientSession
            
        Returns:
            List of area dictionaries with id, name, and hierarchy (includes Global)
        """
        all_areas = []
        
        try:
            url = f"{self.base_url}/dna/intent/api/v1/site/site-hierarchy"
            
            logging.info("Fetching site hierarchy (includes all areas)...")
            
            # Acquire rate limit token
            await self.rate_limiter.acquire()
            
            async with session.get(
                url,
                headers=self.headers,
                ssl=False,
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                response.raise_for_status()
                data = await response.json()
                
                # Parse the flat list response
                if 'response' in data and isinstance(data['response'], list):
                    sites = data['response']
                    
                    logging.info(f"Found {len(sites)} total sites in hierarchy")
                    
                    for site in sites:
                        site_name = site.get('name', '')
                        site_id = site.get('id', '')
                        site_name_hierarchy = site.get('siteNameHierarchy', '')
                        
                        # Include Global (needed for parent ID lookups)
                        if site_name == 'Global':
                            area_info = {
                                'id': site_id,
                                'name': site_name,
                                'nameHierarchy': site_name_hierarchy
                            }
                            all_areas.append(area_info)
                            logging.debug(f"Found Global: ID={site_id}")
                            continue
                        
                        # Check if this is an area
                        if self.is_area(site):
                            area_info = {
                                'id': site_id,
                                'name': site_name,
                                'nameHierarchy': site_name_hierarchy
                            }
                            all_areas.append(area_info)
                            logging.debug(f"Found area: {site_name} (ID: {site_id}, Hierarchy: {site_name_hierarchy})")
                        else:
                            logging.debug(f"Skipping non-area site: {site_name}")
                    
                    logging.info(f"Total areas extracted: {len(all_areas)} (includes Global)")
                    
                    # Log first few areas for verification
                    if all_areas:
                        logging.info("Sample areas:")
                        for idx, area in enumerate(all_areas[:5]):
                            logging.info(f"  {idx+1}. {area['name']} - {area['nameHierarchy']}")
                    
                    return all_areas
                else:
                    logging.error(f"Unexpected response structure. Keys: {data.keys()}")
                    return all_areas
                    
        except aiohttp.ClientError as e:
            logging.error(f"Error fetching site hierarchy: {e}")
            return all_areas
        except Exception as e:
            logging.error(f"Unexpected error fetching site hierarchy: {e}")
            logging.exception("Full traceback:")
            return all_areas
    
    async def update_area_name(
        self, 
        session: aiohttp.ClientSession, 
        area_id: str, 
        current_name: str, 
        new_name: str, 
        name_hierarchy: str,
        all_areas: List[Dict]
    ) -> tuple:
        """
        Update area name with rate limiting and exponential backoff retry
        
        Args:
            session: aiohttp ClientSession
            area_id: Area UUID
            current_name: Current area name
            new_name: New name for the area
            name_hierarchy: Full nameHierarchy path
            all_areas: List of all areas (needed to find parent ID)
            
        Returns:
            Tuple of (success: bool, area_id: str, current_name: str, new_name: str)
        """
        async with self.semaphore:  # Control concurrency
            
            # Validate we have the hierarchy
            if not name_hierarchy:
                logging.error(f"No nameHierarchy provided for area {area_id} ({current_name})")
                return (False, area_id, current_name, new_name)
            
            # Extract parent hierarchy path
            parent_hierarchy = self.get_parent_hierarchy(name_hierarchy)
            
            # Get parent ID from hierarchy
            parent_id = self.get_parent_id_from_hierarchy(all_areas, parent_hierarchy)
            
            if not parent_id:
                logging.error(f"Could not find parent ID for area {area_id} ({current_name})")
                logging.error(f"  Parent hierarchy: {parent_hierarchy}")
                return (False, area_id, current_name, new_name)
            
            url = f"{self.base_url}/dna/intent/api/v1/areas/{area_id}"
            
            # Prepare headers
            update_headers = self.headers.copy()
            update_headers['__runsync'] = 'false'
            update_headers['__timeout'] = '30'
            update_headers['__persistbapioutput'] = 'true'
            
            # Prepare payload
            payload = {
                "name": new_name,
                "parentId": parent_id
            }
            
            logging.info(f"Updating area: {current_name} → {new_name}")
            logging.info(f"  Area ID: {area_id}")
            logging.info(f"  Parent ID: {parent_id}")
            
            # Retry loop with exponential backoff
            for attempt in range(1, self.retry_attempts + 1):
                try:
                    # Acquire rate limit token
                    await self.rate_limiter.acquire()
                    
                    async with session.put(
                        url,
                        headers=update_headers,
                        json=payload,
                        ssl=False,
                        timeout=aiohttp.ClientTimeout(total=30)
                    ) as response:
                        response_text = await response.text()
                        
                        # Handle 429 Too Many Requests
                        if response.status == 429:
                            retry_after = int(response.headers.get('Retry-After', 60))
                            logging.warning(f"⚠️  Rate limited (429). Waiting {retry_after} seconds...")
                            await asyncio.sleep(retry_after)
                            continue  # Retry
                        
                        # Handle other errors
                        if response.status not in [200, 202]:
                            if attempt < self.retry_attempts:
                                backoff_delay = self.base_retry_delay * (2 ** (attempt - 1))
                                logging.warning(f"⚠️  HTTP {response.status} on attempt {attempt}. Retrying in {backoff_delay}s...")
                                await asyncio.sleep(backoff_delay)
                                continue
                            else:
                                logging.error(f"✗ HTTP {response.status} Error updating area {area_id} ({current_name})")
                                logging.error(f"  Response: {response_text}")
                                return (False, area_id, current_name, new_name)
                        
                        response.raise_for_status()
                        
                        try:
                            result = json.loads(response_text)
                        except:
                            result = {"response": response_text}
                        
                        logging.info(f"✓ Successfully updated area {area_id}")
                        
                        # Rate limiting between successful calls
                        await asyncio.sleep(1)
                        return (True, area_id, current_name, new_name)
                        
                except aiohttp.ClientError as e:
                    if attempt < self.retry_attempts:
                        backoff_delay = self.base_retry_delay * (2 ** (attempt - 1))
                        logging.warning(f"⚠️  Error on attempt {attempt}/{self.retry_attempts}: {e}")
                        logging.warning(f"  Retrying in {backoff_delay} seconds...")
                        await asyncio.sleep(backoff_delay)
                    else:
                        logging.error(f"✗ Failed after {self.retry_attempts} attempts: {e}")
                        return (False, area_id, current_name, new_name)
                except Exception as e:
                    logging.error(f"✗ Unexpected error: {e}")
                    logging.exception("Full traceback:")
                    return (False, area_id, current_name, new_name)
            
            return (False, area_id, current_name, new_name)
    
    async def test_single_area_update(
        self, 
        session: aiohttp.ClientSession,
        area: Dict,
        new_name: str,
        all_areas: List[Dict]
    ) -> bool:
        """Test updating a single area"""
        logging.info("\n" + "="*80)
        logging.info("TESTING SINGLE AREA UPDATE")
        logging.info("="*80)
        
        area_id = area['id']
        current_name = area['name']
        name_hierarchy = area['nameHierarchy']
        
        logging.info(f"Test Area: {current_name}")
        logging.info(f"  ID: {area_id}")
        logging.info(f"  Hierarchy: {name_hierarchy}")
        logging.info(f"  New Name: {new_name}")
        
        success, _, _, _ = await self.update_area_name(
            session, area_id, current_name, new_name, name_hierarchy, all_areas
        )
        
        if success:
            logging.info("✓ Test update successful!")
        else:
            logging.error("✗ Test update failed!")
        
        logging.info("="*80 + "\n")
        return success
    
    async def update_areas_batch(
        self, 
        areas_to_update: List[Dict], 
        name_mapping: Dict[str, str],
        all_areas: List[Dict]
    ) -> Dict:
        """Update multiple areas with rate limiting"""
        results = {
            'successful_updates': 0,
            'failed_updates': 0,
            'total_processed': 0
        }
        
        connector = aiohttp.TCPConnector(ssl=False)
        
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = []
            
            for area in areas_to_update:
                area_id = area['id']
                current_name = area['name']
                new_name = name_mapping.get(current_name)
                name_hierarchy = area['nameHierarchy']
                
                if new_name and new_name != current_name:
                    task = self.update_area_name(
                        session, area_id, current_name, new_name, name_hierarchy, all_areas
                    )
                    tasks.append(task)
            
            if not tasks:
                logging.warning("No areas to update")
                return results
            
            # Calculate estimated time
            estimated_time = len(tasks) / (80 / 60)  # 80 calls per minute
            logging.info(f"\nStarting async update of {len(tasks)} areas")
            logging.info(f"  Max concurrent: {self.max_concurrent}")
            logging.info(f"  Rate limit: 80 calls/minute")
            logging.info(f"  Estimated time: {estimated_time:.1f} minutes")
            
            completed = 0
            for coro in asyncio.as_completed(tasks):
                success, area_id, current_name, new_name = await coro
                completed += 1
                
                if success:
                    results['successful_updates'] += 1
                else:
                    results['failed_updates'] += 1
                
                results['total_processed'] = completed
                
                if completed % 10 == 0 or completed == len(tasks):
                    logging.info(f"Progress: {completed}/{len(tasks)} areas processed")
            
            return results


# [Rest of the code remains the same: load_mapping_spreadsheet, get_credentials_from_env, main_async, main]
# ... (keeping the previous implementations)

def load_mapping_spreadsheet(file_path: str) -> Dict[str, str]:
    """
    Load spreadsheet with current and target names
    
    Args:
        file_path: Path to Excel/CSV file
        
    Returns:
        Dictionary mapping current names to target names
    """
    try:
        # Support both Excel and CSV
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
        
        # Assuming Column A is current name, Column B is target name
        if df.shape[1] < 2:
            logging.error("Spreadsheet must have at least 2 columns")
            return {}
        
        # Use first two columns
        df = df.iloc[:, :2]
        df.columns = ['current_name', 'target_name']
        
        # Remove rows with NaN values
        df = df.dropna()
        
        # Create dictionary
        mapping = dict(zip(df['current_name'].str.strip(), df['target_name'].str.strip()))
        
        logging.info(f"Loaded {len(mapping)} name mappings from spreadsheet")
        
        # Log first few mappings for verification
        if mapping:
            logging.info("Sample mappings:")
            for idx, (curr, target) in enumerate(list(mapping.items())[:3]):
                logging.info(f"  {idx+1}. '{curr}' → '{target}'")
        
        return mapping
        
    except Exception as e:
        logging.error(f"Error loading spreadsheet: {e}")
        return {}


def get_credentials_from_env() -> tuple:
    """
    Retrieve credentials from environment variables
    
    Returns:
        Tuple of (base_url, username, password, max_concurrent)
    """
    # Load environment variables from .env file
    load_dotenv()
    
    base_url = os.getenv('CATALYST_CENTER_URL')
    username = os.getenv('CATALYST_USERNAME')
    password = os.getenv('CATALYST_PASSWORD')
    max_concurrent = int(os.getenv('MAX_CONCURRENT_REQUESTS', '10'))
    
    # Validate that all required variables are present
    if not all([base_url, username, password]):
        missing = []
        if not base_url:
            missing.append('CATALYST_CENTER_URL')
        if not username:
            missing.append('CATALYST_USERNAME')
        if not password:
            missing.append('CATALYST_PASSWORD')
        
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")
    
    logging.info(f"Configuration loaded:")
    logging.info(f"  URL: {base_url}")
    logging.info(f"  Max concurrent requests: {max_concurrent}")
    
    return base_url, username, password, max_concurrent


async def main_async():
    """Main async execution function"""
    
    # Initialize results tracking
    results = {
        'total_sites': 0,
        'total_areas': 0,
        'areas_to_update': 0,
        'successful_updates': 0,
        'failed_updates': 0,
        'skipped': 0
    }
    
    try:
        logging.info("="*80)
        logging.info("Catalyst Center Area Rename Script (Async)")
        logging.info("="*80)
        
        # Load credentials from environment variables
        logging.info("\nLoading configuration from environment variables...")
        CATALYST_CENTER_URL, USERNAME, PASSWORD, MAX_CONCURRENT = get_credentials_from_env()
        
        # Get spreadsheet path from environment or use default
        SPREADSHEET_PATH = os.getenv('SPREADSHEET_PATH', 'area_mapping.xlsx')
        logging.info(f"  Spreadsheet path: {SPREADSHEET_PATH}")
        
        # Check for test mode
        TEST_MODE = os.getenv('TEST_MODE', 'false').lower() == 'true'
        if TEST_MODE:
            logging.info("  ⚠️  TEST MODE ENABLED - Will only update first matching area")
        
    except ValueError as e:
        logging.error(f"Configuration error: {e}")
        logging.error("Please ensure your .env file is properly configured")
        return
    except Exception as e:
        logging.error(f"Unexpected error loading configuration: {e}")
        return
    
    # Load name mapping from spreadsheet
    logging.info("\nLoading name mapping from spreadsheet...")
    name_mapping = load_mapping_spreadsheet(SPREADSHEET_PATH)
    
    if not name_mapping:
        logging.error("No valid mappings found in spreadsheet. Exiting.")
        return
    
    # Initialize Catalyst Center async client with values from environment
    logging.info("\nInitializing Catalyst Center async client...")
    client = CatalystCenterAsyncClient(
        CATALYST_CENTER_URL, 
        USERNAME, 
        PASSWORD,
        max_concurrent=MAX_CONCURRENT
    )
    
    # Authenticate
    logging.info("\nAuthenticating to Catalyst Center...")
    if not await client.authenticate():
        logging.error("Failed to authenticate. Exiting.")
        return
    
    # Get all areas from site-hierarchy
    logging.info("\nFetching all areas from site-hierarchy endpoint...")
    connector = aiohttp.TCPConnector(ssl=False)
    
    async with aiohttp.ClientSession(connector=connector) as session:
        all_areas = await client.get_all_areas_from_hierarchy(session)
    
    results['total_areas'] = len(all_areas)
    
    if not all_areas:
        logging.error("No areas retrieved from hierarchy.")
        logging.error("This might mean:")
        logging.error("  1. All sites are buildings/floors (no areas)")
        logging.error("  2. The API response format is different")
        logging.error("  3. Permission issues")
        return
    
    logging.info(f"\nFound {results['total_areas']} total areas (includes Global)")
    
    # Filter areas that need to be updated (exclude Global from updates)
    areas_to_update = []
    
    for area in all_areas:
        current_name = area['name']
        
        # Skip Global - it cannot be renamed
        if current_name == 'Global':
            continue
        
        if current_name in name_mapping:
            target_name = name_mapping[current_name]
            
            # Skip if names are the same
            if current_name == target_name:
                logging.debug(f"Area '{current_name}' unchanged, skipping")
                results['skipped'] += 1
                continue
            
            # Check if nameHierarchy is available
            if not area.get('nameHierarchy'):
                logging.warning(f"No nameHierarchy found for area '{current_name}', skipping")
                results['failed_updates'] += 1
                continue
            
            areas_to_update.append(area)
            results['areas_to_update'] += 1
        else:
            results['skipped'] += 1
    
    if not areas_to_update:
        logging.info("\nNo areas need to be updated.")
        if results['areas_to_update'] == 0:
            logging.info("None of the areas in Catalyst Center match the names in your spreadsheet.")
            logging.info("\nAreas found in Catalyst Center (excluding Global):")
            non_global_areas = [a for a in all_areas if a['name'] != 'Global']
            for idx, area in enumerate(non_global_areas[:10]):
                logging.info(f"  {idx+1}. {area['name']}")
            if len(non_global_areas) > 10:
                logging.info(f"  ... and {len(non_global_areas) - 10} more")
        return
    
    logging.info(f"\n{len(areas_to_update)} areas will be updated")
    
    # TEST MODE: Only update first area
    if TEST_MODE:
        connector = aiohttp.TCPConnector(ssl=False)
        async with aiohttp.ClientSession(connector=connector) as session:
            test_area = areas_to_update[0]
            test_new_name = name_mapping[test_area['name']]
            
            success = await client.test_single_area_update(session, test_area, test_new_name, all_areas)
            
            if success:
                logging.info("\n✓ Test successful! Set TEST_MODE=false in .env to update all areas.")
            else:
                logging.error("\n✗ Test failed! Fix the issues above before proceeding.")
        return
    
    # Update areas concurrently
    logging.info("\n" + "="*80)
    logging.info("Starting async area updates...")
    logging.info("="*80)
    
    update_results = await client.update_areas_batch(areas_to_update, name_mapping, all_areas)
    
    results['successful_updates'] = update_results['successful_updates']
    results['failed_updates'] += update_results['failed_updates']
    
    # Print summary
    logging.info("\n" + "="*80)
    logging.info("EXECUTION SUMMARY")
    logging.info("="*80)
    logging.info(f"Total areas retrieved:    {results['total_areas']}")
    logging.info(f"Areas in mapping:         {results['areas_to_update']}")
    logging.info(f"Successful updates:       {results['successful_updates']}")
    logging.info(f"Failed updates:           {results['failed_updates']}")
    logging.info(f"Areas skipped:            {results['skipped']}")
    logging.info("="*80)
    
    if results['failed_updates'] > 0:
        logging.warning("\n⚠️  Some updates failed. Check the log file for details.")
    
    if results['successful_updates'] > 0:
        logging.info("\n✓ Area rename operation completed successfully!")


def main():
    """Wrapper to run async main function"""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()