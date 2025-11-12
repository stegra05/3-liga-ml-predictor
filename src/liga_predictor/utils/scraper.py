"""
Base scraper class for web scraping with requests and Selenium support
Provides common functionality for all collectors to reduce code duplication
"""

import requests
from bs4 import BeautifulSoup
from typing import Optional
from loguru import logger
import time


class BaseScraper:
    """Base class for web scrapers with requests and Selenium support"""

    DEFAULT_USER_AGENT = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'

    def __init__(self, use_selenium: bool = False, persistent_driver: bool = False):
        self.use_selenium = use_selenium
        self.persistent_driver = persistent_driver
        self.session = self._init_session()
        self.driver = None

    def _init_session(self) -> requests.Session:
        session = requests.Session()
        session.headers.update({
            'User-Agent': self.DEFAULT_USER_AGENT,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Cache-Control': 'max-age=0'
        })
        return session

    def _create_chrome_options(self):
        """Create Chrome options with stealth settings"""
        from selenium.webdriver.chrome.options import Options
        chrome_options = Options()
        chrome_options.add_argument('--headless=new')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-blink-features=AutomationControlled')
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        chrome_options.add_argument(f'user-agent={self.session.headers["User-Agent"]}')
        return chrome_options

    def _init_selenium_driver(self) -> None:
        """Initialize Selenium WebDriver with stealth options"""
        if self.driver is not None:
            return
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.service import Service
            from webdriver_manager.chrome import ChromeDriverManager
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=self._create_chrome_options())
        except ImportError:
            logger.error("Selenium not installed. Install with: pip install selenium webdriver-manager")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Selenium: {e}")
            raise

    def _make_request(self, url: str, delay: float = 3.0, wait_for: str = None) -> Optional[BeautifulSoup]:
        """Make HTTP request with error handling and rate limiting"""
        if self.use_selenium:
            return self._make_request_selenium(url, delay, wait_for)
        else:
            return self._make_request_requests(url, delay)

    def _make_request_requests(self, url: str, delay: float) -> Optional[BeautifulSoup]:
        """Make request using requests library"""
        try:
            response = self.session.get(url, timeout=30, allow_redirects=True)
            response.raise_for_status()
            time.sleep(delay)
            return BeautifulSoup(response.content, 'html.parser')
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 403:
                logger.error(f"403 Forbidden: {url} - use use_selenium=True to bypass")
            else:
                logger.error(f"HTTP error {e.response.status_code}: {url}")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {url} - {e}")
            return None

    def _make_request_selenium(self, url: str, delay: float = 0.0, wait_for: str = None) -> Optional[BeautifulSoup]:
        """Make request using Selenium (bypasses bot detection)"""
        try:
            from selenium.webdriver.common.by import By
            from selenium.webdriver.support.ui import WebDriverWait
            from selenium.webdriver.support import expected_conditions as EC

            # Handle persistent vs per-request driver pattern
            if self.persistent_driver:
                if self.driver is None:
                    self._init_selenium_driver()
                driver = self.driver
                should_quit = False
            else:
                from selenium import webdriver
                from selenium.webdriver.chrome.service import Service
                from webdriver_manager.chrome import ChromeDriverManager
                service = Service(ChromeDriverManager().install())
                driver = webdriver.Chrome(service=service, options=self._create_chrome_options())
                should_quit = True

            try:
                driver.get(url)
                if wait_for:
                    # If wait_for is a simple tag name (no CSS syntax), treat as tag
                    by = By.TAG_NAME if wait_for.isalnum() else By.CSS_SELECTOR
                    WebDriverWait(driver, 15).until(EC.presence_of_element_located((by, wait_for)))
                    if by == By.TAG_NAME:
                        time.sleep(2)  # Extra wait for tag-based waits
                else:
                    time.sleep(3)
                page_source = driver.page_source
                if delay > 0:
                    time.sleep(delay)
                return BeautifulSoup(page_source, 'html.parser')
            finally:
                if should_quit and driver:
                    driver.quit()
        except ImportError:
            logger.error("Selenium not installed. Install with: pip install selenium webdriver-manager")
            return None
        except Exception as e:
            logger.error(f"Selenium request failed: {url} - {e}")
            return None

    def close(self):
        """Clean up resources (close persistent Selenium driver if exists)"""
        if self.driver:
            try:
                self.driver.quit()
                self.driver = None
            except:
                pass

    def __del__(self):
        self.close()

