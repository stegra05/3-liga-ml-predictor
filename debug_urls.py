"""
Debug script to inspect actual URLs and HTML responses
"""
import sys
from pathlib import Path
import requests
from bs4 import BeautifulSoup

sys.path.append(str(Path(__file__).parent))
from scripts.collectors.transfermarkt_collector import TransfermarktCollector
from loguru import logger

# Configure logger
logger.remove()
logger.add(sys.stdout, level="INFO")

def debug_urls():
    """Debug what URLs are being fetched and what data they return"""
    collector = TransfermarktCollector()

    team_name = "1. FC Kaiserslautern"
    team_data = collector.url_mappings.get(team_name)

    if not team_data:
        logger.error(f"No team data found for {team_name}")
        return

    logger.info(f"Team: {team_name}")
    logger.info(f"Team ID: {team_data['team_id']}")
    logger.info(f"URL Slug: {team_data['url_slug']}")
    logger.info("="*80)

    # Test historical season
    historical_season = "2019-2020"
    hist_year = historical_season.split('-')[1]
    hist_date = f"{hist_year}-01-01"

    hist_url = f"{collector.BASE_URL}/{team_data['url_slug']}/marktwertanalyse/verein/{team_data['team_id']}/stichtag/{hist_date}"

    logger.info(f"\nHistorical ({historical_season}):")
    logger.info(f"Reference Date: {hist_date}")
    logger.info(f"URL: {hist_url}")

    # Fetch and parse
    hist_soup = collector._make_request(hist_url)
    if hist_soup:
        # Look for the market value
        mv_box = hist_soup.find('div', class_='data-header__market-value-wrapper')
        if not mv_box:
            mv_box = hist_soup.find('a', class_='data-header__market-value-wrapper')

        if mv_box:
            value_text = mv_box.text.strip()
            logger.info(f"Market Value Found: {value_text}")
            parsed = collector.parse_market_value(value_text)
            logger.info(f"Parsed Value: €{parsed:,}" if parsed else "Could not parse")
        else:
            logger.warning("Market value box not found in HTML")

    # Test current season
    current_season = "2024-2025"
    curr_year = current_season.split('-')[1]
    curr_date = f"{curr_year}-01-01"

    curr_url = f"{collector.BASE_URL}/{team_data['url_slug']}/marktwertanalyse/verein/{team_data['team_id']}/stichtag/{curr_date}"

    logger.info(f"\nCurrent ({current_season}):")
    logger.info(f"Reference Date: {curr_date}")
    logger.info(f"URL: {curr_url}")

    # Fetch and parse
    curr_soup = collector._make_request(curr_url)
    if curr_soup:
        # Look for the market value
        mv_box = curr_soup.find('div', class_='data-header__market-value-wrapper')
        if not mv_box:
            mv_box = curr_soup.find('a', class_='data-header__market-value-wrapper')

        if mv_box:
            value_text = mv_box.text.strip()
            logger.info(f"Market Value Found: {value_text}")
            parsed = collector.parse_market_value(value_text)
            logger.info(f"Parsed Value: €{parsed:,}" if parsed else "Could not parse")
        else:
            logger.warning("Market value box not found in HTML")

    logger.info("\n" + "="*80)
    logger.info("ANALYSIS:")
    logger.info("If both values are the same, Transfermarkt's archive may not have")
    logger.info("historical data for this team/date, or the URL format is incorrect.")
    logger.info("="*80)

if __name__ == "__main__":
    debug_urls()
