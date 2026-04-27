# src/data_loader.py

import requests
import re
import os
from bs4 import BeautifulSoup
from datetime import datetime
from dateutil.relativedelta import relativedelta


def get_insideairbnb_url(city_path, rank=0):
    """
    Get InsideAirbnb download URLs for a given city.
    
    Parameters:
        city_path: str — e.g. 'united-states/ca/los-angeles'
        rank: int — 0 = newest, 1 = second newest, etc.
    
    Returns:
        dict with keys: listings, calendar, neighbourhoods, date
    """
    page = requests.get('https://insideairbnb.com/get-the-data/')
    soup = BeautifulSoup(page.content, 'html.parser')

    pattern = re.compile(
        rf"https://data\.insideairbnb\.com/{re.escape(city_path)}/([\d-]+)/data/listings\.csv\.gz"
    )

    dates = sorted(list(set(
        match.group(1) for link in soup.find_all('a', href=True)
        if (match := pattern.search(link['href']))
    )), reverse=True)

    newest = datetime.strptime(dates[0], '%Y-%m-%d')

    if rank == 0:
        chosen_str = dates[0]
    else:
        estimated = newest - relativedelta(months=3 * rank)
        found = None
        for delta in range(-15, 16):
            candidate = estimated + relativedelta(days=delta)
            candidate_str = candidate.strftime('%Y-%m-%d')
            base = f"https://data.insideairbnb.com/{city_path}/{candidate_str}"
            test = requests.head(f"{base}/data/listings.csv.gz")
            if test.status_code == 200:
                found = candidate_str
                break
        if not found:
            raise ValueError(f"Could not find data around {estimated.strftime('%Y-%m-%d')}")
        chosen_str = found

    print(f"Using date: {chosen_str}")
    base = f"https://data.insideairbnb.com/{city_path}/{chosen_str}"
    return {
        'listings': f"{base}/data/listings.csv.gz",
        'calendar': f"{base}/data/calendar.csv.gz",
        'neighbourhoods': f"{base}/visualisations/neighbourhoods.geojson",
        'date': chosen_str
    }


def download_city_data(city, city_path, rank=1, base_dir='../data/raw'):
    """
    Download all InsideAirbnb files for a city.
    
    Parameters:
        city: str — short name e.g. 'los-angeles'
        city_path: str — InsideAirbnb path e.g. 'united-states/ca/los-angeles'
        rank: int — which date to use (default 1 = second newest)
        base_dir: str — where to save data
    """
    os.makedirs(f'{base_dir}/{city}', exist_ok=True)
    urls = get_insideairbnb_url(city_path, rank=rank)

    for name, url in urls.items():
        if name == 'date':
            continue
        ext = '.geojson' if 'geojson' in url else '.csv.gz'
        filepath = f'{base_dir}/{city}/{name}{ext}'
        print(f'Downloading {city}/{name}...')
        response = requests.get(url)
        with open(filepath, 'wb') as f:
            f.write(response.content)
        print(f'  Saved → {filepath}')

    return urls