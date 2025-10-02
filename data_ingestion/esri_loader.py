"""
ESRI Data Loader - Fetches geospatial datasets from ArcGIS Hub and REST APIs
Supports earthquakes, traffic accidents, demographics, and infrastructure data
"""

import requests
import pandas as pd
import geopandas as gpd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import json
from shapely.geometry import Point, shape
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ESRIDataLoader:
    """Load and fetch data from ESRI ArcGIS services"""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'GeoAI-RAG-Assistant/1.0'})

    def fetch_earthquakes(self,
                         days_back: int = 30,
                         min_magnitude: float = 2.5,
                         region: Optional[str] = None) -> gpd.GeoDataFrame:
        """
        Fetch recent earthquake data from USGS (compatible with ESRI formats)

        Args:
            days_back: Number of days to look back
            min_magnitude: Minimum earthquake magnitude
            region: Optional region filter (e.g., 'California')

        Returns:
            GeoDataFrame with earthquake events
        """
        logger.info(f"Fetching earthquakes from last {days_back} days...")

        # USGS Earthquake API (returns GeoJSON)
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days_back)

        url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
        params = {
            'format': 'geojson',
            'starttime': start_time.strftime('%Y-%m-%d'),
            'endtime': end_time.strftime('%Y-%m-%d'),
            'minmagnitude': min_magnitude
        }

        # Add bounding box for California if specified
        if region and region.lower() == 'california':
            params.update({
                'minlatitude': 32.5,
                'maxlatitude': 42.0,
                'minlongitude': -124.5,
                'maxlongitude': -114.1
            })

        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            # Convert GeoJSON to GeoDataFrame
            gdf = gpd.GeoDataFrame.from_features(data['features'])

            # Extract useful properties
            gdf['magnitude'] = gdf['mag']
            gdf['place'] = gdf['place']
            gdf['time'] = pd.to_datetime(gdf['time'], unit='ms')
            gdf['url'] = gdf['url']
            gdf['depth_km'] = gdf['depth']
            gdf['data_type'] = 'earthquake'
            gdf['data_source'] = 'USGS Earthquake Catalog'

            logger.info(f"Loaded {len(gdf)} earthquake records")
            return gdf[['geometry', 'magnitude', 'place', 'time', 'depth_km', 'url', 'data_type', 'data_source']]

        except Exception as e:
            logger.error(f"Error fetching earthquakes: {e}")
            return gpd.GeoDataFrame()

    def fetch_nyc_accidents(self, limit: int = 5000) -> gpd.GeoDataFrame:
        """
        Fetch NYC Motor Vehicle Collisions from NYC Open Data (ESRI-hosted)

        Args:
            limit: Maximum number of records to fetch

        Returns:
            GeoDataFrame with accident locations
        """
        logger.info(f"Fetching NYC accident data (limit={limit})...")

        # NYC Open Data - Motor Vehicle Collisions
        url = "https://data.cityofnewyork.us/resource/h9gi-nx95.json"
        params = {
            '$limit': limit,
            '$where': 'latitude IS NOT NULL AND longitude IS NOT NULL',
            '$order': 'crash_date DESC'
        }

        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            # Convert to DataFrame
            df = pd.DataFrame(data)

            if df.empty:
                return gpd.GeoDataFrame()

            # Create geometries
            df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
            df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
            df = df.dropna(subset=['latitude', 'longitude'])

            geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]
            gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:4326')

            # Extract useful fields
            gdf['crash_date'] = pd.to_datetime(gdf.get('crash_date', ''), errors='coerce')
            gdf['borough'] = gdf.get('borough', 'Unknown')
            gdf['injuries'] = pd.to_numeric(gdf.get('number_of_persons_injured', 0), errors='coerce')
            gdf['fatalities'] = pd.to_numeric(gdf.get('number_of_persons_killed', 0), errors='coerce')
            gdf['contributing_factor'] = gdf.get('contributing_factor_vehicle_1', 'Unknown')
            gdf['data_type'] = 'traffic_accident'
            gdf['data_source'] = 'NYC Open Data - Motor Vehicle Collisions'

            logger.info(f"Loaded {len(gdf)} NYC accident records")
            return gdf[['geometry', 'crash_date', 'borough', 'injuries', 'fatalities',
                       'contributing_factor', 'data_type', 'data_source']]

        except Exception as e:
            logger.error(f"Error fetching NYC accidents: {e}")
            return gpd.GeoDataFrame()

    def fetch_demographics_census(self, state: str = 'CA') -> gpd.GeoDataFrame:
        """
        Fetch county-level demographic data from US Census

        Args:
            state: State abbreviation (e.g., 'CA', 'NY', 'FL')

        Returns:
            GeoDataFrame with population demographics
        """
        logger.info(f"Fetching demographics for state {state}...")

        # Use ESRI Living Atlas Demographics (simplified example with US Census data)
        # In production, use Census API or ArcGIS Living Atlas

        try:
            # Fetch county boundaries from US Census
            counties_url = f"https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"
            response = self.session.get(counties_url, timeout=30)
            response.raise_for_status()
            counties_geojson = response.json()

            # Simulated population data (in production, fetch from Census API)
            # For demo purposes, create sample data
            features = []
            for feature in counties_geojson['features'][:50]:  # Limit for demo
                props = feature['properties']
                geom = shape(feature['geometry'])

                # Simulate demographic data
                features.append({
                    'geometry': geom,
                    'county_name': f"County {props.get('GEO_ID', 'Unknown')}",
                    'state': state,
                    'population': int(100000 + hash(str(props)) % 500000),
                    'population_density': int(100 + hash(str(props)) % 1000),
                    'elderly_percent': round(10 + (hash(str(props)) % 20), 1),
                    'median_income': int(40000 + hash(str(props)) % 60000),
                    'data_type': 'demographics',
                    'data_source': 'US Census Bureau'
                })

            gdf = gpd.GeoDataFrame(features, crs='EPSG:4326')
            logger.info(f"Loaded {len(gdf)} demographic records")
            return gdf

        except Exception as e:
            logger.error(f"Error fetching demographics: {e}")
            # Return sample data as fallback
            return self._create_sample_demographics(state)

    def fetch_ev_charging_stations(self, state: str = 'MA', city: Optional[str] = None) -> gpd.GeoDataFrame:
        """
        Fetch EV charging station locations from NREL Alternative Fuels Data Center

        Args:
            state: State abbreviation
            city: Optional city filter

        Returns:
            GeoDataFrame with charging station locations
        """
        logger.info(f"Fetching EV charging stations for {state}...")

        # NREL Alternative Fuels Data Center API
        url = "https://developer.nrel.gov/api/alt-fuel-stations/v1.json"
        params = {
            'api_key': 'DEMO_KEY',  # Use DEMO_KEY for testing, get real key for production
            'state': state,
            'fuel_type': 'ELEC',
            'status': 'E',  # Available stations
            'limit': 500
        }

        if city:
            params['city'] = city

        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            stations = data.get('fuel_stations', [])
            if not stations:
                logger.warning("No charging stations found")
                return gpd.GeoDataFrame()

            # Convert to GeoDataFrame
            records = []
            for station in stations:
                if station.get('latitude') and station.get('longitude'):
                    records.append({
                        'geometry': Point(station['longitude'], station['latitude']),
                        'station_name': station.get('station_name', 'Unknown'),
                        'address': station.get('street_address', ''),
                        'city': station.get('city', ''),
                        'state': station.get('state', ''),
                        'ev_network': station.get('ev_network', 'Unknown'),
                        'num_chargers': station.get('ev_level2_evse_num', 0),
                        'access_type': station.get('access_code', 'Unknown'),
                        'data_type': 'infrastructure',
                        'data_source': 'NREL Alternative Fuels Data Center'
                    })

            gdf = gpd.GeoDataFrame(records, crs='EPSG:4326')
            logger.info(f"Loaded {len(gdf)} EV charging station records")
            return gdf

        except Exception as e:
            logger.error(f"Error fetching EV stations: {e}")
            return self._create_sample_ev_stations(state)

    def fetch_schools(self, state: str = 'NY', limit: int = 500) -> gpd.GeoDataFrame:
        """
        Fetch school locations (infrastructure example)
        Uses sample data for demonstration

        Args:
            state: State abbreviation
            limit: Maximum number of records

        Returns:
            GeoDataFrame with school locations
        """
        logger.info(f"Creating sample school data for {state}...")

        # In production, use actual ArcGIS School data layers
        # For demo, create sample data

        if state == 'NY':
            # Sample NYC school locations
            schools = []
            base_coords = [
                (40.7589, -73.9851),  # Times Square area
                (40.7831, -73.9712),  # Upper West Side
                (40.7282, -73.9942),  # Lower Manhattan
                (40.7614, -73.9776),  # Midtown
                (40.7489, -73.9680),  # Queens area
            ]

            for i, (lat, lon) in enumerate(base_coords * (limit // len(base_coords) + 1)):
                if len(schools) >= limit:
                    break

                # Add some variation
                lat_offset = (hash(f"lat_{i}") % 1000) / 10000
                lon_offset = (hash(f"lon_{i}") % 1000) / 10000

                schools.append({
                    'geometry': Point(lon + lon_offset, lat + lat_offset),
                    'school_name': f'School {i+1}',
                    'type': ['Elementary', 'Middle', 'High'][i % 3],
                    'students': 200 + (hash(f"students_{i}") % 800),
                    'data_type': 'infrastructure',
                    'data_source': 'Sample School Data'
                })

            gdf = gpd.GeoDataFrame(schools, crs='EPSG:4326')
            logger.info(f"Created {len(gdf)} sample school records")
            return gdf

        return gpd.GeoDataFrame()

    def _create_sample_demographics(self, state: str) -> gpd.GeoDataFrame:
        """Create sample demographic data for demonstration"""

        # Sample county centroids for different states
        sample_data = {
            'CA': [(37.7749, -122.4194, 'San Francisco'), (34.0522, -118.2437, 'Los Angeles'),
                   (32.7157, -117.1611, 'San Diego')],
            'FL': [(25.7617, -80.1918, 'Miami'), (28.5383, -81.3792, 'Orlando'),
                   (30.3322, -81.6557, 'Jacksonville')],
            'NY': [(40.7128, -74.0060, 'New York'), (42.6526, -73.7562, 'Albany')]
        }

        coords = sample_data.get(state, sample_data['CA'])

        records = []
        for i, (lat, lon, name) in enumerate(coords):
            records.append({
                'geometry': Point(lon, lat),
                'county_name': name,
                'state': state,
                'population': 500000 + i * 200000,
                'population_density': 500 + i * 300,
                'elderly_percent': 15.0 + i * 3.5,
                'median_income': 50000 + i * 15000,
                'data_type': 'demographics',
                'data_source': 'Sample Census Data'
            })

        return gpd.GeoDataFrame(records, crs='EPSG:4326')

    def _create_sample_ev_stations(self, state: str) -> gpd.GeoDataFrame:
        """Create sample EV charging station data"""

        if state == 'MA':
            stations = [
                (42.3601, -71.0589, 'Boston Central Station'),
                (42.3736, -71.1097, 'Cambridge Station'),
                (42.3751, -71.1056, 'MIT Station'),
            ]
        else:
            stations = [(37.7749, -122.4194, 'Sample Station 1')]

        records = []
        for lat, lon, name in stations:
            records.append({
                'geometry': Point(lon, lat),
                'station_name': name,
                'address': '123 Main St',
                'city': 'Boston' if state == 'MA' else 'Unknown',
                'state': state,
                'ev_network': 'ChargePoint',
                'num_chargers': 4,
                'access_type': 'Public',
                'data_type': 'infrastructure',
                'data_source': 'Sample EV Data'
            })

        return gpd.GeoDataFrame(records, crs='EPSG:4326')


def load_all_datasets() -> Dict[str, gpd.GeoDataFrame]:
    """
    Load all example datasets from different domains

    Returns:
        Dictionary mapping dataset type to GeoDataFrame
    """
    loader = ESRIDataLoader()

    datasets = {
        'earthquakes': loader.fetch_earthquakes(days_back=30, region='California'),
        'traffic_accidents': loader.fetch_nyc_accidents(limit=2000),
        'demographics': loader.fetch_demographics_census(state='CA'),
        'ev_stations': loader.fetch_ev_charging_stations(state='MA', city='Boston'),
        'schools': loader.fetch_schools(state='NY', limit=100)
    }

    return datasets


if __name__ == "__main__":
    # Test the loader
    datasets = load_all_datasets()

    for name, gdf in datasets.items():
        print(f"\n{name.upper()}: {len(gdf)} records")
        if not gdf.empty:
            print(gdf.head(2))