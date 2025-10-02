"""
Data Preprocessing and Normalization
Converts geospatial data into text documents suitable for embedding
"""

import pandas as pd
import geopandas as gpd
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GeoDataPreprocessor:
    """Preprocess and convert geospatial data into text for embeddings"""

    def __init__(self):
        pass

    def gdf_to_documents(self, gdf: gpd.GeoDataFrame, data_type: str) -> List[Dict[str, Any]]:
        """
        Convert GeoDataFrame to structured documents for RAG

        Args:
            gdf: GeoDataFrame with geospatial data
            data_type: Type of data (earthquake, traffic, demographics, infrastructure)

        Returns:
            List of document dictionaries with text and metadata
        """
        if gdf.empty:
            logger.warning(f"Empty GeoDataFrame for {data_type}")
            return []

        documents = []

        # Route to appropriate processor based on data type
        if data_type == 'earthquake' or 'magnitude' in gdf.columns:
            documents = self._process_earthquakes(gdf)
        elif data_type == 'traffic_accident' or 'crash_date' in gdf.columns:
            documents = self._process_accidents(gdf)
        elif data_type == 'demographics' or 'population' in gdf.columns:
            documents = self._process_demographics(gdf)
        elif data_type == 'infrastructure':
            if 'station_name' in gdf.columns:
                documents = self._process_ev_stations(gdf)
            elif 'school_name' in gdf.columns:
                documents = self._process_schools(gdf)

        logger.info(f"Processed {len(documents)} documents for {data_type}")
        return documents

    def _process_earthquakes(self, gdf: gpd.GeoDataFrame) -> List[Dict[str, Any]]:
        """Process earthquake data into documents"""
        documents = []

        for idx, row in gdf.iterrows():
            # Extract coordinates
            lon, lat = row.geometry.x, row.geometry.y

            # Format date
            time_str = row['time'].strftime('%Y-%m-%d %H:%M') if pd.notna(row.get('time')) else 'Unknown'

            # Build descriptive text
            text = f"""Earthquake event at {row.get('place', 'Unknown location')}.
Magnitude: {row.get('magnitude', 'N/A')} on the Richter scale.
Location: Latitude {lat:.4f}, Longitude {lon:.4f}.
Depth: {row.get('depth_km', 'N/A')} km below surface.
Time: {time_str}.
This seismic event occurred in a region that may have vulnerable populations and infrastructure nearby.
Data source: {row.get('data_source', 'USGS')}."""

            # Metadata for filtering
            metadata = {
                'data_type': 'earthquake',
                'latitude': lat,
                'longitude': lon,
                'magnitude': float(row.get('magnitude', 0)),
                'place': str(row.get('place', '')),
                'time': time_str,
                'depth_km': float(row.get('depth_km', 0)),
                'url': str(row.get('url', '')),
                'data_source': str(row.get('data_source', 'USGS'))
            }

            documents.append({
                'text': text,
                'metadata': metadata,
                'id': f"earthquake_{idx}"
            })

        return documents

    def _process_accidents(self, gdf: gpd.GeoDataFrame) -> List[Dict[str, Any]]:
        """Process traffic accident data into documents"""
        documents = []

        for idx, row in gdf.iterrows():
            lon, lat = row.geometry.x, row.geometry.y

            crash_date = row.get('crash_date')
            date_str = crash_date.strftime('%Y-%m-%d') if pd.notna(crash_date) else 'Unknown'

            text = f"""Traffic accident in {row.get('borough', 'Unknown')} borough, New York City.
Date: {date_str}.
Location: Latitude {lat:.4f}, Longitude {lon:.4f}.
Injuries: {row.get('injuries', 0)} persons injured.
Fatalities: {row.get('fatalities', 0)} persons killed.
Contributing factor: {row.get('contributing_factor', 'Unknown')}.
This accident location may indicate a high-risk area for pedestrians and vehicles.
Data source: {row.get('data_source', 'NYC Open Data')}."""

            metadata = {
                'data_type': 'traffic_accident',
                'latitude': lat,
                'longitude': lon,
                'borough': str(row.get('borough', 'Unknown')),
                'crash_date': date_str,
                'injuries': int(row.get('injuries', 0)),
                'fatalities': int(row.get('fatalities', 0)),
                'contributing_factor': str(row.get('contributing_factor', '')),
                'data_source': str(row.get('data_source', 'NYC Open Data'))
            }

            documents.append({
                'text': text,
                'metadata': metadata,
                'id': f"accident_{idx}"
            })

        return documents

    def _process_demographics(self, gdf: gpd.GeoDataFrame) -> List[Dict[str, Any]]:
        """Process demographic/census data into documents"""
        documents = []

        for idx, row in gdf.iterrows():
            # Use centroid for geometry
            centroid = row.geometry.centroid
            lon, lat = centroid.x, centroid.y

            text = f"""Demographic information for {row.get('county_name', 'Unknown County')}, {row.get('state', '')}.
Population: {row.get('population', 0):,} residents.
Population density: {row.get('population_density', 0)} persons per square mile.
Elderly population (65+): {row.get('elderly_percent', 0)}% of total population.
Median household income: ${row.get('median_income', 0):,}.
Location centroid: Latitude {lat:.4f}, Longitude {lon:.4f}.
This region's demographics indicate vulnerability factors for disaster planning and resource allocation.
Data source: {row.get('data_source', 'US Census')}."""

            metadata = {
                'data_type': 'demographics',
                'latitude': lat,
                'longitude': lon,
                'county_name': str(row.get('county_name', '')),
                'state': str(row.get('state', '')),
                'population': int(row.get('population', 0)),
                'population_density': int(row.get('population_density', 0)),
                'elderly_percent': float(row.get('elderly_percent', 0)),
                'median_income': int(row.get('median_income', 0)),
                'data_source': str(row.get('data_source', 'US Census'))
            }

            documents.append({
                'text': text,
                'metadata': metadata,
                'id': f"demographics_{idx}"
            })

        return documents

    def _process_ev_stations(self, gdf: gpd.GeoDataFrame) -> List[Dict[str, Any]]:
        """Process EV charging station data into documents"""
        documents = []

        for idx, row in gdf.iterrows():
            lon, lat = row.geometry.x, row.geometry.y

            text = f"""Electric vehicle charging station: {row.get('station_name', 'Unknown Station')}.
Address: {row.get('address', '')}, {row.get('city', '')}, {row.get('state', '')}.
Network: {row.get('ev_network', 'Unknown')}.
Number of chargers: {row.get('num_chargers', 0)}.
Access: {row.get('access_type', 'Unknown')} access.
Location: Latitude {lat:.4f}, Longitude {lon:.4f}.
This infrastructure supports electric vehicle adoption and may indicate areas needing additional charging capacity.
Data source: {row.get('data_source', 'NREL')}."""

            metadata = {
                'data_type': 'infrastructure',
                'subtype': 'ev_charging',
                'latitude': lat,
                'longitude': lon,
                'station_name': str(row.get('station_name', '')),
                'city': str(row.get('city', '')),
                'state': str(row.get('state', '')),
                'ev_network': str(row.get('ev_network', '')),
                'num_chargers': int(row.get('num_chargers', 0)),
                'access_type': str(row.get('access_type', '')),
                'data_source': str(row.get('data_source', 'NREL'))
            }

            documents.append({
                'text': text,
                'metadata': metadata,
                'id': f"ev_station_{idx}"
            })

        return documents

    def _process_schools(self, gdf: gpd.GeoDataFrame) -> List[Dict[str, Any]]:
        """Process school location data into documents"""
        documents = []

        for idx, row in gdf.iterrows():
            lon, lat = row.geometry.x, row.geometry.y

            text = f"""School: {row.get('school_name', 'Unknown School')}.
Type: {row.get('type', 'Unknown')} school.
Enrollment: Approximately {row.get('students', 0)} students.
Location: Latitude {lat:.4f}, Longitude {lon:.4f}.
This educational facility represents a concentration of vulnerable populations (children) and requires consideration in safety and transportation planning.
Data source: {row.get('data_source', 'Education Department')}."""

            metadata = {
                'data_type': 'infrastructure',
                'subtype': 'school',
                'latitude': lat,
                'longitude': lon,
                'school_name': str(row.get('school_name', '')),
                'type': str(row.get('type', '')),
                'students': int(row.get('students', 0)),
                'data_source': str(row.get('data_source', 'Education Dept'))
            }

            documents.append({
                'text': text,
                'metadata': metadata,
                'id': f"school_{idx}"
            })

        return documents

    def batch_process_datasets(self, datasets: Dict[str, gpd.GeoDataFrame]) -> List[Dict[str, Any]]:
        """
        Process multiple datasets into a single document collection

        Args:
            datasets: Dictionary mapping dataset name to GeoDataFrame

        Returns:
            Combined list of all documents
        """
        all_documents = []

        for name, gdf in datasets.items():
            logger.info(f"Processing {name}...")

            # Infer data type from dataset name or columns
            if 'earthquake' in name.lower():
                data_type = 'earthquake'
            elif 'accident' in name.lower() or 'traffic' in name.lower():
                data_type = 'traffic_accident'
            elif 'demographic' in name.lower() or 'census' in name.lower():
                data_type = 'demographics'
            elif 'ev' in name.lower() or 'charging' in name.lower():
                data_type = 'infrastructure'
            elif 'school' in name.lower():
                data_type = 'infrastructure'
            else:
                data_type = 'unknown'

            docs = self.gdf_to_documents(gdf, data_type)
            all_documents.extend(docs)

        logger.info(f"Total documents processed: {len(all_documents)}")
        return all_documents

    def clean_text(self, text: str) -> str:
        """Clean and normalize text for embedding"""
        # Remove extra whitespace
        text = ' '.join(text.split())
        # Remove special characters that might interfere
        text = text.replace('\n', ' ').replace('\r', ' ')
        return text.strip()

    def add_spatial_context(self, doc: Dict[str, Any], nearby_features: List[str]) -> Dict[str, Any]:
        """
        Augment document with spatial context (e.g., nearby features)

        Args:
            doc: Document dictionary
            nearby_features: List of nearby feature descriptions

        Returns:
            Enhanced document
        """
        if nearby_features:
            context = " Nearby: " + ", ".join(nearby_features) + "."
            doc['text'] += context

        return doc


if __name__ == "__main__":
    # Test preprocessing
    from data_ingestion.esri_loader import load_all_datasets

    datasets = load_all_datasets()
    preprocessor = GeoDataPreprocessor()

    all_docs = preprocessor.batch_process_datasets(datasets)

    print(f"\nTotal documents: {len(all_docs)}")
    print("\nSample document:")
    print(json.dumps(all_docs[0], indent=2, default=str))