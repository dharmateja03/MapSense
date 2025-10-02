"""
Retriever - Orchestrates query processing, embedding, and document retrieval
Supports semantic search with optional geospatial and temporal filtering
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import re
from datetime import datetime, timedelta
import logging

from indexing.embedder import DocumentEmbedder
from indexing.vector_store import FAISSVectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GeoRAGRetriever:
    """
    Retriever for geospatial RAG system
    Combines semantic search with location and metadata filtering
    """

    def __init__(self, embedder: DocumentEmbedder, vector_store: FAISSVectorStore):
        """
        Initialize retriever

        Args:
            embedder: Document embedder for query encoding
            vector_store: Vector store for document retrieval
        """
        self.embedder = embedder
        self.vector_store = vector_store

        # Location patterns for query parsing
        self.location_patterns = {
            'california': {'lat': 36.7783, 'lon': -119.4179},
            'ca': {'lat': 36.7783, 'lon': -119.4179},
            'new york': {'lat': 40.7128, 'lon': -74.0060},
            'nyc': {'lat': 40.7128, 'lon': -74.0060},
            'boston': {'lat': 42.3601, 'lon': -71.0589},
            'florida': {'lat': 27.9944, 'lon': -81.7603},
            'san francisco': {'lat': 37.7749, 'lon': -122.4194},
            'los angeles': {'lat': 34.0522, 'lon': -118.2437},
        }

        logger.info("Initialized GeoRAGRetriever")

    def retrieve(self,
                query: str,
                top_k: int = 5,
                filters: Optional[Dict[str, Any]] = None,
                use_smart_filters: bool = True) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query

        Args:
            query: Natural language query
            top_k: Number of documents to retrieve
            filters: Optional explicit filters
            use_smart_filters: Whether to extract filters from query

        Returns:
            List of relevant documents with metadata
        """
        logger.info(f"Processing query: {query}")

        # Parse query for implicit filters
        if use_smart_filters:
            extracted_filters = self._extract_query_filters(query)

            # Merge with explicit filters
            if filters:
                extracted_filters.update(filters)

            filters = extracted_filters

        logger.info(f"Applied filters: {filters}")

        # Generate query embedding
        query_embedding = self.embedder.embed_query(query)

        # Check for location-based search
        location = self._extract_location(query)

        if location:
            logger.info(f"Location-based search: {location}")
            results = self.vector_store.search_with_location_filter(
                query_embedding=query_embedding,
                latitude=location['lat'],
                longitude=location['lon'],
                radius_km=location.get('radius_km', 100),
                top_k=top_k
            )
        else:
            # Standard semantic search
            results = self.vector_store.search(
                query_embedding=query_embedding,
                top_k=top_k,
                filters=filters
            )

        logger.info(f"Retrieved {len(results)} documents")
        return results

    def _extract_query_filters(self, query: str) -> Dict[str, Any]:
        """
        Extract filters from natural language query

        Args:
            query: User query

        Returns:
            Dictionary of filters
        """
        filters = {}
        query_lower = query.lower()

        # Data type detection
        if any(word in query_lower for word in ['earthquake', 'seismic', 'tremor']):
            filters['data_type'] = 'earthquake'

        elif any(word in query_lower for word in ['accident', 'crash', 'collision', 'traffic']):
            filters['data_type'] = 'traffic_accident'

        elif any(word in query_lower for word in ['population', 'demographic', 'census', 'elderly', 'resident']):
            filters['data_type'] = 'demographics'

        elif any(word in query_lower for word in ['charging', 'ev', 'station', 'infrastructure', 'school']):
            filters['data_type'] = 'infrastructure'

        # Magnitude filters for earthquakes
        magnitude_match = re.search(r'magnitude\s+(\d+(?:\.\d+)?)', query_lower)
        if magnitude_match:
            mag = float(magnitude_match.group(1))
            filters['magnitude'] = {'min': mag - 0.5, 'max': mag + 0.5}

        magnitude_above = re.search(r'(?:above|greater than|over)\s+(\d+(?:\.\d+)?)', query_lower)
        if magnitude_above:
            filters['magnitude'] = {'min': float(magnitude_above.group(1))}

        # Time filters
        if 'recent' in query_lower or 'latest' in query_lower:
            filters['time_recent'] = True

        # Severity filters
        if any(word in query_lower for word in ['severe', 'major', 'serious', 'fatal']):
            filters['severity'] = 'high'

        return filters

    def _extract_location(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Extract location from query

        Args:
            query: User query

        Returns:
            Location dict with lat/lon or None
        """
        query_lower = query.lower()

        # Check for known locations
        for location_name, coords in self.location_patterns.items():
            if location_name in query_lower:
                location = coords.copy()

                # Check for radius specification
                radius_match = re.search(r'within\s+(\d+)\s*(?:km|miles?)', query_lower)
                if radius_match:
                    radius = int(radius_match.group(1))
                    if 'mile' in query_lower:
                        radius = int(radius * 1.60934)  # Convert to km
                    location['radius_km'] = radius
                else:
                    location['radius_km'] = 100  # Default radius

                return location

        # Check for coordinate patterns
        coord_pattern = r'(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)'
        coord_match = re.search(coord_pattern, query)
        if coord_match:
            lat, lon = float(coord_match.group(1)), float(coord_match.group(2))
            return {'lat': lat, 'lon': lon, 'radius_km': 50}

        return None

    def retrieve_multi_domain(self,
                            query: str,
                            top_k_per_domain: int = 3) -> Dict[str, List[Dict[str, Any]]]:
        """
        Retrieve documents from multiple domains separately

        Args:
            query: User query
            top_k_per_domain: Number of results per domain

        Returns:
            Dictionary mapping domain to results
        """
        domains = ['earthquake', 'traffic_accident', 'demographics', 'infrastructure']
        results = {}

        query_embedding = self.embedder.embed_query(query)

        for domain in domains:
            domain_results = self.vector_store.search(
                query_embedding=query_embedding,
                top_k=top_k_per_domain,
                filters={'data_type': domain}
            )

            if domain_results:
                results[domain] = domain_results

        return results

    def retrieve_with_context(self,
                            query: str,
                            top_k: int = 5) -> Tuple[List[Dict[str, Any]], str]:
        """
        Retrieve documents and format as context for LLM

        Args:
            query: User query
            top_k: Number of documents

        Returns:
            Tuple of (documents, formatted_context_string)
        """
        documents = self.retrieve(query, top_k=top_k)

        # Format context
        context_parts = []
        for i, doc in enumerate(documents, 1):
            meta = doc.get('metadata', {})
            text = doc.get('text', '')
            score = doc.get('score', 0)

            context_parts.append(f"[Document {i}] (Relevance: {score:.2f})")
            context_parts.append(text)

            # Add source reference
            source = meta.get('data_source', 'Unknown')
            context_parts.append(f"Source: {source}")
            context_parts.append("")  # Blank line

        context_string = "\n".join(context_parts)

        return documents, context_string

    def hybrid_retrieve(self,
                       query: str,
                       location: Optional[Tuple[float, float]] = None,
                       time_filter: Optional[str] = None,
                       data_types: Optional[List[str]] = None,
                       top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Advanced retrieval combining multiple filter types

        Args:
            query: Search query
            location: (latitude, longitude) tuple
            time_filter: 'recent', 'last_week', 'last_month'
            data_types: List of data types to include
            top_k: Number of results

        Returns:
            Filtered and ranked results
        """
        # Start with semantic search
        query_embedding = self.embedder.embed_query(query)

        # Build filters
        filters = {}
        if data_types:
            filters['data_type'] = data_types

        # Retrieve candidates
        if location:
            candidates = self.vector_store.search_with_location_filter(
                query_embedding=query_embedding,
                latitude=location[0],
                longitude=location[1],
                radius_km=200,
                top_k=top_k * 2
            )
        else:
            candidates = self.vector_store.search(
                query_embedding=query_embedding,
                top_k=top_k * 2,
                filters=filters
            )

        # Apply time filter if specified
        if time_filter:
            candidates = self._filter_by_time(candidates, time_filter)

        # Re-rank and return top k
        return candidates[:top_k]

    def _filter_by_time(self, documents: List[Dict[str, Any]], time_filter: str) -> List[Dict[str, Any]]:
        """
        Filter documents by time

        Args:
            documents: List of documents
            time_filter: Time filter type

        Returns:
            Filtered documents
        """
        if time_filter == 'recent':
            days = 30
        elif time_filter == 'last_week':
            days = 7
        elif time_filter == 'last_month':
            days = 30
        else:
            return documents

        cutoff = datetime.now() - timedelta(days=days)

        filtered = []
        for doc in documents:
            meta = doc.get('metadata', {})

            # Check time field (different formats for different data types)
            time_field = meta.get('time') or meta.get('crash_date')

            if time_field:
                # Parse time string
                try:
                    if isinstance(time_field, str):
                        doc_time = datetime.strptime(time_field[:10], '%Y-%m-%d')
                    else:
                        doc_time = time_field

                    if doc_time >= cutoff:
                        filtered.append(doc)
                except:
                    # If parsing fails, include document
                    filtered.append(doc)
            else:
                # If no time field, include document
                filtered.append(doc)

        return filtered


class QueryExpander:
    """Expand queries with synonyms and domain knowledge"""

    def __init__(self):
        self.expansions = {
            'earthquake': ['seismic event', 'tremor', 'quake'],
            'accident': ['crash', 'collision', 'incident'],
            'population': ['residents', 'people', 'inhabitants'],
            'elderly': ['senior', 'aged', 'older adult'],
        }

    def expand_query(self, query: str) -> str:
        """
        Expand query with synonyms

        Args:
            query: Original query

        Returns:
            Expanded query
        """
        expanded_terms = [query]

        for term, synonyms in self.expansions.items():
            if term in query.lower():
                for syn in synonyms:
                    expanded_terms.append(query.replace(term, syn))

        return " OR ".join(expanded_terms[:3])  # Limit expansion


def create_retriever(embedder: DocumentEmbedder,
                    vector_store: FAISSVectorStore) -> GeoRAGRetriever:
    """
    Factory function to create retriever

    Args:
        embedder: Document embedder
        vector_store: Vector store

    Returns:
        GeoRAGRetriever instance
    """
    return GeoRAGRetriever(embedder, vector_store)


if __name__ == "__main__":
    # Test retriever
    from indexing import create_embedder, create_vector_store

    embedder = create_embedder('standard')
    vector_store = create_vector_store(embedding_dim=embedder.get_embedding_dimension())

    retriever = GeoRAGRetriever(embedder, vector_store)

    # Test query parsing
    test_queries = [
        "Show earthquakes in California above magnitude 5",
        "Traffic accidents in NYC with fatalities",
        "Population density in Florida counties"
    ]

    for query in test_queries:
        filters = retriever._extract_query_filters(query)
        location = retriever._extract_location(query)
        print(f"\nQuery: {query}")
        print(f"Filters: {filters}")
        print(f"Location: {location}")