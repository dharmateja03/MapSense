"""
RAG Pipeline - Orchestrates retrieval and LLM generation
Combines retrieved geospatial context with LLM to generate informed answers
Supports both OpenAI and Mistral AI models
"""

from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from typing import List, Dict, Any, Optional
import os
from tenacity import retry, stop_after_attempt, wait_exponential
import logging

from retrieval.retriever import GeoRAGRetriever
from indexing.embedder import DocumentEmbedder
from indexing.vector_store import FAISSVectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GeoRAGPipeline:
    """
    End-to-end RAG pipeline for geospatial queries
    Retrieves relevant documents and generates natural language answers
    """

    def __init__(self,
                 retriever: GeoRAGRetriever,
                 api_key: Optional[str] = None,
                 model: str = "mistral-small-latest",
                 temperature: float = 0.7,
                 provider: str = "mistral"):
        """
        Initialize RAG pipeline

        Args:
            retriever: Document retriever
            api_key: Mistral API key (or set MISTRAL_API_KEY env var)
            model: LLM model to use (mistral-small-latest, mistral-medium-latest, mistral-large-latest)
            temperature: Generation temperature
            provider: LLM provider ('mistral' or 'openai')
        """
        self.retriever = retriever
        self.model = model
        self.temperature = temperature
        self.provider = provider

        # Set up Mistral client (free API)
        api_key = api_key or os.getenv('MISTRAL_API_KEY')
        if api_key:
            self.client = MistralClient(api_key=api_key)
            self.use_llm = True
            logger.info(f"Initialized with Mistral AI model: {model}")
        else:
            self.use_llm = False
            self.client = None
            logger.warning("No Mistral API key found. Using retrieval-only mode.")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def generate_answer(self,
                       query: str,
                       top_k: int = 5,
                       include_sources: bool = True) -> Dict[str, Any]:
        """
        Generate answer to geospatial query using RAG

        Args:
            query: User's natural language query
            top_k: Number of documents to retrieve
            include_sources: Whether to include source documents in response

        Returns:
            Dictionary with answer, sources, and metadata
        """
        logger.info(f"Processing query: {query}")

        # Step 1: Retrieve relevant documents
        documents, context = self.retriever.retrieve_with_context(query, top_k=top_k)

        if not documents:
            return {
                'answer': "I couldn't find any relevant information for your query. Please try rephrasing or ask about earthquakes, traffic accidents, demographics, or infrastructure.",
                'sources': [],
                'confidence': 'low'
            }

        # Step 2: Generate answer using LLM
        if self.use_llm:
            answer = self._generate_llm_answer(query, context, documents)
        else:
            # Fallback: return top document summary
            answer = self._generate_fallback_answer(query, documents)

        # Step 3: Prepare response
        response = {
            'answer': answer,
            'num_sources': len(documents),
            'confidence': self._estimate_confidence(documents)
        }

        if include_sources:
            response['sources'] = self._format_sources(documents)
            response['map_data'] = self._extract_map_data(documents)

        return response

    def _generate_llm_answer(self,
                            query: str,
                            context: str,
                            documents: List[Dict[str, Any]]) -> str:
        """
        Generate answer using LLM with retrieved context

        Args:
            query: User query
            context: Retrieved document context
            documents: Source documents

        Returns:
            Generated answer
        """
        # Build prompt
        system_prompt = """You are a GeoAI assistant specializing in geospatial data analysis.
You help users understand patterns in earthquakes, traffic accidents, demographics, and infrastructure.

Your responses should:
1. Directly answer the user's question using the provided data
2. Highlight key insights and patterns
3. Include specific numbers, locations, and statistics when available
4. Mention data sources for credibility
5. Suggest actionable insights when appropriate
6. Be concise but informative (2-4 paragraphs)

If the data is insufficient, say so clearly and suggest what additional data would help."""

        user_prompt = f"""Based on the following geospatial data, please answer this question:

Question: {query}

Retrieved Data:
{context}

Please provide a clear, data-driven answer with specific details from the sources."""

        try:
            # Call Mistral API
            messages = [
                ChatMessage(role="system", content=system_prompt),
                ChatMessage(role="user", content=user_prompt)
            ]

            chat_response = self.client.chat(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=500
            )

            answer = chat_response.choices[0].message.content.strip()
            logger.info("Generated Mistral answer successfully")
            return answer

        except Exception as e:
            logger.error(f"Error generating Mistral answer: {e}")
            return self._generate_fallback_answer(query, documents)

    def _generate_fallback_answer(self, query: str, documents: List[Dict[str, Any]]) -> str:
        """
        Generate answer without LLM (template-based)

        Args:
            query: User query
            documents: Retrieved documents

        Returns:
            Template-based answer
        """
        if not documents:
            return "No relevant data found for your query."

        # Analyze document types
        doc_types = {}
        for doc in documents:
            dtype = doc.get('metadata', {}).get('data_type', 'unknown')
            doc_types[dtype] = doc_types.get(dtype, 0) + 1

        # Build answer based on document types
        answer_parts = [f"Found {len(documents)} relevant data points for your query:"]

        # Earthquake summary
        if 'earthquake' in doc_types:
            eq_docs = [d for d in documents if d.get('metadata', {}).get('data_type') == 'earthquake']
            magnitudes = [d.get('metadata', {}).get('magnitude', 0) for d in eq_docs]
            if magnitudes:
                max_mag = max(magnitudes)
                answer_parts.append(f"Earthquake data: {len(eq_docs)} events, largest magnitude {max_mag:.1f}.")

        # Traffic accident summary
        if 'traffic_accident' in doc_types:
            acc_docs = [d for d in documents if d.get('metadata', {}).get('data_type') == 'traffic_accident']
            total_injuries = sum(d.get('metadata', {}).get('injuries', 0) for d in acc_docs)
            answer_parts.append(f"Traffic accidents: {len(acc_docs)} incidents with {total_injuries} total injuries.")

        # Demographics summary
        if 'demographics' in doc_types:
            demo_docs = [d for d in documents if d.get('metadata', {}).get('data_type') == 'demographics']
            total_pop = sum(d.get('metadata', {}).get('population', 0) for d in demo_docs)
            answer_parts.append(f"Demographics: {len(demo_docs)} regions with {total_pop:,} total population.")

        # Infrastructure summary
        if 'infrastructure' in doc_types:
            infra_docs = [d for d in documents if d.get('metadata', {}).get('data_type') == 'infrastructure']
            answer_parts.append(f"Infrastructure: {len(infra_docs)} facilities or stations identified.")

        # Add data sources
        sources = set(d.get('metadata', {}).get('data_source', 'Unknown') for d in documents)
        answer_parts.append(f"Data sources: {', '.join(sources)}.")

        return " ".join(answer_parts)

    def _format_sources(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Format source documents for display

        Args:
            documents: Retrieved documents

        Returns:
            Formatted sources
        """
        sources = []

        for i, doc in enumerate(documents):
            meta = doc.get('metadata', {})

            source = {
                'id': i + 1,
                'data_type': meta.get('data_type', 'unknown'),
                'location': {
                    'latitude': meta.get('latitude'),
                    'longitude': meta.get('longitude')
                },
                'data_source': meta.get('data_source', 'Unknown'),
                'score': doc.get('score', 0),
                'summary': doc.get('text', '')[:200] + '...'
            }

            # Add type-specific fields
            if meta.get('data_type') == 'earthquake':
                source['magnitude'] = meta.get('magnitude')
                source['time'] = meta.get('time')

            elif meta.get('data_type') == 'traffic_accident':
                source['injuries'] = meta.get('injuries')
                source['fatalities'] = meta.get('fatalities')
                source['date'] = meta.get('crash_date')

            elif meta.get('data_type') == 'demographics':
                source['population'] = meta.get('population')
                source['county'] = meta.get('county_name')

            elif meta.get('data_type') == 'infrastructure':
                source['name'] = meta.get('station_name') or meta.get('school_name')
                source['subtype'] = meta.get('subtype')

            sources.append(source)

        return sources

    def _extract_map_data(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract data for map visualization

        Args:
            documents: Retrieved documents

        Returns:
            List of map features
        """
        map_features = []

        for doc in documents:
            meta = doc.get('metadata', {})

            if not meta.get('latitude') or not meta.get('longitude'):
                continue

            feature = {
                'lat': meta['latitude'],
                'lon': meta['longitude'],
                'type': meta.get('data_type', 'unknown'),
                'popup_text': self._create_popup_text(meta)
            }

            # Add type-specific styling hints
            if meta.get('data_type') == 'earthquake':
                feature['color'] = 'red'
                feature['size'] = min(meta.get('magnitude', 3) * 2, 15)
                feature['icon'] = 'warning'

            elif meta.get('data_type') == 'traffic_accident':
                feature['color'] = 'orange'
                feature['size'] = 8
                feature['icon'] = 'exclamation-triangle'

            elif meta.get('data_type') == 'demographics':
                feature['color'] = 'blue'
                feature['size'] = 10
                feature['icon'] = 'users'

            elif meta.get('data_type') == 'infrastructure':
                feature['color'] = 'green'
                feature['size'] = 7
                feature['icon'] = 'building'

            map_features.append(feature)

        return map_features

    def _create_popup_text(self, metadata: Dict[str, Any]) -> str:
        """Create popup text for map markers"""

        dtype = metadata.get('data_type', 'unknown')

        if dtype == 'earthquake':
            return f"Magnitude {metadata.get('magnitude', 'N/A')} earthquake\n{metadata.get('place', 'Unknown')}"

        elif dtype == 'traffic_accident':
            return f"Traffic accident in {metadata.get('borough', 'Unknown')}\nInjuries: {metadata.get('injuries', 0)}"

        elif dtype == 'demographics':
            return f"{metadata.get('county_name', 'Unknown')}\nPopulation: {metadata.get('population', 0):,}"

        elif dtype == 'infrastructure':
            return metadata.get('station_name') or metadata.get('school_name', 'Unknown facility')

        return "Unknown location"

    def _estimate_confidence(self, documents: List[Dict[str, Any]]) -> str:
        """
        Estimate confidence in answer based on retrieval quality

        Args:
            documents: Retrieved documents

        Returns:
            Confidence level: 'high', 'medium', 'low'
        """
        if not documents:
            return 'low'

        avg_score = sum(d.get('score', 0) for d in documents) / len(documents)

        if avg_score > 0.7 and len(documents) >= 3:
            return 'high'
        elif avg_score > 0.5:
            return 'medium'
        else:
            return 'low'

    def batch_query(self, queries: List[str], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Process multiple queries in batch

        Args:
            queries: List of queries
            top_k: Documents per query

        Returns:
            List of responses
        """
        responses = []

        for query in queries:
            logger.info(f"Processing batch query: {query}")
            response = self.generate_answer(query, top_k=top_k)
            responses.append(response)

        return responses


def create_rag_pipeline(embedder: DocumentEmbedder,
                       vector_store: FAISSVectorStore,
                       api_key: Optional[str] = None) -> GeoRAGPipeline:
    """
    Factory function to create RAG pipeline

    Args:
        embedder: Document embedder
        vector_store: Vector store
        api_key: OpenAI API key

    Returns:
        GeoRAGPipeline instance
    """
    from retrieval import create_retriever

    retriever = create_retriever(embedder, vector_store)
    return GeoRAGPipeline(retriever, api_key=api_key)


if __name__ == "__main__":
    # Test RAG pipeline (without actual data)
    from indexing import create_embedder, create_vector_store

    embedder = create_embedder('standard')
    vector_store = create_vector_store(embedding_dim=embedder.get_embedding_dimension())

    pipeline = create_rag_pipeline(embedder, vector_store)

    # Test query
    test_query = "What are recent earthquakes in California?"
    print(f"Query: {test_query}")

    # This would fail without data, but shows the structure
    # response = pipeline.generate_answer(test_query)
    # print(f"Response: {response}")