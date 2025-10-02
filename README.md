# MapSense -RAG

A production-ready **Retrieval-Augmented Generation (RAG)** system for answering natural-language geospatial queries using multiple ESRI datasets. The system combines semantic search, vector embeddings, and LLM-powered generation with interactive map visualizations.

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/streamlit-1.31-red)
![License](https://img.shields.io/badge/license-MIT-green)

##  Features

- **Multi-Domain Geospatial Data**: Earthquakes, traffic accidents, demographics, and infrastructure
- **Semantic Search**: Powered by sentence-transformers and FAISS vector store
- **Natural Language Queries**: Ask questions in plain English
- **Interactive Maps**: Folium-based visualizations with multiple layers
- **LLM Integration**: Mistral AI-powered context-aware answers (FREE API!)
- **Real-Time Data**: Fetches from USGS, NYC Open Data, NREL, and Census APIs
- **Streamlit UI**: Beautiful, interactive web interface



### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) Mistral AI API key for LLM-powered answers - **Get it FREE at [console.mistral.ai](https://console.mistral.ai/)**


## ⚡ Quick Start

### Run the Streamlit App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

##  Screenshots

### Chat Interface
![Chat Interface](screenshots/chat.png)
*Natural language queries with Mistral AI-powered answers*

### Interactive Map
![Interactive Map](screenshots/map.png)
*Real-time earthquake and accident data visualization*

### Statistics Dashboard
![Statistics](screenshots/stats.png)
*Data analytics with charts and metrics*

### Test Individual Components

#### Load Data
```python
from data_ingestion import load_all_datasets

datasets = load_all_datasets()
print(f"Loaded {len(datasets)} datasets")
```

#### Generate Embeddings
```python
from data_ingestion import GeoDataPreprocessor
from indexing import create_embedder

preprocessor = GeoDataPreprocessor()
documents = preprocessor.batch_process_datasets(datasets)

embedder = create_embedder('standard')
documents = embedder.embed_documents(documents)
```

#### Query the System
```python
from rag_pipeline import create_rag_pipeline
from indexing import create_vector_store

vector_store = create_vector_store(embedding_dim=384)
vector_store.add_documents(documents)

rag_pipeline = create_rag_pipeline(embedder, vector_store)
response = rag_pipeline.generate_answer("Show earthquakes in California")
print(response['answer'])
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit UI (app.py)                    │
│            Chat Interface + Interactive Maps                │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│              RAG Pipeline (rag_pipeline.py)                 │
│    Orchestrates Retrieval + LLM Generation                  │
└────────────────────┬────────────────────────────────────────┘
                     │
         ┌───────────┴───────────┐
         │                       │
┌────────▼──────────┐   ┌────────▼────────────┐
│    Retriever      │   │    LLM (OpenAI)     │
│ (retriever.py)    │   │   gpt-3.5-turbo     │
└────────┬──────────┘   └─────────────────────┘
         │
┌────────▼───────────────────────────┐
│    Vector Store (vector_store.py)  │
│    FAISS Index + Metadata          │
└────────┬───────────────────────────┘
         │
┌────────▼──────────────────────────┐
│   Embedder (embedder.py)          │
│   sentence-transformers           │
└────────┬──────────────────────────┘
         │
┌────────▼──────────────────────────┐
│   Data Ingestion                  │
│   • esri_loader.py                │
│   • preprocess.py                 │
└────────┬──────────────────────────┘
         │
┌────────▼──────────────────────────┐
│   External Data Sources           │
│   • USGS Earthquakes              │
│   • NYC Open Data                 │
│   • NREL EV Stations              │
│   • US Census                     │
└───────────────────────────────────┘
```

##  Usage

### Example Queries

The system supports complex geospatial queries across multiple domains:

#### Earthquakes
```
"Show recent earthquakes in California and populations at risk"
"What are the strongest earthquakes in the last 30 days?"
"Find earthquakes above magnitude 5.0 near urban areas"
```

#### Traffic Accidents
```
"Which neighborhoods in NYC have the highest accident rates near schools?"
"Show traffic accidents with fatalities in Manhattan"
"Where are the most dangerous intersections in Brooklyn?"
```

#### Demographics
```
"Which counties in Florida have the highest elderly population?"
"Show population density in California counties"
"Find areas with high population density near fault lines"
```

#### Infrastructure
```
"Where should new EV charging stations go in Boston?"
"Show the distribution of charging stations in Massachusetts"
"Which areas lack adequate EV infrastructure?"
```

#### Multi-Domain Queries
```
"Which counties in Florida are most vulnerable to both flooding and elderly population density?"
"Show earthquake risk combined with population density in California"
"Find high-accident areas near schools in NYC"
```


### Dataset Limits

Control dataset sizes for faster loading during development:

```python
# In config.py or .env
EARTHQUAKE_LIMIT=500
ACCIDENT_LIMIT=2000
DEMOGRAPHICS_LIMIT=100
```

##  Data Sources

| Domain | Source | API | Update Frequency |
|--------|--------|-----|------------------|
| **Earthquakes** | USGS | [Earthquake API](https://earthquake.usgs.gov/fdsnws/event/1/) | Real-time |
| **Traffic Accidents** | NYC Open Data | [Motor Vehicle Collisions](https://data.cityofnewyork.us/) | Daily |
| **Demographics** | US Census Bureau | [Census API](https://www.census.gov/data/developers.html) | Annual |
| **EV Charging** | NREL | [Alternative Fuels API](https://developer.nrel.gov/) | Monthly |

## 🛠️ Development

### Project Structure

```
mapSensef/
├── app.py                      # Streamlit UI
├── rag_pipeline.py             # RAG orchestration
├── config.py                   # Configuration management
├── requirements.txt            # Dependencies
├── .env.example               # Environment template
├── README.md                  # Documentation
│
├── data_ingestion/
│   ├── __init__.py
│   ├── esri_loader.py         # Data fetching from APIs
│   └── preprocess.py          # Data cleaning and text conversion
│
├── indexing/
│   ├── __init__.py
│   ├── embedder.py            # Embedding generation
│   └── vector_store.py        # FAISS vector storage
│
└── retrieval/
    ├── __init__.py
    └── retriever.py           # Semantic search and filtering
```

### Running Tests

Test individual modules:

```bash
# Test data loader
python -m data_ingestion.esri_loader

# Test embeddings
python -m indexing.embedder

# Test vector store
python -m indexing.vector_store

# Test retriever
python -m retrieval.retriever

# Test RAG pipeline
python -m rag_pipeline
```

### Adding New Data Sources

1. Add fetcher method to `data_ingestion/esri_loader.py`:
```python
def fetch_new_dataset(self, params):
    # Fetch data
    # Return GeoDataFrame
```

2. Add preprocessor to `data_ingestion/preprocess.py`:
```python
def _process_new_data(self, gdf):
    # Convert to documents
    # Return list of dicts
```

3. Update `load_all_datasets()` to include new source

### Customizing the UI

Edit `app.py` to:
- Add new tabs
- Customize map styling
- Add analytics charts
- Modify chat interface




**Built with ❤️ using Python, Streamlit, and ESRI datasets**