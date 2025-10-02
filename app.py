"""
FAST MAIN APP - Real Data + Mistral AI 
Loads real USGS earthquakes and NYC accidents, uses Mistral AI directly
"""

import streamlit as st
import folium
from streamlit_folium import st_folium
import os
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from data_ingestion import ESRIDataLoader
import pickle
from pathlib import Path

st.set_page_config(page_title="GeoAI Fast", page_icon="📍", layout="wide")

# Cache for data
CACHE_DIR = Path("./cache")
CACHE_DIR.mkdir(exist_ok=True)
DATA_CACHE = CACHE_DIR / "real_data.pkl"

st.title("GeoAI RAG Assistant")
st.caption("Real-time geospatial intelligence powered by Mistral AI")

# Sidebar
with st.sidebar:
    st.markdown("### Mistral API")

    api_key = st.text_input(
        "API Key",
        value=os.getenv('MISTRAL_API_KEY', ''),
        type="password"
    )

    if api_key:
        os.environ['MISTRAL_API_KEY'] = api_key
        st.success("API Key loaded")
    else:
        st.warning("No API key")
        st.info("[Get FREE key](https://console.mistral.ai/)")

    st.markdown("---")
    st.markdown("### Data Loading")

    if DATA_CACHE.exists():
        st.success("Data cached")
        if st.button("Reload Fresh Data"):
            DATA_CACHE.unlink()
            st.rerun()
    else:
        st.info("Will load fresh data...")

    days_back = st.slider("Earthquake days", 1, 30, 7)
    accident_limit = st.slider("NYC accidents", 100, 2000, 500)

# Load data function
@st.cache_resource
def load_real_data(days: int, limit: int):
    """Load real data from APIs with caching"""

    if DATA_CACHE.exists():
        with open(DATA_CACHE, 'rb') as f:
            data = pickle.load(f)
            if data['config'] == {'days': days, 'limit': limit}:
                return data['earthquakes'], data['accidents']

    # Fresh load
    with st.spinner("Loading fresh data from APIs..."):
        loader = ESRIDataLoader()

        st.info(f"Fetching {days} days of CA earthquakes...")
        eq_gdf = loader.fetch_earthquakes(days_back=days, region='California')

        st.info(f"Fetching {limit} NYC accidents...")
        acc_gdf = loader.fetch_nyc_accidents(limit=limit)

        # Convert to simple dicts for easier handling
        earthquakes = []
        for idx, row in eq_gdf.iterrows():
            try:
                earthquakes.append({
                    'lat': row.geometry.y,
                    'lon': row.geometry.x,
                    'mag': row.get('magnitude', 0),
                    'place': row.get('place', 'Unknown'),
                    'time': row.get('time', 'N/A'),
                    'depth': row.get('depth_km', 0)
                })
            except:
                continue

        accidents = []
        for idx, row in acc_gdf.iterrows():
            try:
                accidents.append({
                    'lat': row.geometry.y,
                    'lon': row.geometry.x,
                    'borough': row.get('borough', 'Unknown'),
                    'date': row.get('crash_date', 'N/A'),
                    'injuries': row.get('number_of_persons_injured', 0),
                    'fatalities': row.get('number_of_persons_killed', 0)
                })
            except:
                continue

        # Cache it
        with open(DATA_CACHE, 'wb') as f:
            pickle.dump({
                'config': {'days': days, 'limit': limit},
                'earthquakes': earthquakes,
                'accidents': accidents
            }, f)

        st.success(f"Loaded {len(earthquakes)} earthquakes, {len(accidents)} accidents")

        return earthquakes, accidents

# Load the data
earthquakes, accidents = load_real_data(days_back, accident_limit)

# Main tabs
tab1, tab2, tab3 = st.tabs(["Chat", "Map", "Stats"])

with tab1:
    st.markdown("### Ask About Real Data")

    query = st.text_input(
        "Your question:",
        placeholder="e.g., What's the strongest earthquake? or Show NYC accident summary"
    )

    if st.button("Ask Mistral AI", type="primary") and query:
        if not os.getenv('MISTRAL_API_KEY'):
            st.error("Please add your Mistral API key in the sidebar")
        else:
            with st.spinner("Asking Mistral AI..."):
                try:
                    # Prepare context from REAL data
                    eq_context = "\n".join([
                        f"- Magnitude {eq['mag']:.1f} earthquake in {eq['place']} (depth: {eq['depth']:.1f}km)"
                        for eq in earthquakes[:20]  # Top 20 to avoid token limits
                    ])

                    acc_context = "\n".join([
                        f"- Accident in {acc['borough']} on {acc['date']}, {acc['injuries']} injured, {acc['fatalities']} killed"
                        for acc in accidents[:20]
                    ])

                    context = f"""Real Earthquake Data (last {days_back} days in California):
{eq_context}

Real NYC Traffic Accident Data ({accident_limit} recent records):
{acc_context}"""

                    # Call Mistral
                    client = MistralClient(api_key=os.getenv('MISTRAL_API_KEY'))

                    messages = [
                        ChatMessage(role="system", content="You are a GeoAI assistant analyzing real earthquake and traffic accident data. Answer questions based on the provided real data."),
                        ChatMessage(role="user", content=f"Question: {query}\n\nReal Data:\n{context}")
                    ]

                    response = client.chat(
                        model="mistral-small-latest",
                        messages=messages,
                        max_tokens=400
                    )

                    answer = response.choices[0].message.content

                    st.success("**Answer:**")
                    st.write(answer)

                    # Show data sources
                    with st.expander("Real Data Used"):
                        st.markdown(f"**{len(earthquakes)} earthquakes** from USGS (last {days_back} days)")
                        st.markdown(f"**{len(accidents)} accidents** from NYC Open Data")
                        st.code(context[:1000] + "..." if len(context) > 1000 else context)

                except Exception as e:
                    st.error(f"Error: {e}")
                    st.info("Check your API key and internet connection")

    # Sample questions
    st.markdown("---")
    st.markdown("**Try these questions:**")
    col1, col2 = st.columns(2)
    with col1:
        st.code("What's the strongest earthquake?")
        st.code("How many earthquakes above magnitude 4?")
    with col2:
        st.code("Which NYC borough had most accidents?")
        st.code("Total injuries from all accidents?")

with tab2:
    st.markdown("### Interactive Map - Real Data")

    data_type = st.radio("Show:", ["Earthquakes", "Accidents", "Both"], horizontal=True)

    # Create map centered on US
    m = folium.Map(location=[37.0, -95.0], zoom_start=4)

    # Add earthquakes
    if data_type in ["Earthquakes", "Both"]:
        for eq in earthquakes:
            folium.Marker(
                location=[eq['lat'], eq['lon']],
                popup=f"<b>Magnitude {eq['mag']:.1f}</b><br>{eq['place']}<br>Depth: {eq['depth']:.1f}km",
                icon=folium.Icon(color='red', icon='warning', prefix='fa'),
                tooltip=f"M{eq['mag']:.1f} - {eq['place']}"
            ).add_to(m)

    # Add accidents
    if data_type in ["Accidents", "Both"]:
        for acc in accidents:
            folium.Marker(
                location=[acc['lat'], acc['lon']],
                popup=f"<b>{acc['borough']}</b><br>{acc['date']}<br>{acc['injuries']} injuries, {acc['fatalities']} deaths",
                icon=folium.Icon(color='orange', icon='car', prefix='fa'),
                tooltip=f"Accident in {acc['borough']}"
            ).add_to(m)

    st_folium(m, width=None, height=500)

    # Stats
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Earthquakes", len(earthquakes))
    with col2:
        st.metric("Accidents", len(accidents))
    with col3:
        if earthquakes:
            avg_mag = sum(eq['mag'] for eq in earthquakes) / len(earthquakes)
            st.metric("Avg Magnitude", f"{avg_mag:.2f}")
        else:
            st.metric("Avg Magnitude", "N/A")
    with col4:
        total_injuries = sum(acc['injuries'] for acc in accidents)
        st.metric("Total Injuries", total_injuries)

with tab3:
    st.markdown("### Statistics - Real Data")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Earthquake Magnitudes")
        if earthquakes:
            mags = [eq['mag'] for eq in earthquakes]
            import plotly.graph_objects as go
            fig = go.Figure(data=[go.Histogram(x=mags, nbinsx=20)])
            fig.update_layout(
                xaxis_title="Magnitude",
                yaxis_title="Count",
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown(f"**Range:** {min(mags):.1f} - {max(mags):.1f}")
            st.markdown(f"**Average:** {sum(mags)/len(mags):.2f}")
        else:
            st.info("No earthquake data")

    with col2:
        st.markdown("#### Accidents by Borough")
        if accidents:
            boroughs = {}
            for acc in accidents:
                b = acc['borough']
                boroughs[b] = boroughs.get(b, 0) + 1

            import plotly.express as px
            fig = px.bar(
                x=list(boroughs.keys()),
                y=list(boroughs.values()),
                labels={'x': 'Borough', 'y': 'Count'}
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown(f"**Total Boroughs:** {len(boroughs)}")
        else:
            st.info("No accident data")

# Footer

