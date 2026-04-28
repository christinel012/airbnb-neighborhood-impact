# Airbnb & the Housing Crisis

A multi-city analysis of how short-term rental activity shapes neighborhood housing affordability across the United States.

**[Live Dashboard](YOUR_STREAMLIT_URL_HERE)**

---

## Overview

This project analyzes 112,892 Airbnb listings across Los Angeles, New York City, Chicago, Seattle, and Austin — examining how rental activity clusters geographically, which neighborhoods face the most pressure, and whether Airbnb activity correlates with rising long-term rents.

The findings challenge simple narratives: the relationship between short-term rentals and housing affordability is more nuanced than headlines suggest. Austin saw rents fall 10% despite moderate Airbnb activity, while Chicago had the highest rent growth despite a smaller STR market than LA or NYC.

---

## Key Findings

- **No simple Airbnb to rent relationship** at the city level. Occupancy and rent growth are weakly correlated across cities — local housing supply and policy matter more.
- **Commercial hosts dominate everywhere.** 51–69% of listings across all 5 cities come from hosts with multiple properties, not individuals sharing spare rooms.
- **Licensing compliance is low.** Despite regulations in 4 of 5 cities, compliance ranges from 83% in Seattle to 0% in Austin.
- **Neighborhood characteristics predict activity better than city identity.** A LightGBM model (R²=0.708) found that average occupancy and commercial host rate are the strongest predictors — the city a neighborhood is in ranks last.
- **Activity clusters in specific neighborhoods.** Seattle's Belltown, NYC's Bedford-Stuyvesant, and LA's Long Beach show disproportionate Airbnb concentration.

---

## Dashboard

| Tab | Description |
|-----|-------------|
| City Overview | Rent trends (2015–2026), occupancy vs rent growth, city deep dive with photos |
| Neighborhood Map | Interactive choropleth with metric toggle and minimum listings filter |
| Top Neighborhoods | Per-city bar charts with city selector and key observations |
| Model Insights | Interactive SHAP feature importance, predicted vs actual, key findings |
| Summary | Research questions, methodology, findings, future directions, data sources |
| About | Project context and author info |

---

## Tech Stack

| Category | Tools |
|----------|-------|
| Data ingestion | requests, BeautifulSoup, python-dateutil |
| Data processing | pandas, numpy, geopandas |
| Modeling | lightgbm, scikit-learn, shap |
| Visualization | plotly, streamlit |
| Deployment | Streamlit Community Cloud |

---

## Data Sources

- **InsideAirbnb** — listings, calendar, and GeoJSON neighborhood boundaries for 5 US cities (Sep–Nov 2025 snapshots)
- **Zillow Research** — ZORI city-level rent index (2015–2026)

---

## Project Structure

airbnb-neighborhood-impact/
├── app/
│   ├── app.py                    # Streamlit dashboard
│   └── style.css                 # Dashboard styling
├── data/
│   ├── processed/                # Cleaned datasets used by dashboard
│   └── raw/                      # GeoJSON neighborhood boundaries
├── notebooks/
│   ├── 01_eda_listings.ipynb     # Data pipeline and EDA
│   ├── 02_visualizations.ipynb   # Chart exploration
│   └── 03_modeling.ipynb         # LightGBM and SHAP analysis
├── notes/
│   └── findings.md               # Running analysis notes
├── src/
│   └── data_loader.py            # Reusable data ingestion functions
└── requirements.txt

---

## Run Locally

```bash
# Clone the repo
git clone https://github.com/christinel012/airbnb-neighborhood-impact.git
cd airbnb-neighborhood-impact

# Create environment
python -m venv airbnb-env
source airbnb-env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run dashboard
streamlit run app/app.py
```

Note: Raw Airbnb data files are not included in the repo due to size. The `src/data_loader.py` module automatically downloads the latest available data from InsideAirbnb. Run `notebooks/01_eda_listings.ipynb` first to generate the processed data files.

---

## Author

**Christine Li** · Statistics @ UC Davis · Master of Analytics @ UC Berkeley (Fall 2026)

[LinkedIn](https://www.linkedin.com/in/christineli012) · [GitHub](https://github.com/christinel012) · ytingli0210@gmail.com

Open to Data Analytics, AI Product, and Machine Learning roles — internships, new grad, and full-time opportunities.

---

*Built solo as a portfolio project · April 2026*