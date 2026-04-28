import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import os

# --- Page config ---
st.set_page_config(
    page_title="Airbnb & Housing | Christine Li",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Custom CSS ---
def load_css():
    css_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'style.css')
    with open(css_path, 'r') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css()

# --- Load data ---
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@st.cache_data
def load_data():
    city_analysis = pd.read_csv(f'{BASE}/data/processed/city_analysis.csv')
    neighborhood = pd.read_csv(f'{BASE}/data/processed/neighborhood_activity.csv')
    rent_ts = pd.read_csv(f'{BASE}/data/processed/city_rent_timeseries.csv')
    rent_ts['date'] = pd.to_datetime(rent_ts['date'])
    shap_importance = pd.read_csv(f'{BASE}/data/processed/shap_importance.csv')
    predictions = pd.read_csv(f'{BASE}/data/processed/model_predictions.csv')
    return city_analysis, neighborhood, rent_ts, shap_importance, predictions

@st.cache_data
def load_geodata():
    cities = ['los-angeles', 'new-york-city', 'chicago', 'seattle', 'austin']
    geo = {}
    for city in cities:
        path = f'{BASE}/data/raw/{city}/neighbourhoods.geojson'
        with open(path, 'r') as f:
            geo[city] = json.load(f)
    return geo

city_analysis, neighborhood, rent_ts, shap_importance, predictions = load_data()
geo_data = load_geodata()

# --- Constants ---
city_labels = {
    'los-angeles': 'Los Angeles',
    'new-york-city': 'New York City',
    'chicago': 'Chicago',
    'seattle': 'Seattle',
    'austin': 'Austin'
}

city_centers = {
    'los-angeles': {'lat': 34.05, 'lon': -118.25, 'zoom': 9},
    'new-york-city': {'lat': 40.71, 'lon': -74.00, 'zoom': 10},
    'chicago': {'lat': 41.83, 'lon': -87.65, 'zoom': 10},
    'seattle': {'lat': 47.61, 'lon': -122.33, 'zoom': 10},
    'austin': {'lat': 30.25, 'lon': -97.75, 'zoom': 10}
}

city_photos = {
    'los-angeles': 'https://images.unsplash.com/photo-1534190760961-74e8c1c5c3da?w=800&q=80',
    'new-york-city': 'https://images.unsplash.com/photo-1546436836-07a91091f160?w=800&q=80',
    'chicago': 'https://images.unsplash.com/photo-1477959858617-67f85cf4f1df?w=800&q=80',
    'seattle': 'https://images.unsplash.com/photo-1502175353174-a7a70e73b362?w=800&q=80',
    'austin': 'https://images.unsplash.com/photo-1531218150217-54595bc2b934?w=800&q=80'
}

PLOTLY_THEME = dict(
    template='plotly_dark',
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(13,27,42,0.5)',
    font=dict(family='Inter', color='#e0e8f0'),
    xaxis=dict(gridcolor='rgba(255,255,255,0.05)', linecolor='rgba(255,255,255,0.05)'),
    yaxis=dict(gridcolor='rgba(255,255,255,0.05)', linecolor='rgba(255,255,255,0.05)'),
)

rent_ts['city_label'] = rent_ts['city'].map(city_labels)
city_analysis['city_label'] = city_analysis['city'].map(city_labels)
neighborhood['city_label'] = neighborhood['city'].map(city_labels)

# --- Hero ---
total_listings = city_analysis['total_listings'].sum()
avg_rent = city_analysis['rent_latest'].mean()

st.markdown(f"""
<div class="hero animate-in">
    <div style="display:flex;gap:2.5rem;align-items:center">
        <div style="flex:1.2">
            <div class="hero-eyebrow">Data Analysis · 5 US Cities · 2015–2026</div>
            <div style="font-size:0.8rem;color:#8ba3bc;margin-bottom:0.8rem;font-weight:400">
                by <span style="color:#e0e8f0;font-weight:500">Christine Li</span> · 
                UC Davis Statistics '26 · UC Berkeley MAnalytics '27
            </div>
            <div class="hero-title">Airbnb & the<br><span>Housing Crisis</span></div>
            <div class="hero-body">
                Short-term rentals have reshaped American cities over the past decade. 
                This project analyzes <strong style="color:#e0e8f0">112,892 Airbnb listings</strong> across 
                Los Angeles, New York City, Chicago, Seattle, and Austin — examining how 
                rental activity clusters geographically, which neighborhoods face the most 
                pressure, and whether Airbnb activity correlates with rising long-term rents.<br><br>
                The findings challenge simple narratives: the relationship between 
                short-term rentals and housing affordability is more nuanced than headlines suggest.
            </div>
            <div class="hero-stats">
                <div class="hero-stat">
                    <div class="hero-stat-number">112,892</div>
                    <div class="hero-stat-label">Listings analyzed</div>
                </div>
                <div class="hero-stat">
                    <div class="hero-stat-number">5</div>
                    <div class="hero-stat-label">US cities</div>
                </div>
                <div class="hero-stat">
                    <div class="hero-stat-number">596</div>
                    <div class="hero-stat-label">Neighborhoods</div>
                </div>
                <div class="hero-stat">
                    <div class="hero-stat-number">11yrs</div>
                    <div class="hero-stat-label">Rent history</div>
                </div>
                <div class="hero-stat">
                    <div class="hero-stat-number">0.71</div>
                    <div class="hero-stat-label">Model R²</div>
                </div>
            </div>
        </div>
<div style="flex:0.8;position:relative;height:380px;min-width:300px">
<!-- NYC - large back -->
            <div style="position:absolute;top:0;right:20px;width:240px;
                        transform:rotate(3deg);
                        animation:popIn 0.6s ease 0.1s both;
                        filter:drop-shadow(0 8px 24px rgba(0,0,0,0.5))">
                <div style="background:#1a1a1a;padding:8px 8px 32px 8px;
                            border-radius:4px;transition:transform 0.3s ease;
                            cursor:pointer"
                     onmouseover="this.style.transform='scale(1.04)'"
                     onmouseout="this.style.transform='scale(1)'">
                    <img src="{city_photos['new-york-city']}" 
                         style="width:100%;height:150px;object-fit:cover;display:block"/>
                    <div style="font-size:0.65rem;color:#aaa;text-align:center;
                                margin-top:6px;font-family:'Inter',sans-serif">
                        New York City
                    </div>
                </div>
            </div>
            <!-- LA -->
            <div style="position:absolute;top:60px;left:0;width:180px;
                        transform:rotate(-4deg);
                        animation:popIn 0.6s ease 0.25s both;
                        filter:drop-shadow(0 8px 24px rgba(0,0,0,0.5))">
                <div style="background:#1a1a1a;padding:8px 8px 32px 8px;
                            border-radius:4px;transition:transform 0.3s ease;
                            cursor:pointer"
                     onmouseover="this.style.transform='scale(1.04)'"
                     onmouseout="this.style.transform='scale(1)'">
                    <img src="{city_photos['los-angeles']}" 
                         style="width:100%;height:120px;object-fit:cover;display:block"/>
                    <div style="font-size:0.65rem;color:#aaa;text-align:center;
                                margin-top:6px;font-family:'Inter',sans-serif">
                        Los Angeles
                    </div>
                </div>
            </div>
            <!-- Chicago -->
            <div style="position:absolute;bottom:20px;right:0;width:190px;
                        transform:rotate(2deg);
                        animation:popIn 0.6s ease 0.4s both;
                        filter:drop-shadow(0 8px 24px rgba(0,0,0,0.5))">
                <div style="background:#1a1a1a;padding:8px 8px 32px 8px;
                            border-radius:4px;transition:transform 0.3s ease;
                            cursor:pointer"
                     onmouseover="this.style.transform='scale(1.04)'"
                     onmouseout="this.style.transform='scale(1)'">
                    <img src="{city_photos['chicago']}" 
                         style="width:100%;height:120px;object-fit:cover;display:block"/>
                    <div style="font-size:0.65rem;color:#aaa;text-align:center;
                                margin-top:6px;font-family:'Inter',sans-serif">
                        Chicago
                    </div>
                </div>
            </div>
            <!-- Seattle -->
            <div style="position:absolute;bottom:10px;left:20px;width:160px;
                        transform:rotate(-2deg);
                        animation:popIn 0.6s ease 0.55s both;
                        filter:drop-shadow(0 8px 24px rgba(0,0,0,0.5))">
                <div style="background:#1a1a1a;padding:8px 8px 32px 8px;
                            border-radius:4px;transition:transform 0.3s ease;
                            cursor:pointer"
                     onmouseover="this.style.transform='scale(1.04)'"
                     onmouseout="this.style.transform='scale(1)'">
                    <img src="{city_photos['seattle']}" 
                         style="width:100%;height:100px;object-fit:cover;display:block"/>
                    <div style="font-size:0.65rem;color:#aaa;text-align:center;
                                margin-top:6px;font-family:'Inter',sans-serif">
                        Seattle
                    </div>
                </div>
            </div>
            <!-- Austin -->
            <div style="position:absolute;top:150px;left:120px;width:150px;
                        transform:rotate(1deg);
                        animation:popIn 0.6s ease 0.7s both;
                        filter:drop-shadow(0 8px 24px rgba(0,0,0,0.5))">
                <div style="background:#1a1a1a;padding:8px 8px 32px 8px;
                            border-radius:4px;transition:transform 0.3s ease;
                            cursor:pointer"
                     onmouseover="this.style.transform='scale(1.04)'"
                     onmouseout="this.style.transform='scale(1)'">
                    <img src="{city_photos['austin']}" 
                         style="width:100%;height:100px;object-fit:cover;display:block"/>
                    <div style="font-size:0.65rem;color:#aaa;text-align:center;
                                margin-top:6px;font-family:'Inter',sans-serif">
                        Austin
                    </div>
                </div>
            </div>
            <!-- Stat callout 1 -->
            <div style="position:absolute;top:10px;left:30px;
                        background:#ff6b6b;border-radius:8px;
                        padding:0.5rem 0.8rem;transform:rotate(-3deg);
                        animation:popIn 0.5s ease 0.8s both;
                        filter:drop-shadow(0 4px 12px rgba(255,107,107,0.4))">
                <div style="font-family:'Playfair Display',serif;
                            font-size:1.3rem;color:#fff;font-weight:900;line-height:1">
                    112K+
                </div>
                <div style="font-size:0.65rem;color:rgba(255,255,255,0.85);
                            text-transform:uppercase;letter-spacing:0.08em">
                    Listings
                </div>
            </div>
            <!-- Stat callout 2 -->
            <div style="position:absolute;bottom:120px;right:10px;
                        background:#ffd43b;border-radius:8px;
                        padding:0.5rem 0.8rem;transform:rotate(4deg);
                        animation:popIn 0.5s ease 0.9s both;
                        filter:drop-shadow(0 4px 12px rgba(255,212,59,0.3))">
                <div style="font-family:'Playfair Display',serif;
                            font-size:1.3rem;color:#1a1a1a;font-weight:900;line-height:1">
                    596
                </div>
                <div style="font-size:0.65rem;color:#555;
                            text-transform:uppercase;letter-spacing:0.08em">
                    Neighborhoods
                </div>
            </div>
            <!-- Stat callout 1 -->
            <div style="position:absolute;top:10px;left:30px;
                        background:#ff6b6b;border-radius:8px;
                        padding:0.5rem 0.8rem;transform:rotate(-3deg);
                        filter:drop-shadow(0 4px 12px rgba(255,107,107,0.4))">
                <div style="font-family:'Playfair Display',serif;
                            font-size:1.3rem;color:#fff;font-weight:900;
                            line-height:1">112K+</div>
                <div style="font-size:0.65rem;color:rgba(255,255,255,0.85);
                            text-transform:uppercase;letter-spacing:0.08em">
                    Listings
                </div>
            </div>
            <!-- Stat callout 2 -->
            <div style="position:absolute;bottom:120px;right:10px;
                        background:#ffd43b;border-radius:8px;
                        padding:0.5rem 0.8rem;transform:rotate(4deg);
                        filter:drop-shadow(0 4px 12px rgba(255,212,59,0.3))">
                <div style="font-family:'Playfair Display',serif;
                            font-size:1.3rem;color:#1a1a1a;font-weight:900;
                            line-height:1">596</div>
                <div style="font-size:0.65rem;color:#555;
                            text-transform:uppercase;letter-spacing:0.08em">
                    Neighborhoods
                </div>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- Tabs ---
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🏙️  City Overview",
    "🗺️  Neighborhood Map",
    "🏆  Top Neighborhoods",
    "🤖  Model Insights",
    "📋  Summary",
    "👩‍💻  About"
])

# ── Tab 1: City Overview ──
with tab1:
    st.markdown('<div class="section-intro">A high-level comparison of Airbnb activity and rental market trends across the five cities. Key metrics include current rent levels, 3-year rent growth, average occupancy, and the share of commercial hosts — operators who manage multiple listings and have an outsized impact on housing supply.</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-label">Current Rent & 3-Year Growth</div>', unsafe_allow_html=True)

    cols = st.columns(5)
    city_order = ['austin', 'chicago', 'los-angeles', 'new-york-city', 'seattle']
    for i, city in enumerate(city_order):
        row = city_analysis[city_analysis['city'] == city].iloc[0]
        growth = row['rent_growth_3yr_pct']
        delta_class = 'metric-positive' if growth > 0 else 'metric-negative'
        delta_symbol = '↑' if growth > 0 else '↓'
        with cols[i]:
            st.markdown(f"""
            <div class="metric-card animate-in" style="animation-delay:{i*0.1}s">
                <div class="metric-city">{city_labels[city]}</div>
                <div class="metric-rent">${row['rent_latest']:,.0f}<span style="font-size:0.85rem;color:#8ba3bc">/mo</span></div>
                <div class="{delta_class}">{delta_symbol} {abs(growth):.1f}% (3yr)</div>
                <div style="font-size:0.75rem;color:#8ba3bc;margin-top:0.4rem">
                    {row['avg_occupancy']:.0f} days avg · {row['commercial_pct']:.0f}% commercial
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown('<div class="section-label">Rental Market Trends (2015–2026)</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-intro" style="font-size:0.85rem">Zillow Observed Rent Index (ZORI) — monthly median asking rent per city. Note the COVID-19 dip in 2020 and subsequent surge, with Austin uniquely reversing course after 2022 due to a housing supply boom.</div>', unsafe_allow_html=True)
        fig = px.line(
            rent_ts, x='date', y='rent', color='city_label',
            labels={'rent': 'Median Rent ($)', 'date': '', 'city_label': 'City'},
            color_discrete_sequence=['#ff6b6b', '#ffd43b', '#69db7c', '#74c0fc', '#da77f2']
        )
        fig.add_vline(x=pd.Timestamp('2020-03-01').timestamp() * 1000,
                      line_dash='dash', line_color='rgba(255,255,255,0.2)',
                      annotation_text='COVID-19', annotation_font_color='#8ba3bc')
        fig.add_vline(x=pd.Timestamp('2022-01-01').timestamp() * 1000,
                      line_dash='dash', line_color='rgba(255,255,255,0.1)',
                      annotation_text='Rent surge', annotation_font_color='#8ba3bc')
        fig.update_layout(**PLOTLY_THEME, hovermode='x unified', height=380,
                          legend=dict(orientation='h', y=1.1,
                                      font=dict(size=11)))
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown('<div class="section-label">Occupancy vs Rent Growth</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-intro" style="font-size:0.85rem">Does higher Airbnb occupancy lead to higher rent growth? The relationship is surprisingly weak — Austin has moderate occupancy but falling rents, while Chicago has high rent growth despite a smaller Airbnb market.</div>', unsafe_allow_html=True)
        fig2 = px.scatter(
            city_analysis,
            x='avg_occupancy', y='rent_growth_3yr_pct',
            text='city_label',
            size='total_listings',
            color='commercial_pct',
            color_continuous_scale='Reds',
            range_color=[45, 75],
            labels={
                'avg_occupancy': 'Avg Occupancy (days/year)',
                'rent_growth_3yr_pct': 'Rent Growth (%)',
                'commercial_pct': 'Commercial %',
            }
        )
        fig2.update_traces(textposition='top center',
                           textfont=dict(color='#e0e8f0', size=11))
        fig2.add_hline(y=0, line_dash='dash',
                       line_color='rgba(255,255,255,0.15)')
        fig2.update_layout(**PLOTLY_THEME, height=380,
                           coloraxis_colorbar=dict(
                               title=dict(text='Comm %',
                                         font=dict(color='#8ba3bc')),
                               tickfont=dict(color='#8ba3bc')
                           ))
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    # Rent trajectory per city
    st.markdown('<div class="section-label">City Deep Dive</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-intro" style="font-size:0.85rem">Select a city to explore its full rental trajectory alongside key Airbnb metrics.</div>', unsafe_allow_html=True)

    col_city_sel, _ = st.columns([2, 3])
    with col_city_sel:
        selected_overlay_city = st.selectbox(
            "", options=list(city_labels.keys()),
            format_func=lambda x: city_labels[x],
            label_visibility='collapsed',
            key='overlay_city'
        )

    city_rent_ts = rent_ts[rent_ts['city'] == selected_overlay_city].copy()
    city_row = city_analysis[city_analysis['city'] == selected_overlay_city].iloc[0]

    col_photo, col_chart = st.columns([1, 3])

    with col_photo:
        st.markdown(f"""
        <div class="city-img-card">
            <img src="{city_photos[selected_overlay_city]}" 
                 alt="{city_labels[selected_overlay_city]}"/>
            <div class="city-img-overlay">
                <div style="font-family:'Playfair Display',serif;font-size:1.2rem;
                            color:#fff;font-weight:700">
                    {city_labels[selected_overlay_city]}
                </div>
                <div style="font-size:0.75rem;color:#ccc;margin-top:0.2rem">
                    {city_row['total_listings']:,} listings · 
                    {city_row['avg_occupancy']:.0f} days avg occupancy
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        ctx_items = [
            ("Current Rent", f"${city_row['rent_latest']:,.0f}/mo"),
            ("3yr Growth", f"{'↑' if city_row['rent_growth_3yr_pct'] > 0 else '↓'}{abs(city_row['rent_growth_3yr_pct']):.1f}%"),
            ("Avg Occupancy", f"{city_row['avg_occupancy']:.0f} days/yr"),
            ("Commercial Hosts", f"{city_row['commercial_pct']:.0f}%"),
            ("Licensed Listings", f"{city_row['licensed_pct']:.0f}%"),
        ]
        for label, value in ctx_items:
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;
                        padding:0.5rem 0;
                        border-bottom:1px solid rgba(255,255,255,0.05);
                        font-size:0.85rem">
                <span style="color:#8ba3bc">{label}</span>
                <span style="color:#ffffff;font-weight:500">{value}</span>
            </div>
            """, unsafe_allow_html=True)

    with col_chart:
        fig_overlay = go.Figure()
        fig_overlay.add_trace(go.Scatter(
            x=city_rent_ts['date'], y=city_rent_ts['rent'],
            name='Median Rent ($)',
            line=dict(color='#ff6b6b', width=2.5),
            fill='tozeroy',
            fillcolor='rgba(255,107,107,0.05)'
        ))
        fig_overlay.add_vrect(
            x0='2021-01-01', x1='2023-06-01',
            fillcolor='rgba(255,107,107,0.06)',
            line_width=0,
            annotation_text='Peak STR growth',
            annotation_position='top left',
            annotation_font_color='#8ba3bc',
            annotation_font_size=11
        )
        fig_overlay.add_vline(
            x=pd.Timestamp('2020-03-01').timestamp() * 1000,
            line_dash='dash', line_color='rgba(255,255,255,0.15)',
            annotation_text='COVID-19',
            annotation_font_color='#8ba3bc'
        )
        latest = city_rent_ts.iloc[-1]
        fig_overlay.add_annotation(
            x=latest['date'], y=latest['rent'],
            text=f"${latest['rent']:,.0f}",
            showarrow=True, arrowhead=2,
            arrowcolor='#ff6b6b',
            font=dict(color='#ff6b6b', size=12),
            bgcolor='#152236',
            bordercolor='#ff6b6b',
            borderwidth=1,
            borderpad=4
        )
        fig_overlay.update_layout(
            **PLOTLY_THEME, height=320,
            hovermode='x unified',
            showlegend=False,
            margin=dict(t=20, b=20)
        )
        fig_overlay.update_yaxes(title='Median Rent ($)',
                                  title_font=dict(color='#8ba3bc'))
        st.plotly_chart(fig_overlay, use_container_width=True)

    st.divider()

    st.markdown('<div class="section-label">City Summary</div>', unsafe_allow_html=True)
    table_rows = []
    for _, row in city_analysis.iterrows():
        growth = row['rent_growth_3yr_pct']
        growth_color = '#ff6b6b' if growth > 0 else '#51cf66'
        growth_symbol = '↑' if growth > 0 else '↓'
        table_rows.append(
            f"<tr>"
            f"<td style='text-align:left;padding:0.85rem 1.2rem;font-weight:600;color:#fff'>{city_labels[row['city']]}</td>"
            f"<td style='text-align:right;padding:0.85rem 1rem'>{row['total_listings']:,.0f}</td>"
            f"<td style='text-align:right;padding:0.85rem 1rem'>{row['avg_occupancy']:.0f} days</td>"
            f"<td style='text-align:right;padding:0.85rem 1rem'>${row['median_price']:,.0f}</td>"
            f"<td style='text-align:right;padding:0.85rem 1rem'>{row['commercial_pct']:.0f}%</td>"
            f"<td style='text-align:right;padding:0.85rem 1rem'>{row['licensed_pct']:.0f}%</td>"
            f"<td style='text-align:right;padding:0.85rem 1rem;color:#fff'>${row['rent_latest']:,.0f}</td>"
            f"<td style='text-align:right;padding:0.85rem 1rem;color:{growth_color};font-weight:600'>{growth_symbol} {abs(growth):.1f}%</td>"
            f"</tr>"
        )

    headers = ['City', 'Listings', 'Avg Occupancy', 'Median Price',
            'Commercial %', 'Licensed %', 'Current Rent', '3yr Growth']
    header_html = ''.join(
        f"<th style='padding:0.9rem 1rem;text-align:{'left' if i==0 else 'right'};"
        f"color:#8ba3bc;font-weight:500;font-size:0.72rem;"
        f"text-transform:uppercase;letter-spacing:0.08em;padding-left:{'1.2rem' if i==0 else '1rem'}'>"
        f"{h}</th>"
        for i, h in enumerate(headers)
    )

    st.markdown(
        f"<div style='background:linear-gradient(135deg,#152236 0%,#1a2d45 100%);"
        f"border:1px solid rgba(255,255,255,0.06);border-radius:14px;overflow:hidden'>"
        f"<table style='width:100%;border-collapse:collapse;font-size:0.88rem;color:#8ba3bc'>"
        f"<thead><tr style='border-bottom:1px solid rgba(255,255,255,0.08)'>{header_html}</tr></thead>"
        f"<tbody>{''.join(table_rows)}</tbody>"
        f"</table></div>",
        unsafe_allow_html=True
    )

    st.markdown("""
    <div style="display:flex;gap:0.8rem;flex-wrap:wrap;margin-top:1rem">
        <div class="finding-card" style="flex:1;min-width:180px;margin-bottom:0">
            <div style="font-weight:600;color:#fff;margin-bottom:0.25rem;font-size:0.82rem">
                📉 Austin is the outlier
            </div>
            <div class="finding-text" style="font-size:0.8rem">
                Only city with falling rents (-10.1%) — a housing supply boom absorbed demand faster than it grew.
            </div>
        </div>
        <div class="finding-card" style="flex:1;min-width:180px;margin-bottom:0">
            <div style="font-weight:600;color:#fff;margin-bottom:0.25rem;font-size:0.82rem">
                🏢 Chicago most commercial
            </div>
            <div class="finding-text" style="font-size:0.8rem">
                69% commercial hosts — highest in sample. Property managers dominate, not individuals.
            </div>
        </div>
        <div class="finding-card" style="flex:1;min-width:180px;margin-bottom:0">
            <div style="font-weight:600;color:#fff;margin-bottom:0.25rem;font-size:0.82rem">
                🌊 LA largest market
            </div>
            <div class="finding-text" style="font-size:0.8rem">
                45,886 listings — biggest market in the sample. Moderate rent growth (+4.1%) despite scale, only 28% licensed.
            </div>
        </div>
                <div class="finding-card" style="flex:1;min-width:180px;margin-bottom:0">
            <div style="font-weight:600;color:#fff;margin-bottom:0.25rem;font-size:0.82rem">
                🏙️ NYC most expensive
            </div>
            <div class="finding-text" style="font-size:0.8rem">
                $3,811/mo and still climbing +14% — strict 2023 STR regulations haven't stopped rent growth.
            </div>
        </div>
        <div class="finding-card" style="flex:1;min-width:180px;margin-bottom:0">
            <div style="font-weight:600;color:#fff;margin-bottom:0.25rem;font-size:0.82rem">
                📋 Seattle most compliant
            </div>
            <div class="finding-text" style="font-size:0.8rem">
                83% licensed — far above LA (28%) and NYC (15%). Strong enforcement, rents still up 6.4%.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Tab 2: Neighborhood Map ──
with tab2:
    st.markdown('<div class="section-intro">Explore Airbnb activity at the neighborhood level. Use the controls to switch between cities, change the color metric, and filter by minimum listing count to focus on neighborhoods with meaningful data. Click any neighborhood on the map to see its details.</div>', unsafe_allow_html=True)

    ctrl1, ctrl2, ctrl3 = st.columns([2, 2, 2])

    with ctrl1:
        st.markdown('<div class="section-label">Select City</div>', unsafe_allow_html=True)
        selected_city = st.selectbox(
            "", options=list(city_labels.keys()),
            format_func=lambda x: city_labels[x],
            label_visibility='collapsed'
        )

    with ctrl2:
        st.markdown('<div class="section-label">Color Metric</div>', unsafe_allow_html=True)
        metric_options = {
            'activity_score': 'Activity Score',
            'avg_occupancy': 'Avg Occupancy (days)',
            'commercial_pct': 'Commercial Host %',
            'avg_availability': 'Avg Availability (days)'
        }
        selected_metric = st.selectbox(
            "", options=list(metric_options.keys()),
            format_func=lambda x: metric_options[x],
            label_visibility='collapsed'
        )

    with ctrl3:
        st.markdown('<div class="section-label">Min Listings</div>', unsafe_allow_html=True)
        min_listings = st.slider(
            "", min_value=10, max_value=200,
            value=10, step=10,
            label_visibility='collapsed'
        )

    city_nb = neighborhood[
        (neighborhood['city'] == selected_city) &
        (neighborhood['total_listings'] >= min_listings)
    ].copy()

    col_map, col_info = st.columns([3, 1])

    with col_map:
        fig3 = px.choropleth_map(
            city_nb,
            geojson=geo_data[selected_city],
            locations='neighbourhood_cleansed',
            featureidkey='properties.neighbourhood',
            color=selected_metric,
            color_continuous_scale='Reds',
            map_style='carto-darkmatter',
            zoom=city_centers[selected_city]['zoom'],
            center={'lat': city_centers[selected_city]['lat'],
                    'lon': city_centers[selected_city]['lon']},
            opacity=0.8,
            labels={selected_metric: metric_options[selected_metric]},
            hover_data={
                'neighbourhood_cleansed': False,
                'total_listings': True,
                'avg_occupancy': True,
                'commercial_pct': True,
                'activity_score': True
            }
        )
        fig3.update_traces(
            hovertemplate=(
                "<b>%{location}</b><br>"
                "Activity Score: %{customdata[3]:.3f}<br>"
                "Listings: %{customdata[0]:.0f}<br>"
                "Avg Occupancy: %{customdata[1]:.0f} days<br>"
                "Commercial: %{customdata[2]:.0%}<br>"
                "<extra></extra>"
            )
        )
        fig3.update_layout(
            height=600,
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            coloraxis_colorbar=dict(
                title=dict(text=metric_options[selected_metric],
                          font=dict(color='#8ba3bc')),
                tickfont=dict(color='#8ba3bc')
            )
        )
        st.plotly_chart(fig3, use_container_width=True)

        # City stats below map
        st.markdown('<div class="section-label" style="margin-top:1rem">City Stats</div>', 
                    unsafe_allow_html=True)
        s1, s2, s3, s4 = st.columns(4)
        stat_data = [
            (s1, "Neighborhoods", f"{len(city_nb)}"),
            (s2, "Total Listings", f"{city_nb['total_listings'].sum():,.0f}"),
            (s3, "Avg Occupancy", f"{city_nb['avg_occupancy'].mean():.0f} days"),
            (s4, "Avg Activity Score", f"{city_nb['activity_score'].mean():.3f}"),
        ]
        for i, (col, label, value) in enumerate(stat_data):
            with col:
                st.markdown(f"""
                <div class="metric-card animate-in" 
                     style="padding:0.9rem 1rem;margin-bottom:0;
                            animation-delay:{i*0.1}s">
                    <div class="metric-city">{label}</div>
                    <div style="font-family:'Playfair Display',serif;
                                font-size:1.4rem;color:#fff">{value}</div>
                </div>
                """, unsafe_allow_html=True)

    with col_info:
        # City photo
        st.markdown(f"""
        <div class="animate-in" style="border-radius:12px;overflow:hidden;
                    height:110px;margin-bottom:1rem;position:relative">
            <img src="{city_photos[selected_city]}" 
                 style="width:100%;height:100%;object-fit:cover;
                        filter:brightness(0.7);display:block;
                        transition:filter 0.3s ease"/>
            <div style="position:absolute;bottom:0;left:0;right:0;padding:0.7rem;
                        background:linear-gradient(transparent,rgba(0,0,0,0.85))">
                <div style="font-family:'Playfair Display',serif;font-size:1rem;
                            color:#fff;font-weight:700">{city_labels[selected_city]}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-label">Top 5 Neighborhoods</div>',
                    unsafe_allow_html=True)
        top5 = city_nb.nlargest(5, selected_metric)
        for i, (_, row) in enumerate(top5.iterrows()):
            st.markdown(f"""
            <div class="finding-card animate-in" style="animation-delay:{i*0.08}s">
                <div style="font-weight:600;color:#fff;margin-bottom:0.3rem;font-size:0.88rem">
                    {row['neighbourhood_cleansed']}
                </div>
                <div class="finding-text">
                    {metric_options[selected_metric]}: 
                    <strong style="color:#ff6b6b">{row[selected_metric]:.2f}</strong><br>
                    Listings: {row['total_listings']:.0f} · 
                    Occ: {row['avg_occupancy']:.0f}d
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="section-label" style="margin-top:1.5rem">All Neighborhoods</div>',
                unsafe_allow_html=True)

    display_cols = ['neighbourhood_cleansed', 'total_listings',
                    'avg_occupancy', 'commercial_pct', 'activity_score']

    table_rows = []
    for _, row in city_nb[display_cols].sort_values(
            selected_metric, ascending=False).iterrows():
        table_rows.append(
            f"<tr>"
            f"<td style='text-align:left;padding:0.7rem 1.2rem;color:#fff'>"
            f"{row['neighbourhood_cleansed']}</td>"
            f"<td style='text-align:right;padding:0.7rem 1rem'>"
            f"{row['total_listings']:.0f}</td>"
            f"<td style='text-align:right;padding:0.7rem 1rem'>"
            f"{row['avg_occupancy']:.0f} days</td>"
            f"<td style='text-align:right;padding:0.7rem 1rem'>"
            f"{row['commercial_pct']*100:.0f}%</td>"
            f"<td style='text-align:right;padding:0.7rem 1rem;color:#ff6b6b;font-weight:600'>"
            f"{row['activity_score']:.3f}</td>"
            f"</tr>"
        )

    headers = ['Neighborhood', 'Listings', 'Avg Occupancy',
               'Commercial %', 'Activity Score']
    header_html = ''.join(
        f"<th style='padding:0.8rem {'1.2rem' if i==0 else '1rem'};"
        f"text-align:{'left' if i==0 else 'right'};"
        f"color:#8ba3bc;font-weight:500;font-size:0.7rem;"
        f"text-transform:uppercase;letter-spacing:0.08em'>{h}</th>"
        for i, h in enumerate(headers)
    )

    st.markdown(
        f"<div style='background:linear-gradient(135deg,#152236 0%,#1a2d45 100%);"
        f"border:1px solid rgba(255,255,255,0.06);border-radius:14px;"
        f"overflow:hidden;max-height:400px;overflow-y:auto'>"
        f"<table style='width:100%;border-collapse:collapse;"
        f"font-size:0.85rem;color:#8ba3bc'>"
        f"<thead style='position:sticky;top:0;background:#152236'>"
        f"<tr style='border-bottom:1px solid rgba(255,255,255,0.08)'>"
        f"{header_html}</tr></thead>"
        f"<tbody>{''.join(table_rows)}</tbody>"
        f"</table></div>",
        unsafe_allow_html=True
    )

# ── Tab 3: Top Neighborhoods ──
with tab3:
    st.markdown('<div class="section-intro">The five highest-activity neighborhoods in each city, ranked by the composite activity score. Activity score combines listing density (40%), average occupancy (40%), and commercial host rate (20%), normalized within each city for fair comparison.</div>', unsafe_allow_html=True)

    st.markdown('<div style="font-size:1rem;color:#555;margin-bottom:1.5rem">* Austin uses postal codes as neighborhood boundaries</div>', unsafe_allow_html=True)

    city_colors = {
        'los-angeles': '#ff6b6b',
        'new-york-city': '#74c0fc',
        'chicago': '#ffd43b',
        'seattle': '#da77f2',
        'austin': '#69db7c'
    }

    top_neighborhoods = (
        neighborhood
        .groupby('city')
        .apply(lambda x: x.nlargest(5, 'activity_score'), include_groups=False)
        .reset_index()
    )

    cities_list = list(city_labels.keys())
    
    # Render in pairs
    col_sel, _ = st.columns([2, 3])
    with col_sel:
        st.markdown('<div class="section-label">Select Cities to Compare</div>', unsafe_allow_html=True)
        selected_cities = st.multiselect(
            "",
            options=list(city_labels.keys()),
            default=list(city_labels.keys())[:4],
            format_func=lambda x: city_labels[x],
            label_visibility='collapsed'
        )

    if not selected_cities:
        st.markdown('<div class="finding-text" style="color:#555;padding:2rem 0">Select at least one city above.</div>', unsafe_allow_html=True)
    else:
        for i in range(0, len(selected_cities), 2):
            cols = st.columns(2) if len(selected_cities[i:i+2]) == 2 else st.columns([1, 1])
            for j, city in enumerate(selected_cities[i:i+2]):
                with cols[j]:
                    city_data = top_neighborhoods[top_neighborhoods['city'] == city].sort_values('activity_score')
                    color = city_colors[city]

                    r, g, b = int(color[1:3],16), int(color[3:5],16), int(color[5:7],16)

                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=city_data['activity_score'],
                        y=city_data['neighbourhood_cleansed'],
                        orientation='h',
                        marker=dict(
                            color=city_data['activity_score'],
                            colorscale=[[0, f'rgba({r},{g},{b},0.3)'], [1, color]],
                            line=dict(width=0)
                        ),
                        hovertemplate='<b>%{y}</b><br>Activity Score: %{x:.3f}<extra></extra>'
                    ))

                    fig.update_layout(
                        template='plotly_dark',
                        paper_bgcolor='rgba(21,34,54,0.5)',
                        plot_bgcolor='rgba(21,34,54,0.5)',
                        font=dict(family='Inter', color='#e0e8f0'),
                        height=260,
                        margin=dict(l=0, r=20, t=40, b=20),
                        title=dict(
                            text=city_labels[city],
                            font=dict(family='Playfair Display', size=16, color='#ffffff'),
                            x=0
                        ),
                        bargap=0.3,
                    )
                    fig.update_xaxes(range=[0, 1], showgrid=True,
                                     gridcolor='rgba(255,255,255,0.05)',
                                     tickfont=dict(size=10, color='#8ba3bc'))
                    fig.update_yaxes(showgrid=False,
                                     tickfont=dict(size=11, color='#e0e8f0'))

                    st.markdown(f"""
                    <div class="metric-card animate-in" style="padding:0;overflow:hidden;
                                margin-bottom:0;animation-delay:{i*0.1+j*0.15}s">
                    """, unsafe_allow_html=True)
                    st.plotly_chart(fig, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)

    st.divider()
    st.markdown('<div class="section-label">Key Observations</div>', unsafe_allow_html=True)

    obs_cols = st.columns(3)
    observations = [
        ("🏙️", "Seattle dominates",
         "Belltown (0.89) and Broadway (0.84) top the sample. Seattle's compact urban core, high tourism, and strong short-term rental demand create intense concentration in just a few neighborhoods. Despite this, only moderate rent growth (6.4%) — suggesting supply has kept pace."),
        ("🏘️", "Bedford-Stuyvesant, NYC",
         "NYC's highest-activity neighborhood is not Midtown or SoHo — it's Bed-Stuy, a historically Black neighborhood that has undergone rapid gentrification since 2015. High Airbnb density here directly competes with long-term housing for existing residents."),
        ("🌊", "Long Beach surprises LA",
         "The top LA neighborhood by activity score isn't Hollywood or Venice — it's Long Beach. This suggests Airbnb pressure in LA has spread well beyond traditional tourist zones into working-class coastal communities."),
        ("🏢", "Near North Side, Chicago",
         "Chicago's second-ranked neighborhood has a 91% commercial host rate — almost entirely property managers, not individuals. This level of commercialization raises serious questions about how many units have been permanently converted from long-term to short-term rental."),
        ("📋", "NYC regulations not working as expected",
         "Despite Local Law 18 (2023) requiring host registration, NYC's top neighborhoods still show significant Airbnb activity. Only 15% of listings are licensed citywide — suggesting widespread non-compliance rather than market suppression."),
        ("🗺️", "Austin zip codes tell a different story",
         "Austin's top areas (78702, 78704) correspond to East Austin and South Congress — historically working-class neighborhoods now undergoing rapid change. The zip code granularity obscures neighborhood-level displacement that likely exists."),
    ]

    for i in range(0, len(observations), 3):
        cols = st.columns(3)
        for j, (icon, title, text) in enumerate(observations[i:i+3]):
            with cols[j]:
                st.markdown(f"""
                <div class="metric-card animate-in" 
                     style="height:100%;animation-delay:{i*0.05+j*0.1}s">
                    <div style="font-size:1.5rem;margin-bottom:0.5rem">{icon}</div>
                    <div style="font-weight:600;color:#fff;margin-bottom:0.5rem;
                                font-size:0.88rem">{title}</div>
                    <div class="finding-text">{text}</div>
                </div>
                """, unsafe_allow_html=True)

# ── Tab 4: Model Insights ──
with tab4:
    st.markdown('<div class="section-intro">A LightGBM model trained to predict neighborhood-level Airbnb activity scores from listing characteristics. SHAP (SHapley Additive exPlanations) values reveal which features drive predictions and in which direction — turning the model into an interpretable analytical tool.</div>', unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown("""
        <div class="metric-card animate-in">
            <div class="metric-city">R² Score</div>
            <div class="metric-rent">0.708</div>
            <div class="finding-text">Explains 71% of variance in neighborhood activity</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown("""
        <div class="metric-card animate-in" style="animation-delay:0.1s">
            <div class="metric-city">Mean Absolute Error</div>
            <div class="metric-rent">0.047</div>
            <div class="finding-text">Average prediction error on a 0–1 scale</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown("""
        <div class="metric-card animate-in" style="animation-delay:0.2s">
            <div class="metric-city">Training Samples</div>
            <div class="metric-rent">476</div>
            <div class="finding-text">80% of 596 neighborhoods across 5 cities</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    st.markdown('<div class="section-label">Feature Importance — click a bar to explore</div>',
                unsafe_allow_html=True)

    feature_name_map = {
        'avg_occupancy': 'Avg Occupancy',
        'commercial_pct': 'Commercial Host %',
        'occupancy_rate': 'Occupancy Rate',
        'median_price': 'Median Price',
        'avg_availability': 'Avg Availability',
        'avg_reviews': 'Avg Reviews',
        'licensed_pct': 'Licensed %',
        'city': 'City'
    }

    feature_descriptions = {
        'avg_occupancy': 'Average days per year a listing is booked. The strongest predictor by far — high occupancy neighborhoods have consistently higher activity scores across all cities.',
        'commercial_pct': 'Share of listings from hosts with more than one property. High commercial rates signal professional operators dominating the local market.',
        'occupancy_rate': 'Occupancy as a fraction of the year (0–1). Correlated with avg_occupancy but captures the rate dimension separately.',
        'median_price': 'Median nightly listing price. Higher prices moderately increase predicted activity — premium markets attract more investment.',
        'avg_availability': 'Average days listings are available to book. Counterintuitively, high availability DECREASES predicted activity — always-available listings tend to be inactive or poorly managed.',
        'avg_reviews': 'Average review score across listings. Modest positive effect — well-reviewed neighborhoods attract more repeat bookings.',
        'licensed_pct': 'Share of listings with a valid operating license. Mixed signal — licensing requirements exist in many cities but compliance alone does not reliably suppress activity.',
        'city': 'Which of the 5 cities the neighborhood belongs to. Ranks last — local neighborhood characteristics predict activity better than city identity.'
    }

    shap_importance['feature_label'] = shap_importance['feature'].map(
        lambda x: feature_name_map.get(x, x)
    )

    col_shap, col_detail = st.columns([2, 1])

    with col_shap:
        fig5 = px.bar(
            shap_importance.sort_values('mean_shap'),
            x='mean_shap',
            y='feature_label',
            orientation='h',
            labels={'mean_shap': 'Mean |SHAP Value|', 'feature_label': ''},
            color='mean_shap',
            color_continuous_scale='Reds',
        )
        fig5.update_layout(
            **PLOTLY_THEME,
            height=400,
            showlegend=False,
            coloraxis_showscale=False,
            clickmode='event+select'
        )
        fig5.update_traces(
            hovertemplate='<b>%{y}</b><br>SHAP: %{x:.4f}<extra></extra>'
        )
        selected_points = st.plotly_chart(
            fig5, use_container_width=True,
            on_select='rerun', key='shap_chart'
        )

    with col_detail:
        st.markdown('<div class="section-label">Feature Detail</div>',
                    unsafe_allow_html=True)

        selected_feature = None
        if (selected_points and
            selected_points.get('selection') and
            selected_points['selection'].get('points')):
            point = selected_points['selection']['points'][0]
            selected_feature = shap_importance.sort_values('mean_shap').iloc[
                point['point_index']
            ]['feature']

        if selected_feature:
            feat_label = feature_name_map.get(selected_feature, selected_feature)
            feat_shap = shap_importance[
                shap_importance['feature'] == selected_feature
            ]['mean_shap'].iloc[0]
            feat_desc = feature_descriptions.get(selected_feature, '')

            st.markdown(f"""
            <div class="finding-card animate-in">
                <div style="font-weight:600;color:#ff6b6b;font-size:1.05rem;
                            margin-bottom:0.4rem">{feat_label}</div>
                <div style="color:#666;font-size:0.78rem;margin-bottom:0.7rem">
                    Mean |SHAP|: {feat_shap:.4f}
                </div>
                <div class="finding-text">{feat_desc}</div>
            </div>
            """, unsafe_allow_html=True)

            if selected_feature in neighborhood.columns:
                fig_dist = px.histogram(
                    neighborhood, x=selected_feature,
                    color='city_label', nbins=30,
                    labels={selected_feature: feat_label, 'count': ''},
                    color_discrete_sequence=[
                        '#ff6b6b', '#ffd43b', '#69db7c', '#74c0fc', '#da77f2'
                    ],
                    barmode='overlay', opacity=0.7
                )
                fig_dist.update_layout(
                    **PLOTLY_THEME, height=260,
                    showlegend=True,
                    legend=dict(orientation='h', y=1.15, font=dict(size=9)),
                    title=dict(text=f'Distribution of {feat_label}',
                               font=dict(size=11, color='#8ba3bc')),
                    margin=dict(t=40, b=20)
                )
                st.plotly_chart(fig_dist, use_container_width=True)
        else:
            st.markdown("""
            <div class="finding-card">
                <div class="finding-text" style="color:#555;font-style:italic">
                    👆 Click any bar to explore that feature's distribution 
                    across all neighborhoods and cities.
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    col_pred, col_findings = st.columns([1, 1])

    with col_pred:
        st.markdown('<div class="section-label">Predicted vs Actual</div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="section-intro" style="font-size:0.82rem">Points close to the diagonal indicate accurate predictions. The model performs well for low-to-mid activity scores but underpredicts high-activity neighborhoods — a common limitation with small datasets.</div>', unsafe_allow_html=True)
        fig6 = px.scatter(
            predictions, x='actual', y='predicted',
            labels={'actual': 'Actual Activity Score',
                    'predicted': 'Predicted Activity Score'},
            opacity=0.6,
            color_discrete_sequence=['#ff6b6b']
        )
        fig6.add_shape(
            type='line', x0=0, y0=0, x1=1, y1=1,
            line=dict(dash='dash', color='rgba(255,255,255,0.2)')
        )
        fig6.update_layout(**PLOTLY_THEME, height=380)
        st.plotly_chart(fig6, use_container_width=True)

    with col_findings:
        st.markdown('<div class="section-label">Key Findings</div>',
                    unsafe_allow_html=True)
        findings = [
            ("🏠 Occupancy drives everything",
             "Avg occupancy is the dominant predictor — far above price, licensing, or city identity."),
            ("🏢 Commercial hosts amplify pressure",
             "Neighborhoods dominated by property managers show consistently higher activity scores."),
            ("📋 Licensing doesn't suppress activity",
             "Licensed % has mixed SHAP signal — regulations exist but compliance alone isn't enough."),
            ("🏙️ City matters less than neighborhood",
             "City ranks last — local characteristics predict activity better than which city you're in."),
            ("📉 Austin defies the narrative",
             "Moderate Airbnb activity but falling rents (-10%) — driven by a housing supply boom, not STR suppression."),
        ]
        for i, (title, text) in enumerate(findings):
            st.markdown(f"""
            <div class="finding-card animate-in" style="animation-delay:{i*0.08}s">
                <div style="font-weight:600;color:#fff;margin-bottom:0.35rem;font-size:0.88rem">
                    {title}
                </div>
                <div class="finding-text">{text}</div>
            </div>
            """, unsafe_allow_html=True)

# ── Tab 5: Summary ──
with tab5:
    st.markdown('<div class="section-intro">A consolidated overview of the research questions, methodology, key findings, and directions for future work.</div>', unsafe_allow_html=True)

    # Research questions
    st.markdown('<div class="section-label">Research Questions</div>', unsafe_allow_html=True)
    q_cols = st.columns(3)
    questions = [
        ("❓", "Geographic clustering",
         "Which neighborhoods across LA, NYC, Chicago, Seattle, and Austin have the highest Airbnb activity — and do high-activity neighborhoods cluster geographically?"),
        ("❓", "Housing affordability link",
         "Does neighborhood-level Airbnb activity correlate with rising long-term rents? Is the relationship consistent across cities with different regulatory environments?"),
        ("❓", "What drives activity?",
         "Which listing and host characteristics best predict neighborhood-level Airbnb activity? Can a model identify the levers cities could pull to reduce pressure?"),
    ]
    for i, (icon, title, text) in enumerate(questions):
        with q_cols[i]:
            st.markdown(f"""
            <div class="metric-card animate-in" style="animation-delay:{i*0.1}s;height:100%">
                <div style="font-size:1.5rem;margin-bottom:0.5rem">{icon}</div>
                <div style="font-weight:600;color:#ff6b6b;margin-bottom:0.5rem;
                            font-size:0.88rem">{title}</div>
                <div class="finding-text">{text}</div>
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    # Methodology
    st.markdown('<div class="section-label">Methodology</div>', unsafe_allow_html=True)
    m_cols = st.columns(4)
    methods = [
        ("📥", "Data Collection",
         "InsideAirbnb listings, calendar, and GeoJSON files for 5 US cities (Sep–Nov 2025). Zillow ZORI city-level rent index (2015–2026). Automated pipeline with dynamic date detection."),
        ("🔧", "Feature Engineering",
         "Neighborhood-level aggregation of 112,892 listings. Composite activity score combining listing density (40%), occupancy (40%), and commercial host rate (20%), normalized within each city."),
        ("📊", "Analysis",
         "City-level rent growth vs Airbnb occupancy comparison. Neighborhood activity mapping using GeoJSON choropleth. Zillow ZORI rent trajectory analysis with COVID and post-pandemic annotations."),
        ("🤖", "Modeling",
         "LightGBM regressor with early stopping to predict neighborhood activity scores. SHAP values for feature interpretability. Train/test split 80/20 across 596 neighborhoods."),
    ]
    for i, (icon, title, text) in enumerate(methods):
        with m_cols[i]:
            st.markdown(f"""
            <div class="finding-card animate-in" style="animation-delay:{i*0.1}s;height:100%">
                <div style="font-size:1.3rem;margin-bottom:0.4rem">{icon}</div>
                <div style="font-weight:600;color:#fff;margin-bottom:0.4rem;
                            font-size:0.82rem">{title}</div>
                <div class="finding-text" style="font-size:0.8rem">{text}</div>
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    # Key findings
    st.markdown('<div class="section-label">Key Findings</div>', unsafe_allow_html=True)
    f_cols = st.columns(2)
    findings = [
        ("🔍", "No simple Airbnb → rent relationship",
         "At the city level, higher Airbnb occupancy does not reliably predict higher rent growth. Chicago has the highest rent growth (17.5%) despite a smaller Airbnb market than LA or NYC. Austin saw rents fall 10.1% despite moderate Airbnb activity — driven by a housing supply boom that added tens of thousands of units in 2021–2023."),
        ("📍", "Activity clusters in specific neighborhoods",
         "Airbnb pressure is highly concentrated. Seattle's Belltown and Broadway, NYC's Bedford-Stuyvesant and Midtown, and LA's Long Beach and Hollywood account for a disproportionate share of activity. High-activity neighborhoods are not always tourist zones — gentrifying residential areas are increasingly affected."),
        ("🏢", "Commercial hosts dominate everywhere",
         "Across all 5 cities, more than 50% of listings come from hosts with multiple properties. Chicago leads at 69%. This finding challenges the 'sharing economy' narrative — short-term rental markets are increasingly run by professional operators, not individuals sharing spare rooms."),
        ("📋", "Licensing compliance is low",
         "Despite regulations in NYC, LA, Chicago, and Seattle, licensing compliance varies dramatically — from 83% in Seattle to just 15% in NYC and 0% in Austin. Regulations without enforcement appear to have limited effect on market activity."),
        ("🤖", "Neighborhood characteristics > city identity",
         "The LightGBM model (R²=0.708) found that occupancy rate and commercial host percentage are the strongest predictors of neighborhood activity — while which city a neighborhood is in ranks last. Local dynamics matter more than city-level policy in determining where Airbnb concentrates."),
        ("⚠️", "Data limitations",
         "InsideAirbnb removed price data from all US cities in late 2025, limiting price-based analysis. Austin uses postal codes rather than neighborhood names, reducing geographic interpretability. City-level Zillow data cannot capture neighborhood-level rent dynamics where the most interesting variation likely exists."),
    ]
    for i in range(0, len(findings), 2):
        cols = st.columns(2)
        for j, (icon, title, text) in enumerate(findings[i:i+2]):
            with cols[j]:
                st.markdown(f"""
                <div class="metric-card animate-in" 
                     style="animation-delay:{i*0.05+j*0.1}s;margin-bottom:0.8rem">
                    <div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:0.5rem">
                        <span style="font-size:1.2rem">{icon}</span>
                        <span style="font-weight:600;color:#fff;font-size:0.88rem">{title}</span>
                    </div>
                    <div class="finding-text">{text}</div>
                </div>
                """, unsafe_allow_html=True)

    st.divider()

    # Future perspectives
    st.markdown('<div class="section-label">Future Directions</div>', unsafe_allow_html=True)
    fp_cols = st.columns(3)
    future = [
        ("🗺️", "Neighborhood-level rent data",
         "ZIP-code or neighborhood-level rent data (e.g. from HUD, Apartment List, or CoStar) would allow direct correlation between Airbnb activity scores and local rent changes — the core analysis this project points toward but cannot fully execute with city-level ZORI."),
        ("🏗️", "Housing supply controls",
         "Adding building permit data and new construction rates as control variables would help isolate the Airbnb effect from supply-side dynamics — the Austin case demonstrates why this matters. A city adding housing fast can absorb STR pressure that would otherwise drive rents up."),
        ("⏳", "Longitudinal analysis",
         "Tracking the same neighborhoods across multiple InsideAirbnb snapshots over 3–5 years would reveal whether activity is rising or falling, and whether regulatory interventions (like NYC's 2023 law) have measurable effects on neighborhood-level pressure over time."),
        ("🌍", "International expansion",
         "Extending the analysis to cities like Barcelona, Amsterdam, and London — where STR regulations are stricter and longer-established — would provide stronger natural experiments for testing whether regulation actually reduces housing pressure."),
        ("🧠", "Richer modeling",
         "Adding proximity to tourist attractions, transit access, and zoning data as features could improve the model's R² beyond 0.708 and provide more actionable policy insights. The current model's inability to predict high-activity neighborhoods well likely reflects missing geographic features."),
        ("📱", "Real-time monitoring",
         "Automating the data pipeline to refresh monthly with new InsideAirbnb snapshots and Zillow data would turn this into a live monitoring tool — useful for city planners, journalists, and housing advocates tracking the evolving impact of short-term rentals."),
    ]
    for i in range(0, len(future), 3):
        cols = st.columns(3)
        for j, (icon, title, text) in enumerate(future[i:i+3]):
            with cols[j]:
                st.markdown(f"""
                <div class="finding-card animate-in"
                     style="animation-delay:{i*0.05+j*0.1}s;margin-bottom:0.8rem">
                    <div style="font-size:1.3rem;margin-bottom:0.4rem">{icon}</div>
                    <div style="font-weight:600;color:#fff;margin-bottom:0.4rem;
                                font-size:0.82rem">{title}</div>
                    <div class="finding-text" style="font-size:0.8rem">{text}</div>
                </div>
                """, unsafe_allow_html=True)

    st.divider()

    # Data sources
    st.markdown('<div class="section-label">Data Sources</div>', unsafe_allow_html=True)

    src1, src2, src3, src4 = st.columns(4)

    sources = [
        (src1, "Airbnb Data", "InsideAirbnb", "insideairbnb.com · Sep–Nov 2025", "Listings, calendar, GeoJSON for 5 US cities"),
        (src2, "Rent Data", "Zillow Research", "zillow.com/research · 2015–2026", "ZORI City-Level Rent Index"),
        (src3, "Tools & Libraries", "Python Stack", "Pandas · GeoPandas · LightGBM", "SHAP · Plotly · Streamlit · BeautifulSoup"),
        (src4, "Coverage", "5 US Cities", "112,892 listings · 596 neighborhoods", "11 years of rent history"),
    ]

    for col, label, title, sub1, sub2 in sources:
        with col:
            st.markdown(
                f"<div class='finding-card'>"
                f"<div style='font-size:0.7rem;text-transform:uppercase;letter-spacing:0.12em;"
                f"color:#ff6b6b;margin-bottom:0.4rem;font-weight:600'>{label}</div>"
                f"<div style='color:#e0e8f0;font-size:0.88rem;font-weight:600;"
                f"margin-bottom:0.3rem'>{title}</div>"
                f"<div style='color:#8ba3bc;font-size:0.78rem;margin-bottom:0.2rem'>{sub1}</div>"
                f"<div style='color:#8ba3bc;font-size:0.78rem'>{sub2}</div>"
                f"</div>",
                unsafe_allow_html=True
            )

with tab6:
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("""
        <div class="animate-in">
            <div class="hero-eyebrow">About the Author</div>
            <div style="font-family:'Playfair Display',serif;font-size:2.2rem;
                        color:#ffffff;line-height:1.1;margin-bottom:0.5rem">
                Christine Li
            </div>
            <div style="color:#ff6b6b;font-size:0.85rem;margin-bottom:1.2rem;font-weight:500">
                Statistics @ UC Davis · Master of Analytics @ UC Berkeley
            </div>
            <div style="font-size:0.92rem;color:#8ba3bc;line-height:1.8;
                        max-width:560px;margin-bottom:1.5rem">
                I build data projects at the intersection of rigorous analysis and 
                product thinking. This dashboard is one of two solo portfolio projects 
                I built from scratch — end-to-end data pipeline, modeling, and a 
                product-quality interface — to demonstrate what I can do independently.
                <br><br>
                Open to <strong style="color:#e0e8f0">Data Analytics</strong>, 
                <strong style="color:#e0e8f0">AI Product</strong>, and 
                <strong style="color:#e0e8f0">Machine Learning</strong> roles — 
                internships, new grad, and full-time opportunities.
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Links
        st.markdown("""
        <div style="display:flex;gap:0.8rem;flex-wrap:wrap;margin-bottom:2rem">
            <a href="https://github.com/christinel012" target="_blank"
               style="background:linear-gradient(135deg,#152236,#1a2d45);
                      border:1px solid rgba(255,255,255,0.1);border-radius:10px;
                      padding:0.55rem 1.1rem;color:#e0e8f0;text-decoration:none;
                      font-size:0.82rem;font-weight:500">⌥ GitHub</a>
            <a href="https://www.linkedin.com/in/christineli012" target="_blank"
               style="background:linear-gradient(135deg,#152236,#1a2d45);
                      border:1px solid rgba(255,255,255,0.1);border-radius:10px;
                      padding:0.55rem 1.1rem;color:#e0e8f0;text-decoration:none;
                      font-size:0.82rem;font-weight:500">in LinkedIn</a>
            <a href="mailto:ytingli0210@gmail.com"
               style="background:linear-gradient(135deg,#152236,#1a2d45);
                      border:1px solid rgba(255,255,255,0.1);border-radius:10px;
                      padding:0.55rem 1.1rem;color:#e0e8f0;text-decoration:none;
                      font-size:0.82rem;font-weight:500">✉ Email</a>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-label">This Project</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="finding-card animate-in">
            <div class="finding-text">
                Built solo over ~2 weeks. Covers automated data ingestion from InsideAirbnb 
                and Zillow, neighborhood-level feature engineering across 596 neighborhoods, 
                LightGBM modeling with SHAP interpretability (R²=0.708), and this Streamlit 
                dashboard. Full source code on GitHub.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="section-label">Skills</div>', unsafe_allow_html=True)

        skill_groups = [
            ("Languages", ["Python", "R", "SQL", "MATLAB"]),
            ("ML & Modeling", ["LightGBM", "XGBoost", "SHAP", "LSTM", "Scikit-learn"]),
            ("Data & Viz", ["Pandas", "GeoPandas", "Plotly", "Streamlit"]),
            ("Web", ["React", "JavaScript", "Leaflet.js"]),
            ("Tools", ["Git", "GitHub", "Jupyter"]),
        ]

        for i, (group, skills) in enumerate(skill_groups):
            pills = ''.join(
                f"<span style='background:rgba(255,107,107,0.12);"
                f"border:1px solid rgba(255,107,107,0.2);border-radius:20px;"
                f"padding:0.2rem 0.6rem;font-size:0.73rem;color:#e0e8f0;"
                f"display:inline-block;margin:0.15rem'>{s}</span>"
                for s in skills
            )
            st.markdown(
                f"<div class='metric-card animate-in' style='animation-delay:{i*0.08}s;"
                f"margin-bottom:0.5rem;padding:0.8rem 1rem'>"
                f"<div style='font-size:0.65rem;text-transform:uppercase;letter-spacing:0.1em;"
                f"color:#8ba3bc;margin-bottom:0.4rem'>{group}</div>"
                f"<div>{pills}</div></div>",
                unsafe_allow_html=True
            )

        st.markdown(
            "<div style='margin-top:1rem;padding:0.8rem 1rem;"
            "background:linear-gradient(135deg,#152236,#1a2d45);"
            "border:1px solid rgba(255,255,255,0.06);border-radius:12px;"
            "font-size:0.8rem;color:#8ba3bc;line-height:1.7'>"
            "🎓 GPA 3.95 · Dean's Honor List<br>"
            "📜 NVIDIA Deep Learning Certification<br>"
            "🌏 Mandarin (Native) · English (Fluent)<br>"
            "✈️ Traveling · 📷 Photography · 🍞 Baking"
            "</div>",
            unsafe_allow_html=True
        )