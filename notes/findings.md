## City-Level Summary — findings

Data: InsideAirbnb second-newest snapshot (Sep-Nov 2025 depending on city)
Cities: Los Angeles, New York City, Chicago, Seattle, Austin

**Occupancy:**
- Seattle highest at 108 days/year — strong STR demand, relatively small market
- NYC lowest at 47 days — likely effect of strict 2023 STR regulations
- Chicago 87 days despite having regulations — enforcement may be weak

**Licensing:**
- Seattle 83% licensed — strongest enforcement in sample
- Austin 0% licensed — no licensing requirement, most permissive market
- NYC 15% — regulations exist (Local Law 18, 2023) but compliance is low
- LA only 28% — large market with weak enforcement

**Commercialization:**
- Chicago 69% commercial hosts — most dominated by property managers
- NYC 51% — lowest despite being most regulated, suggests regulations 
  are keeping individual hosts out more than commercial ones
- All cities above 50% commercial — short term rental market is not 
  a "sharing economy" anywhere in this sample

**Price:**
- Seattle highest median $177, Austin lowest $138
- Prices fairly similar across cities — surprising, expected more variation
- Price less reliable metric due to missing data in some cities

**Questions to follow up:**
- Why is Chicago commercialization so high despite regulations?
- Does Seattle's high occupancy reflect tourism or business travel?
- Austin 0% licensing — is this by policy or just no data?


## Neighborhood Analysis — findings

- Bedsty tops activity score — good story hook, connects to gentrification literature
- Austin zip codes = problem for mapping and labeling, investigate if neighborhood 
  names exist elsewhere
- Little Neck outlier: 4 listings, 214 avg occupancy — filter neighborhoods 
  below min listing threshold (maybe n < 20?)
- Need within-city normalization for activity score — current version biased toward 
  large cities
- Follow up: does high activity score correlate with Zillow rent increases? 
  That's the key question


## Within-City Activity Score

- Seattle Belltown/Broadway highest scores — very high occupancy + commercial
- Chicago Near North Side 91% commercial — almost entirely property managers
- NYC scores lower overall — regulation suppressing activity even in hot neighborhoods
- Austin zip codes still a problem — need to find neighborhood name mapping
- Long Beach surprisingly high for LA — not a typical tourist area, worth investigating
- Fort Hamilton NYC: small neighborhood, very high occupancy (147 days) — 
  could be interesting outlier story


## City Rent + Airbnb Merge — findings

Data: Zillow ZORI city-level, comparing 2023-03-31 → 2026-03-31
Merged with InsideAirbnb city-level summary stats.

**Rent growth ranking:**
1. Chicago +17.5% — high occupancy (87 days), high commercial rate (69%)
2. NYC +14.0% — low occupancy (47 days) due to strict regulations
3. Seattle +6.4% — highest occupancy (108 days) but moderate rent growth
4. LA +4.1% — largest market, moderate everything
5. Austin -10.1% — rents actually fell despite moderate Airbnb activity

**Key insight:** No clean positive correlation between Airbnb activity and rent growth.
- Austin destroys the simple narrative — massive housing supply boom in 2021-2023
  absorbed demand and drove rents down despite active STR market
- Chicago is the opposite — smaller market, high commercial operators, high rent growth
- NYC regulations suppressed occupancy but didn't stop rent growth

**Implications for project framing:**
- Don't oversell "Airbnb causes rent increases" — data doesn't support it at city level
- Better framing: "Airbnb is one of several pressures on housing affordability,
  and its impact varies significantly by local policy and housing supply"
- Neighborhood level might tell a different story — need to dig deeper
- Consider adding housing supply data (permits, new construction) as control variable

**Questions to follow up:**
- Does neighborhood-level analysis show stronger correlation?
- Can we control for housing supply to isolate Airbnb effect?
- Why is Chicago rent growth so high relative to its Airbnb market size?
- Austin: when did rents peak and start falling? Does it align with supply boom?


## Visualizations — findings

### Chart 1: Rent Trends Over Time (2015–2026)
- NYC had sharpest COVID dip in 2020 — people fled the city, rents collapsed
- Austin peaked ~2022-2023 then fell sharply — housing supply boom absorbed demand
- Chicago and Seattle steady climb throughout, no major disruption
- LA relatively flat post-COVID — large, stable market
- All cities show rent surge starting 2021, aligning with post-COVID return + inflation

### Chart 2: Airbnb Occupancy vs Rent Growth Scatter
- No clean positive correlation between occupancy and rent growth at city level
- Chicago: moderate occupancy (87 days) but highest rent growth (17.5%)
- Seattle: highest occupancy (108 days) but only 6.4% rent growth
- Austin: moderate occupancy but negative rent growth — outlier driven by supply
- Suggests Airbnb activity is one factor among many, not the primary driver

### Chart 3: Neighborhood Activity Score Distribution
- Seattle most uniformly active — tight distribution, high median
- NYC most suppressed — lowest median, tight range, likely regulation effect
- Austin widest spread — huge variation between zip codes
- Chicago has highest outlier neighborhoods despite mid-range city average

### Chart 4: Top 5 Neighborhoods by City
- Seattle: Belltown and Broadway dominate — dense urban core
- Chicago: West Town and Near North Side — high commercial host rates
- LA: Long Beach surprisingly tops the list — not a typical tourist area
- NYC: Bedford-Stuyvesant and Midtown — gentrifying neighborhood vs tourist hub
- Austin: zip codes only, no neighborhood name resolution possible

### Chart 5: Neighborhood Maps
- All 5 cities show clear geographic clustering of high activity
- LA: coastal neighborhoods and Long Beach hottest
- NYC: lower Manhattan and Brooklyn hottest
- Chicago: north side neighborhoods dominate
- Seattle: downtown core and Capitol Hill area
- Austin: central zip codes (78702, 78704) most active

### Overall visualization takeaway
- City-level story: Airbnb impact not uniform, policy and supply matter more
- Neighborhood-level story: activity clusters geographically in predictable ways
- Most interesting finding: Austin defies the narrative — high STR activity, falling rents
- Best visual: neighborhood maps — most intuitive