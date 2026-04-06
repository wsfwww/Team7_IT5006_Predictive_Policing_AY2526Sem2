import streamlit as st
import pandas as pd
import pydeck as pdk
import plotly.express as px
import matplotlib.pyplot as plt


# ================= 1. Base Config =================
st.set_page_config(
    page_title="Chicago Crime Analytics",
    page_icon="🚓",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .block-container {padding-top: 1rem; padding-bottom: 2rem;}
    h1 {font-size: 2.5rem;}
    h2 {font-size: 1.8rem; margin-top: 2rem;}
</style>
""", unsafe_allow_html=True)

st.title("🚓 Chicago Crime Predictive Policing Dashboard")
st.markdown("**Phase 1: Exploratory Data Analysis (EDA) & Visualization**")

# ================= 2. Data Lazy Loading =================
@st.cache_data
def load_data_by_range(start_year, end_year):
    # Optimization: Load only columns used in the dashboard to save memory
    # Excludes heavy unused columns like 'Date', 'ID', 'Case Number', 'Updated On', etc.
    required_cols = [
        'Year', 'Month', 'DayOfWeek', 'Hour', 
        'Primary Type', 'Location Description', 'Arrest', 'Domestic',
        'District', 'Community Area', 'Block', 
        'Longitude', 'Latitude'
    ]
    
    all_dfs = []
    years_to_load = range(start_year, end_year + 1)
    
    for year in years_to_load:
        file_path = f"./data_chunks/crimes_{year}.parquet"
        try:
            # 1. Load specific columns only
            try:
                df_year = pd.read_parquet(file_path, columns=required_cols)
            except ValueError:
                # Fallback: if columns mismatch in older files, load all then filter
                df_year = pd.read_parquet(file_path)
                df_year = df_year[df_year.columns.intersection(required_cols)]

            # 2. Optimize Data Types (Crucial for Memory)
            # Convert Strings to Category (Huge memory savings)
            cat_cols = ['Primary Type', 'Location Description', 'Block', 'DayOfWeek', 'District', 'Community Area']
            for col in cat_cols:
                if col in df_year.columns:
                    df_year[col] = df_year[col].astype('category')
            
            # Downcast Numerics (Float64 -> Float32, Int64 -> Int16/8)
            for col in ['Longitude', 'Latitude']:
                if col in df_year.columns:
                    df_year[col] = df_year[col].astype('float32')
            
            for col in ['Year', 'Month', 'Hour']:
                if col in df_year.columns:
                    df_year[col] = pd.to_numeric(df_year[col], downcast='unsigned')

            all_dfs.append(df_year)
            print(f"Loaded data for year: {year} ({len(df_year)} records)")
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            continue
            
    if not all_dfs:
        return None
        
    full_df = pd.concat(all_dfs, ignore_index=True)
    
    return full_df

# ================= 3. Sidebar Controls =================
st.sidebar.header("🎛️ Filter Controls")

# 3.1 Year Range Selection (Default: Select All)
min_year, max_year = 2015, 2025
selected_years = st.sidebar.slider(
    "Select Year Range",
    min_value=min_year,
    max_value=max_year,
    value=(2023, 2025) # Default to last 3 years to prevent OOM on startup
)

# Load data (with Loading spinner)
with st.spinner(f"Loading data for {selected_years[0]} - {selected_years[1]}..."):
    data = load_data_by_range(selected_years[0], selected_years[1])

if data is None or data.empty:
    st.error("No data found for the selected range!")
    st.stop()

# 3.2 Crime Type Filter (Logic Optimization: Default = Select All)
# Get all types from currently loaded data
available_types = sorted(data['Primary Type'].unique())

# Here default=[] means no specific type selected, but logically treated as "All"
filter_types = st.sidebar.multiselect(
    "Filter by Crime Type", 
    options=available_types,
    default=[] # Default is empty, logic below handles empty as select all
)

# 3.3 Additional Filters (District, Community Area, Location)
available_districts = sorted(data['District'].dropna().unique())
filter_districts = st.sidebar.multiselect(
    "District",
    options=available_districts,
    default=[]
)

available_communities = sorted(data['Community Area'].dropna().unique())
filter_communities = st.sidebar.multiselect(
    "Community Area",
    options=available_communities,
    default=[]
)

available_locations = sorted(data['Location Description'].dropna().unique())
filter_locations = st.sidebar.multiselect(
    "Location Description",
    options=available_locations,
    default=[]
)

# 3.3 Data Filtering Logic
# Start with full data
filtered_data = data
if filter_types:
    filtered_data = filtered_data[filtered_data['Primary Type'].isin(filter_types)]
if filter_districts:
    filtered_data = filtered_data[filtered_data['District'].isin(filter_districts)]
if filter_communities:
    filtered_data = filtered_data[filtered_data['Community Area'].isin(filter_communities)]
if filter_locations:
    filtered_data = filtered_data[filtered_data['Location Description'].isin(filter_locations)]

# Show current data overview
st.sidebar.markdown("---")
# Use Metric to show data count, adding thousands separator
st.sidebar.metric("Total Records Loaded", f"{len(data):,}")
st.sidebar.metric("Records Displayed", f"{len(filtered_data):,}")

# --- ⚠️ Critical Performance Protection ---
# If data volume is too large (e.g., > 50k rows), sampling is recommended before map plotting to prevent browser freeze
# However, statistical charts (Bar Chart/Line Chart) can use full data
map_data = filtered_data
if len(map_data) > 50000:
    st.sidebar.warning(f"⚠️ Map data sampled (50k/{len(map_data):,}) for performance.")
    map_data = filtered_data.sample(50000)

# ================= 4. Core Functional Modules =================

# --- Tab 1: Spatio-Temporal Analysis (Integrating Person B) ---
# --- Define Tabs: Completely separate spatio-temporal analysis into two independent Tabs ---
tab1, tab2, tab3 = st.tabs(["⏰ Temporal Patterns", "🗺️ Spatial Patterns", "🔗 Correlation & Insights"])

# ================= Tab 1: Temporal =================
with tab1:
    st.subheader("📈 Temporal Trends Analysis")
    
    if not filtered_data.empty:
        t_col1, t_col2 = st.columns(2)
        
        # 1. Yearly
        with t_col1:
            yearly_counts = filtered_data['Year'].value_counts().sort_index()
            fig_year = px.line(
                x=yearly_counts.index, 
                y=yearly_counts.values,
                markers=True,
                labels={'x': 'Year', 'y': 'Count'},
                title="1. Crime Trend by Year",
                color_discrete_sequence=['purple']
            )
            fig_year.update_xaxes(type='category') # Force display of integer years
            st.plotly_chart(fig_year, width="stretch")

        # 2. Monthly
        with t_col2:
            monthly_counts = filtered_data['Month'].value_counts().sort_index()
            fig_month = px.bar(
                x=monthly_counts.index, 
                y=monthly_counts.values,
                labels={'x': 'Month', 'y': 'Count'},
                title="2. Seasonal Trend (by Month)",
                color_discrete_sequence=['lightgreen']
            )
            fig_month.update_xaxes(dtick=1)
            st.plotly_chart(fig_month, width="stretch")

        # 3. Weekly
        with t_col1:
            daily_counts = filtered_data['DayOfWeek'].value_counts()
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            # Reindex only if data contains these days to prevent errors
            daily_counts = daily_counts.reindex([d for d in day_order if d in daily_counts.index])
            
            fig_day = px.bar(
                x=daily_counts.index, 
                y=daily_counts.values,
                labels={'x': 'Day of Week', 'y': 'Count'},
                title="3. Weekly Patterns",
                color_discrete_sequence=['lightcoral']
            )
            st.plotly_chart(fig_day, width="stretch")

        # 4. Hourly Distribution
        with t_col2:
            hourly_counts = filtered_data['Hour'].value_counts().sort_index()
            fig_hour = px.bar(
                x=hourly_counts.index, 
                y=hourly_counts.values,
                labels={'x': 'Hour of Day', 'y': 'Count'},
                title="4. Hourly Patterns",
                color_discrete_sequence=['#ff4b4b']
            )
            st.plotly_chart(fig_hour, width="stretch")

    else:
        st.warning("No data available to display temporal trends.")

# ================= Tab 2: Pure Spatial Analysis =================
with tab2:
    st.subheader("📍 Spatial Distribution Analysis")
    
    if not filtered_data.empty:
        
        st.markdown("#### 1. Crime Density Heatmap (Hexbin)")
        
        # Filter out invalid coordinates for mapping
        map_df = filtered_data.dropna(subset=['Longitude', 'Latitude'])
        
        if not map_df.empty:
            # Fix 1: Adjust canvas aspect ratio to vertical (10, 12) to fit Chicago geography
            fig_map, ax = plt.subplots(figsize=(10, 12))
            
            # Automatically calculate boundaries
            lon_min, lon_max = map_df['Longitude'].quantile([0.01, 0.99])
            lat_min, lat_max = map_df['Latitude'].quantile([0.01, 0.99])
            
            # Plot Hexbin
            hb = ax.hexbin(
                map_df['Longitude'], 
                map_df['Latitude'], 
                gridsize=60, 
                cmap='inferno', 
                bins='log', 
                mincnt=1,
                extent=[lon_min, lon_max, lat_min, lat_max]
            )
            
            # Fix 2: Force equal aspect ratio for longitude/latitude to prevent map distortion
            ax.set_aspect('equal')
            
            pad_x = (lon_max - lon_min) * 0.05  # 5% padding on left/right
            pad_y = (lat_max - lat_min) * 0.05  # 5% padding on top/bottom
            
            # Set view range with padding
            ax.set_xlim(lon_min - pad_x, lon_max + pad_x)
            ax.set_ylim(lat_min - pad_y, lat_max + pad_y)
            ax.axis('off') # Remove border
            
            # Fix 3: Unify title size (Matplotlib)
            ax.set_title("Crime Density Distribution", fontsize=16, pad=20)
            
            # Adjust Colorbar
            if hb.get_array().size > 0:
                cb = fig_map.colorbar(hb, ax=ax, label='log10(Crime Count)', fraction=0.03, pad=0.04)
            
            st.pyplot(fig_map)
            st.caption("Note: Log-scale visualization used to highlight density differences.")
        else:
            st.info("No valid coordinates available for heatmap.")
        
        st.divider() # Divider

        # --- Part 2: Districts & Communities (Interactive Plotly) ---
        s_col1, s_col2 = st.columns(2)
        
        # Unified Plotly layout config (Fix inconsistent title sizes)
        plotly_layout_config = dict(
            title_font_size=16,   # Unified title font size
            title_x=0,            # Title left aligned
            margin=dict(t=50)     # Top margin
        )
        
        # 2. By District (Police Jurisdiction)
        with s_col1:
            if 'District' in filtered_data.columns:
                district_counts = filtered_data['District'].value_counts().sort_index()
                # Clean District names (remove decimals)
                district_index = district_counts.index.astype(str).str.replace(r'\.0$', '', regex=True)
                
                fig_dist = px.bar(
                    x=district_index,
                    y=district_counts.values,
                    labels={'x': 'Police District', 'y': 'Count'},
                    title="2. Crimes by Police District",
                    color_discrete_sequence=['teal']
                )
                fig_dist.update_layout(**plotly_layout_config)
                st.plotly_chart(fig_dist, width="stretch")

        # 3. By Community Area
        with s_col2:
            if 'Community Area' in filtered_data.columns:
                comm_counts = filtered_data['Community Area'].value_counts().sort_index()
                comm_index = comm_counts.index.astype(str).str.replace(r'\.0$', '', regex=True)

                fig_comm = px.bar(
                    x=comm_index,
                    y=comm_counts.values,
                    labels={'x': 'Community Area Code', 'y': 'Count'},
                    title="3. Crimes by Community Area",
                    color_discrete_sequence=['steelblue']
                )
                fig_comm.update_layout(**plotly_layout_config)
                st.plotly_chart(fig_comm, width="stretch")

        # --- Part 3: High-Crime Blocks (Top Blocks) ---
        # 4. Top N Blocks
        if 'Block' in filtered_data.columns:
            st.markdown("#### 4. Top 20 High-Crime Blocks") # Use Markdown for unified main title
            
            top_n = 20
            block_counts = filtered_data['Block'].value_counts().head(top_n)
            
            fig_block = px.bar(
                x=block_counts.index,
                y=block_counts.values,
                labels={'x': 'Block Address', 'y': 'Count'},
                # title="Top 20 Blocks", # Internal title can be removed, using Markdown above directly
                color_discrete_sequence=['darkorange']
            )
            fig_block.update_layout(
                xaxis_tickangle=-45, 
                **plotly_layout_config # Apply unified config
            )
            st.plotly_chart(fig_block, width="stretch")

    else:
        st.info("No data available for spatial analysis.")

# --- Tab 3: Correlation Analysis  ---
with tab3:
    st.subheader("🔍 Deep Insights: Correlation & Behavioral Patterns")

    if not filtered_data.empty:
        # ================= 1. Location & Crime Correlation (Core Heatmap) =================
        st.markdown("### 🏢 Location vs. Crime Type Correlation")
        
        top_n_corr = 15
        top_locs = filtered_data['Location Description'].value_counts().head(top_n_corr).index
        top_crimes_c = filtered_data['Primary Type'].value_counts().head(top_n_corr).index
        
        matrix_df = filtered_data[
            filtered_data['Location Description'].isin(top_locs) & 
            filtered_data['Primary Type'].isin(top_crimes_c)
        ]

        if not matrix_df.empty:
            st.markdown("#### 1. Crime-Centric: Where does it happen?")
            
            crosstab_crime = pd.crosstab(
                matrix_df['Primary Type'], 
                matrix_df['Location Description'], 
                normalize='index'
            )
            fig_h1 = px.imshow(
                crosstab_crime, 
                text_auto='.1%', 
                aspect="auto", 
                color_continuous_scale="Blues",
                labels=dict(x="Location Description", y="Crime Type", color="Probability")
            )
            # Increase height to make each cell clearer
            fig_h1.update_layout(height=500)
            st.plotly_chart(fig_h1, width="stretch")

            st.markdown("<br>", unsafe_allow_html=True) # Add some spacing
            st.divider() # Add a divider in between

            # --- Perspective 2: Location Perspective (Column Normalization) ---
            st.markdown("#### 2. Location-Centric: Risk Profile")
            
            crosstab_loc = pd.crosstab(
                matrix_df['Primary Type'], 
                matrix_df['Location Description'], 
                normalize='columns'
            )
            fig_h2 = px.imshow(
                crosstab_loc, 
                text_auto='.1%', 
                aspect="auto", 
                color_continuous_scale="Reds",
                labels=dict(x="Location Description", y="Crime Type", color="Probability")
            )
            fig_h2.update_layout(height=500)
            st.plotly_chart(fig_h2, width="stretch")

        else:
            st.info("No data available for correlation analysis.")

        st.divider()

        # ================= 2. Arrest & Domestic Analysis (Police Efficiency & Domestic Factors) =================
        st.markdown("### ⚖️ Police Efficiency & Incident Nature")
        col_a1, col_a2 = st.columns(2)

        with col_a1:
            # --- Arrest Rate Analysis ---
            st.markdown("#### Arrest Rate by Top 15 Crimes")
            arrest_rates = filtered_data.groupby('Primary Type', observed=True)['Arrest'].mean()
            
            # Fix: value_counts on categorical data might return 0-count categories, causing KeyError
            type_counts = filtered_data['Primary Type'].value_counts()
            top_15_types = type_counts[type_counts > 0].head(15).index
            
            # Use reindex to safely select, dropping any missing keys if mismatch occurs
            arrest_rates_top15 = arrest_rates.reindex(top_15_types).dropna().sort_values(ascending=True)
            
            # Use color mapping to reflect warning logic (below 15% is light red, above is sky blue)
            fig_arrest = px.bar(
                x=arrest_rates_top15.values,
                y=arrest_rates_top15.index,
                orientation='h',
                text_auto='.1%',
                title="Arrest Efficiency (Rate by Crime Type)",
                color=arrest_rates_top15.values,
                color_continuous_scale=['#ff9999', '#66b3ff'], 
                range_color=[0, 0.5] # Focus on differentiated range
            )
            st.plotly_chart(fig_arrest, width="stretch")

        with col_a2:
            # --- Domestic Violence Ratio Analysis ---
            st.markdown("#### Domestic Nature Analysis")
            # Calculate top 10 crimes with highest domestic background ratio among top 50 common crimes
            common_crimes = filtered_data['Primary Type'].value_counts().head(50).index
            domestic_rates = filtered_data[filtered_data['Primary Type'].isin(common_crimes)].groupby('Primary Type', observed=True)['Domestic'].mean().sort_values(ascending=False)
            top_domestic = domestic_rates.head(10)

            fig_domestic = px.bar(
                x=top_domestic.values,
                y=top_domestic.index,
                orientation='h',
                text_auto='.1%',
                title="Top 10 Crimes by Domestic Incident Rate",
                color=top_domestic.index,
                color_discrete_sequence=px.colors.sequential.Magma
            )
            fig_domestic.update_layout(showlegend=False)
            st.plotly_chart(fig_domestic, width="stretch")
        

        st.divider()

        # ================= 3. Peak Hours & Seasonality (Scheduling & Seasonality) =================
        st.markdown("### 📅 Temporal Behavior Patterns")
        
        # --- Crime Schedule ---
        st.markdown("#### Crime Schedule: Peak Hours by Type")
        # Use filtered_data to ensure the heatmap reflects the sidebar selection
        top_10_crimes = filtered_data['Primary Type'].value_counts().head(10).index
        df_subset = filtered_data[filtered_data['Primary Type'].isin(top_10_crimes)]
        
        # Cross-analysis of crime type and hour
        time_crime_matrix = pd.crosstab(df_subset['Primary Type'], df_subset['Hour'], normalize='index')
        
        fig_time_heat = px.imshow(
            time_crime_matrix,
            labels=dict(x="Hour of Day", y="Crime Type", color="Probability"),
            # color_continuous_scale='coolwarm',
            title="Peak Hours Heatmap (Distribution by Crime Type)",
            aspect="auto", 
        )
        st.plotly_chart(fig_time_heat, width="stretch")
        

        # --- Seasonal Trends (Selected Crimes) ---
        st.markdown("#### Seasonal Trends for Top Crimes")
        # Dynamically select top 5 crimes from the current filtered data
        target_crimes = filtered_data['Primary Type'].value_counts().head(5).index
        
        if not target_crimes.empty:
            monthly_trends = filtered_data[filtered_data['Primary Type'].isin(target_crimes)].groupby(['Month', 'Primary Type'], observed=True).size().unstack()
            
            fig_seasonal = px.line(
                monthly_trends,
                x=monthly_trends.index,
                y=monthly_trends.columns,
                markers=True,
                title="Monthly Incident Trends: Top Categories",
                labels={'value': 'Number of Incidents', 'Month': 'Month'}
            )
            fig_seasonal.update_xaxes(dtick=1)
            st.plotly_chart(fig_seasonal, width="stretch")
    else:
        st.info("No data available for correlation analysis.")

# ================= 5. Footer =================
st.markdown("---")
st.markdown("© 2026 IT5006 Project Team7 | Data Source: Chicago Data Portal")