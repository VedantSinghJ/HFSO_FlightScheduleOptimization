# Honeywell Flight Scheduling Optimizer (HFSO)
 
*Author:* Vedant Singh Jadon  
*Project Type:* Honeywell Hackathon Prototype
 
## Overview
 
The Honeywell Flight Scheduling Optimizer (HFSO) is an AI-powered tool designed to address the critical scheduling challenges faced by busy airports, particularly Mumbai (BOM) and Delhi (DEL). This system analyzes flight data to provide actionable insights for optimizing runway capacity utilization and minimizing cascading delays.
 
## Problem Statement
 
Due to capacity limitations and heavy passenger load, flight operations at busy airports are becoming a scheduling nightmare. Controllers and operators need to find efficiency in scheduling within system constraints and find means to de-congest flight traffic.
 
## Key Features
 
### 🤖 Natural Language Processing Interface
- Query flight data using natural language prompts
- Ask questions like "What are the busiest slots in DEL?" or "Best time to depart Mumbai?"
- Intelligent intent recognition for various scheduling queries
 
### 📊 Advanced Analytics
- *Runway Load Index (RLI)*: Real-time capacity utilization metrics
- *Busiest Slots Analysis*: Identify peak congestion periods
- *Optimal Timing*: Find best departure/arrival windows with minimal delays
- *Cascading Impact Analysis*: Identify flights with highest downstream delay effects
 
### 🔧 Schedule Optimization
- *Greedy Slot Tuner*: Intelligent flight rescheduling recommendations
- *Capacity Management*: Adjust for weather conditions and runway limitations
- *Impact Visualization*: Before/after comparisons of schedule changes
 
### 🧪 Scenario Planning
- A/B testing for different capacity and weather scenarios
- Multi-parameter optimization (capacity, weather factor, shift limits)
- Export capabilities for operational planning
 
## System Architecture
 
```
├── src/
│   ├── processing.py     # Data ingestion & normalization
│   ├── metrics.py        # Core analytics functions
│   ├── optimizer.py      # Schedule tuning algorithms
│   ├── influence.py      # Cascading impact analysis
│   ├── nlp.py           # Natural language interface
│   └── pipeline.py      # Data processing pipeline
├── app.py               # Streamlit web application
└── data/               # Processed flight data
```
 
## Installation
 
### Prerequisites
- Python 3.8+
- Required packages (install via pip):
 
```bash
pip install streamlit pandas numpy duckdb altair openpyxl
```
 
### Quick Start
 
1. *Clone the repository*
```bash
git clone <repository-url>
cd flight-scheduling-optimizer
```
 
2. *Prepare your data*
   - Place your flight data file (Excel/CSV) in the project directory
   - Supported formats: .xlsx, .xls, .csv
 
3. *Process the data* (optional - for better performance)
```bash
python src/pipeline.py --input Flight_Data.xlsx --out data/processed.parquet
```
 
4. *Launch the application*
```bash
streamlit run app.py
```
 
5. *Access the web interface*
   - Open your browser to http://localhost:8501
 
## Usage Guide
 
### Data Input Options
1. *Upload File*: Use the sidebar to upload Excel/CSV files directly
2. *Processed Data*: Use pre-processed parquet files for faster loading
 
### Main Features
 
#### 🤖 NLP Assistant Tab
Ask natural language questions:
- "What are the busiest time slots in BOM?"
- "Best time to depart from Delhi?"
- "Show me runway load index for capacity 10"
- "Which flights have the biggest cascading impact?"
 
#### 📈 Analytics Tab
- View busiest departure slots
- Find optimal operating windows
- Analyze runway load index (RLI)
- Visualize congestion heatmaps
- Identify high-impact flights for delay propagation
 
#### 🔧 Tuner Tab
- Set runway capacity per 5-minute window
- Configure maximum flight shift time
- Generate schedule optimization recommendations
- Compare before/after scenarios
- Export tuned schedules
 
#### 🧪 Scenarios Tab
- Create multiple "what-if" scenarios
- Compare different capacity/weather combinations
- A/B testing for operational planning
 
### Key Metrics Explained
 
- *Runway Load Index (RLI)*: Ratio of actual movements to theoretical capacity
  - RLI < 1.0: Under capacity
  - RLI > 1.0: Over capacity (delays expected)
 
- *Cascading Impact Score*: Measures how delays from one flight affect downstream operations
 
## Data Requirements
 
### Input Data Format
The system expects flight schedule data with these columns:
- Flight identification (flight number, carrier)
- Scheduled/actual departure times
- Scheduled/actual arrival times
- Origin/destination airports
- Aircraft tail numbers (optional)
- Date information
 
### Supported Data Sources
- FlightRadar24 exports
- FlightAware data
- Custom Excel/CSV files
- Airport operations data
 
## Sample Queries
 
### Analytics Queries
```
"Show busiest slots in Mumbai"
"Best arrival time at DEL"
"Runway congestion with capacity 8"
"Weather impact analysis"
```
 
### Optimization Queries
```
"Tune schedule for capacity 10"
"Shift flights by maximum 20 minutes"
"Top cascading impact flights"
"Consolidation patterns"
```
 
## Technical Implementation
 
### Core Algorithms
 
1. *Runway Load Index Calculation*
 
   ```
   RLI = (movements × occupancy_time) / (capacity × weather_factor)
   ```
 
2. *Cascading Impact Analysis*
   - Graph-based propagation model
   - Aircraft rotation dependencies
   - Time-slot adjacency effects
 
3. *Greedy Schedule Optimization*
   - Prioritizes low-impact flight moves
   - Respects operational constraints
   - Minimizes total system delay
 
### Performance Features
- Efficient data processing with DuckDB
- Cached computations for real-time queries
- Streamlined Excel parsing for complex formats
 
## Configuration Options
 
### Capacity Settings
- Runway capacity per 5-minute window (default: 12)
- Weather impact factor (0.5 - 1.5)
- Maximum flight shift time (5-60 minutes)
 
### Analysis Parameters
- Time window sizes (5-30 minutes)
- Cascading analysis depth (1-5 steps)
- Top results limits
 
## Export Capabilities
 
- *Tuned Schedules*: CSV export with before/after comparisons
- *Analytics Reports*: Detailed metrics and insights
- *Operational Briefs*: Summary statistics for planning
 
## Limitations & Considerations
 
- Synthetic delay modeling when actual delay data is sparse
- Greedy optimization (may not find global optimum)
- Static capacity modeling (doesn't account for dynamic conditions)
- Requires quality input data for accurate results
 
## Future Enhancements
 
- Machine learning-based delay prediction
- Real-time data integration
- Multi-airport coordination
- Advanced weather modeling
- Integration with ATC systems
 
## Support
 
For technical issues or feature requests, please refer to the project documentation or contact the development team.