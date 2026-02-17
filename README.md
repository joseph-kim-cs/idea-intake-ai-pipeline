# Idea Intake AI Pipeline

An automated pipeline for processing, cleaning, clustering, and analyzing idea intake form submissions using IBM watsonx.ai.

## Overview

This pipeline takes raw Excel data from idea intake forms and:
1. **Cleans and normalizes** the data (removes test entries, standardizes fields)
2. **Clusters similar ideas** using TF-IDF vectorization and hierarchical clustering
3. **Generates AI-powered recommendations** using IBM watsonx.ai
4. **Exports structured JSON** with executive summaries and prioritized initiatives

## Prerequisites

- Python 3.11+
- IBM watsonx.ai credentials (API key, Project ID, Model ID, URL)

## Installation

1. Clone the repository:
```bash
cd idea-intake-ai-pipeline
```

2. Install dependencies:
```bash
pip3 install -r requirements.txt
```

3. Set up environment variables in `.env`:
```env
API_KEY=your_watsonx_api_key
PROJECT_ID=your_project_id
MODEL_ID=your_model_id
URL=your_watsonx_url
```

## Usage

### Basic Usage

Run the pipeline with default settings:
```bash
python3 main.py
```

This will:
- Read from `data/Horizon_Market_Idea_Sample_Data.xlsx`
- Save cleaned data to `data/cleaned_ideas.csv`
- Export analysis to `results/analysis_output.json`

### Custom Input/Output

Specify custom file paths:
```bash
python3 main.py --input path/to/your/file.xlsx --output path/to/output.json
```

### Advanced Options

```bash
python3 main.py \
  --input data/my_ideas.xlsx \
  --output results/my_analysis.json \
  --similarity-threshold 0.85 \
  --no-intermediate
```

**Options:**
- `--input, -i`: Path to input Excel file (default: `data/Horizon_Market_Idea_Sample_Data.xlsx`)
- `--output, -o`: Path to output JSON file (default: `results/analysis_output.json`)
- `--similarity-threshold, -s`: Clustering similarity threshold 0-1 (default: 0.78, higher = stricter clustering)
- `--no-intermediate`: Don't save intermediate cleaned CSV file

## Pipeline Components

### 1. Data Cleaning (`scripts/pipeline_cleaning.py`)

**Functions:**
- `remove_test_entries()` - Removes test/placeholder data
- `normalize_domains()` - Consolidates domain fields
- `clean_product_names()` - Standardizes product names
- `extract_and_validate_urls()` - Extracts URLs from asset links
- `clean_text_fields()` - Normalizes text formatting
- `handle_missing_values()` - Manages missing data
- `add_metadata()` - Adds word count, asset link flags, IDs

**Output:** Cleaned CSV with standardized columns

### 2. Analysis & Clustering (`scripts/pipeline_analysis.py`)

**Functions:**
- `cluster_ideas()` - Groups similar ideas using TF-IDF + hierarchical clustering
- `build_input_object()` - Creates structured input for AI model
- `representative_idea()` - Selects most representative idea per cluster

**Output:** Clustered data with initiative candidates

### 3. AI Recommendations (`main.py`)

Uses IBM watsonx.ai to:
- Generate executive summary bullets
- Recommend top 3 initiatives
- Score feasibility (1-5) and value (1-5)
- Suggest first milestone deliverables

## Output Format

The pipeline generates a JSON file with:

```json
{
  "executive_summary_bullets": [
    "Key insight 1",
    "Key insight 2",
    "..."
  ],
  "top_3_recommendations": [
    {
      "rank": 1,
      "initiative_title": "Initiative Name",
      "cluster_id": 0,
      "what_it_is": "Description",
      "why_now": {
        "popularity": "Explanation",
        "feasibility_score_1_to_5": 4,
        "feasibility_justification": "Reasoning",
        "value_score_1_to_5": 5,
        "value_justification": "Reasoning"
      },
      "first_milestone_deliverable": "Next step"
    }
  ],
  "metadata": {
    "input_file": "path/to/input.xlsx",
    "total_ideas_processed": 42,
    "total_clusters": 5,
    "similarity_threshold": 0.78,
    "generated_at": "2026-02-17T10:30:00",
    "pipeline_version": "1.0.0"
  }
}
```

## Input Data Format

The Excel file should have these columns:
- **Name** - Submitter name
- **Software Portfolio Domain** - Domain (Data, AI, Automation, etc.)
- **Other Domain** - Alternative domain if "Other" selected
- **Product** - Product name (watsonx.ai, watsonx.orchestrate, etc.)
- **Idea Description** - Full idea description
- **Asset Links** - URLs to related assets

## Customization

### Modify Clustering Threshold

Adjust similarity threshold (0-1):
- **Lower (0.6-0.75)**: More aggressive clustering, fewer clusters
- **Higher (0.8-0.9)**: Stricter clustering, more clusters

### Customize AI Prompt

Edit `prompts/idea_summary_prompt.txt` to change:
- Output format
- Scoring criteria
- Recommendation style
- Number of recommendations

### Adjust Model Parameters

Edit `main.py` or `scripts/pipeline_analysis.py`:
```python
PARAMS = {
    "decoding_method": "greedy",
    "max_new_tokens": 1000,
    "min_new_tokens": 0,
    "repetition_penalty": 1,
    "stop_sequences": ["<|eom_id|>"]
}
```

## Project Structure

```
idea-intake-ai-pipeline/
├── main.py                          # Main pipeline orchestrator
├── requirements.txt                 # Python dependencies
├── .env                            # Environment variables (not in git)
├── README.md                       # This file
├── data/
│   ├── Horizon_Market_Idea_Sample_Data.xlsx
│   └── cleaned_ideas.csv           # Generated by pipeline
├── scripts/
│   ├── pipeline_cleaning.py        # Data cleaning functions
│   └── pipeline_analysis.py        # Clustering & analysis
├── prompts/
│   └── idea_summary_prompt.txt     # AI prompt template
└── results/
    └── analysis_output.json        # Generated recommendations
```

## Troubleshooting

### Missing Dependencies
```bash
pip3 install pandas openpyxl scikit-learn ibm-watsonx-ai python-dotenv
```

### API Authentication Errors
- Verify `.env` file has correct credentials
- Check API key has not expired
- Ensure project ID is correct

### Clustering Issues
- Try adjusting `--similarity-threshold`
- Check that idea descriptions are substantial (>15 characters)
- Verify data cleaning removed test entries

## License

MIT License

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request