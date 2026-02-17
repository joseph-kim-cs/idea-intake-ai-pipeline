# Idea Intake AI Pipeline

An automated pipeline for processing, cleaning, clustering, and analyzing idea intake form submissions using IBM watsonx.ai. Available as both a **web application** and **command-line tool**.

## 🌟 Features

- **Web Interface** - Beautiful, user-friendly UI with drag & drop file upload
- **CLI Tool** - Command-line interface for automation and scripting
- **Data Cleaning** - Automatic normalization and validation
- **Smart Clustering** - Groups similar ideas using ML algorithms
- **AI Recommendations** - Top 3 prioritized initiatives with feasibility/value scores
- **JSON Export** - Structured output for further analysis

## 🚀 Quick Start

### Installation

1. **Install dependencies:**
```bash
pip3 install -r requirements.txt
```

2. **Set up environment variables** (create `.env` file):
```env
API_KEY=your_watsonx_api_key
PROJECT_ID=your_project_id
MODEL_ID=your_model_id
URL=your_watsonx_url
```

### Option 1: Web Application (Recommended)

**Start the web server:**
```bash
python3 main.py
```

**Open your browser:**
Navigate to **http://localhost:5000**

**Upload and analyze:**
1. Drag & drop your Excel file or click to browse
2. Adjust similarity threshold (optional)
3. Click "Analyze Ideas"
4. View recommendations and download JSON

### Option 2: Command Line

**Basic usage:**
```bash
python3 -c "from routes.pipeline_routes import run_pipeline; run_pipeline('data/your_file.xlsx', 'results/output.json')"
```

**Or create a CLI script** (`cli.py`):
```python
import sys
from routes.pipeline_routes import run_pipeline

if __name__ == "__main__":
    input_file = sys.argv[1] if len(sys.argv) > 1 else "data/Horizon_Market_Idea_Sample_Data.xlsx"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "results/analysis_output.json"
    run_pipeline(input_file, output_file)
```

Then run:
```bash
python3 cli.py data/your_file.xlsx results/output.json
```

## 📁 Project Structure

```
idea-intake-ai-pipeline/
├── main.py                          # Flask web application
├── routes/
│   └── pipeline_routes.py          # Core pipeline logic
├── templates/
│   └── index.html                  # Web UI
├── static/
│   └── css/
│       └── style.css               # Styling
├── scripts/
│   ├── pipeline_cleaning.py        # Data cleaning functions
│   └── pipeline_analysis.py        # Clustering & analysis
├── prompts/
│   └── idea_summary_prompt.txt     # AI prompt template
├── uploads/                        # Temporary uploads (auto-created)
├── results/                        # Generated JSON results
└── data/                           # Sample data
    └── Horizon_Market_Idea_Sample_Data.xlsx
```

## 🎨 Web Interface Guide

### Upload Excel File

Your Excel file should contain these columns:
- **Name** - Submitter name
- **Software Portfolio Domain** - Domain (Data, AI, Automation, etc.)
- **Other Domain** - Alternative domain if "Other" selected
- **Product** - Product name (watsonx.ai, watsonx.orchestrate, etc.)
- **Idea Description** - Full idea description (required)
- **Asset Links** - URLs to related assets (optional)

### Similarity Threshold

Controls how strictly ideas are clustered:
- **Lower (0.6-0.75)**: More aggressive clustering, fewer groups
- **Higher (0.8-0.9)**: Stricter clustering, more groups
- **Default: 0.78** - Balanced approach

### Results Display

The web interface shows:
- **Summary Statistics** - Total ideas, clusters, processing time
- **Executive Summary** - Key insights and themes
- **Top 3 Recommendations** - Prioritized initiatives with:
  - Feasibility scores (1-5)
  - Value scores (1-5)
  - Detailed justifications
  - First milestone deliverables

## 📊 Pipeline Components

### 1. Data Cleaning (`scripts/pipeline_cleaning.py`)

**Functions:**
- `remove_test_entries()` - Removes test/placeholder data
- `normalize_domains()` - Consolidates domain fields
- `clean_product_names()` - Standardizes product names
- `extract_and_validate_urls()` - Extracts URLs from asset links
- `clean_text_fields()` - Normalizes text formatting
- `handle_missing_values()` - Manages missing data
- `add_metadata()` - Adds word count, asset link flags, IDs

### 2. Analysis & Clustering (`scripts/pipeline_analysis.py`)

**Functions:**
- `cluster_ideas()` - Groups similar ideas using TF-IDF + hierarchical clustering
- `build_input_object()` - Creates structured input for AI model
- `representative_idea()` - Selects most representative idea per cluster

### 3. AI Recommendations (`routes/pipeline_routes.py`)

Uses IBM watsonx.ai to:
- Generate executive summary bullets
- Recommend top 3 initiatives
- Score feasibility (1-5) and value (1-5)
- Suggest first milestone deliverables

## 🔧 API Endpoints (Web Mode)

### `GET /`
Main web interface

### `POST /upload`
Upload and process Excel file

**Request:**
- `file`: Excel file (multipart/form-data)
- `similarity_threshold`: Float (0-1, optional)

**Response:**
```json
{
  "success": true,
  "result": {
    "executive_summary_bullets": [...],
    "top_3_recommendations": [...],
    "metadata": {...}
  },
  "output_file": "20260217_123456_analysis.json"
}
```

### `GET /download/<filename>`
Download result JSON file

### `GET /health`
Health check endpoint

## 📤 Output Format

The pipeline generates JSON with:

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

## 🛠️ Customization

### Change AI Model Parameters

Edit `routes/pipeline_routes.py`:

```python
PARAMS = {
    "decoding_method": "greedy",
    "max_new_tokens": 1000,
    "min_new_tokens": 0,
    "repetition_penalty": 1,
    "stop_sequences": ["<|eom_id|>"]
}
```

### Modify Prompt Template

Edit `prompts/idea_summary_prompt.txt` to change:
- Output format
- Number of recommendations
- Scoring criteria
- Analysis style

### Customize Web UI

Edit `static/css/style.css` to change:
- Colors and gradients
- Layout and spacing
- Fonts and typography
- Responsive breakpoints

### Adjust File Upload Limits

Edit `main.py`:

```python
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
```

## 🐛 Troubleshooting

### Web Application Issues

**Port Already in Use:**
```python
# Change port in main.py
app.run(debug=True, host='0.0.0.0', port=5001)
```

**File Upload Fails:**
- Check file size (max 16MB)
- Verify file format (.xlsx or .xls)
- Ensure Excel file has required columns

### API Authentication Errors
- Verify `.env` file has correct credentials
- Check API key hasn't expired
- Ensure project ID is correct

### Clustering Issues
- Try adjusting `similarity_threshold`
- Check that idea descriptions are substantial (>15 characters)
- Verify data cleaning removed test entries

## 🚀 Production Deployment

### Using Gunicorn

```bash
pip3 install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 main:app
```

### Using Docker

Create `Dockerfile`:

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "main:app"]
```

Build and run:

```bash
docker build -t idea-intake-pipeline .
docker run -p 5000:5000 --env-file .env idea-intake-pipeline
```

## 📊 Performance Tips

- **Smaller files process faster** - Consider filtering data before upload
- **Adjust similarity threshold** - Higher values = faster clustering
- **Use production mode** - Set `debug=False` in `main.py` for production
- **Enable caching** - Cache model initialization for repeated requests

## 🔒 Security Notes

- Never commit `.env` file to version control
- Use HTTPS in production
- Implement authentication for production deployments
- Sanitize file uploads
- Set appropriate CORS policies
- Limit file upload sizes

## 📝 License

MIT License

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

---

**Built with ❤️ using Flask, IBM watsonx.ai, scikit-learn, and modern web technologies**