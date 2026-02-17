"""
Main pipeline orchestrator for idea intake processing.

This script:
1. Loads and cleans data from an Excel file
2. Performs clustering and analysis
3. Generates AI-powered recommendations using watsonx.ai
4. Exports results as JSON

Usage:
    python3 main.py --input data/your_file.xlsx --output results/output.json
    python3 main.py  # Uses default paths
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv

# Import cleaning functions
from scripts.pipeline_cleaning import clean_pipeline

# Import analysis functions
from scripts.pipeline_analysis import (
    build_input_object,
    load_prompt_template,
    make_final_prompt
)

from ibm_watsonx_ai.foundation_models import ModelInference


def setup_watsonx_model():
    """Initialize watsonx.ai model with credentials from environment."""
    load_dotenv()
    
    API_KEY = os.getenv('API_KEY')
    PROJECT_ID = os.getenv('PROJECT_ID')
    MODEL_ID = os.getenv('MODEL_ID')
    URL = os.getenv('URL')
    
    # Validate credentials
    missing = []
    if not API_KEY:
        missing.append('API_KEY')
    if not PROJECT_ID:
        missing.append('PROJECT_ID')
    if not MODEL_ID:
        missing.append('MODEL_ID')
    if not URL:
        missing.append('URL')
    
    if missing:
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")
    
    PARAMS = {
        "decoding_method": "greedy",
        "max_new_tokens": 1000,
        "min_new_tokens": 0,
        "repetition_penalty": 1,
        "stop_sequences": ["<|eom_id|>"]
    }
    
    model = ModelInference(
        model_id=MODEL_ID,
        params=PARAMS,
        credentials={
            "url": URL,
            "apikey": API_KEY
        },
        project_id=PROJECT_ID,
    )
    
    return model


def run_pipeline(input_file, output_file=None, similarity_threshold=0.78, save_intermediate=True):
    """
    Run the complete idea intake pipeline.
    
    Args:
        input_file: Path to input Excel file
        output_file: Path to output JSON file (optional)
        similarity_threshold: Threshold for clustering similar ideas (0-1)
        save_intermediate: Whether to save intermediate cleaned CSV
    
    Returns:
        dict: Final analysis results as JSON object
    """
    print("="*70)
    print("IDEA INTAKE AI PIPELINE")
    print("="*70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Step 1: Data Cleaning
    print("STEP 1: Data Cleaning")
    print("-" * 70)
    
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    # Create temporary cleaned CSV path
    temp_csv = Path("data/temp_cleaned_ideas.csv")
    if save_intermediate:
        temp_csv = Path("data/cleaned_ideas.csv")
    
    cleaned_df = clean_pipeline(input_path, temp_csv)
    print(f"Cleaned data: {len(cleaned_df)} rows\n")
    
    # Step 2: Clustering and Analysis
    print("STEP 2: Clustering and Analysis")
    print("-" * 70)
    
    input_object = build_input_object(cleaned_df, similarity_threshold=similarity_threshold)
    
    print(f"Total clusters identified: {input_object['dataset_summary']['total_clusters']}")
    print(f"Initiative candidates: {len(input_object['initiative_candidates'])}")
    
    # Display cluster summary
    print("\nCluster Summary:")
    for candidate in input_object['initiative_candidates'][:5]:  # Show top 5
        print(f"  - Cluster {candidate['cluster_id']}: {candidate['popularity_count']} ideas")
    
    if len(input_object['initiative_candidates']) > 5:
        print(f"  ... and {len(input_object['initiative_candidates']) - 5} more clusters")
    print()
    
    # Step 3: AI-Powered Recommendations
    print("STEP 3: Generating AI Recommendations")
    print("-" * 70)
    
    try:
        model = setup_watsonx_model()
        print("✓ Connected to watsonx.ai")
        
        # Load prompt template
        prompt_template = load_prompt_template("prompts/idea_summary_prompt.txt")
        final_prompt = make_final_prompt(prompt_template, input_object)
        
        print("✓ Generating recommendations...")
        response = model.generate(final_prompt)
        raw_text = response["results"][0]["generated_text"]
        
        # Parse JSON from response
        try:
            # Try to extract JSON from the response
            json_start = raw_text.find('{')
            json_end = raw_text.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                json_text = raw_text[json_start:json_end]
                result = json.loads(json_text)
            else:
                result = json.loads(raw_text)
        except json.JSONDecodeError:
            print("⚠ Warning: Could not parse JSON from model response")
            result = {
                "raw_response": raw_text,
                "error": "Failed to parse JSON from model output"
            }
        
        print("Recommendations generated\n")
        
    except Exception as e:
        print(f"AI generation failed: {str(e)}")
        print("Returning analysis without AI recommendations\n")
        result = {
            "input_analysis": input_object,
            "error": f"AI generation failed: {str(e)}"
        }
    
    # Add metadata
    result["metadata"] = {
        "input_file": str(input_file),
        "total_ideas_processed": len(cleaned_df),
        "total_clusters": input_object['dataset_summary']['total_clusters'],
        "similarity_threshold": similarity_threshold,
        "generated_at": datetime.now().isoformat(),
        "pipeline_version": "1.0.0"
    }
    
    # Step 4: Export Results
    print("STEP 4: Exporting Results")
    print("-" * 70)
    
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"Results saved to: {output_file}")
    
    # Clean up temporary file if not saving intermediate
    if not save_intermediate and temp_csv.exists():
        temp_csv.unlink()
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print("="*70)
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    return result


def main():
    """Main entry point for the pipeline."""
    parser = argparse.ArgumentParser(
        description="Process idea intake data through cleaning, clustering, and AI analysis"
    )
    parser.add_argument(
        '--input',
        '-i',
        default='data/Horizon_Market_Idea_Sample_Data.xlsx',
        help='Path to input Excel file (default: data/Horizon_Market_Idea_Sample_Data.xlsx)'
    )
    parser.add_argument(
        '--output',
        '-o',
        default='results/analysis_output.json',
        help='Path to output JSON file (default: results/analysis_output.json)'
    )
    parser.add_argument(
        '--similarity-threshold',
        '-s',
        type=float,
        default=0.78,
        help='Similarity threshold for clustering (0-1, default: 0.78)'
    )
    parser.add_argument(
        '--no-intermediate',
        action='store_true',
        help='Do not save intermediate cleaned CSV file'
    )
    
    args = parser.parse_args()
    
    try:
        result = run_pipeline(
            input_file=args.input,
            output_file=args.output,
            similarity_threshold=args.similarity_threshold,
            save_intermediate=not args.no_intermediate
        )
        
        # Print summary
        if "top_3_recommendations" in result:
            print("\nTOP 3 RECOMMENDATIONS:")
            for rec in result["top_3_recommendations"]:
                print(f"\n{rec['rank']}. {rec['initiative_title']}")
                print(f"   Cluster ID: {rec['cluster_id']}")
                print(f"   Feasibility: {rec['why_now']['feasibility_score_1_to_5']}/5")
                print(f"   Value: {rec['why_now']['value_score_1_to_5']}/5")
        
        return 0
        
    except Exception as e:
        print(f"\nERROR: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

# Made with Bob
