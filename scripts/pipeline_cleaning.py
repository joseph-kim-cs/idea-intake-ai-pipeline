import pandas as pd
import re
from pathlib import Path


def load_data(file_path):
    df = pd.read_excel(file_path, sheet_name='horizon market idea intake form', header=2)
    return df

'''
def remove_test_entries(df):
    # Remove rows where Name or Idea Description contains 'test' (case-insensitive)
    test_pattern = r'\btest\b'
    mask = (
        df['Name'].str.contains(test_pattern, case=False, na=False, regex=True) |
        df['Idea Description'].str.contains(test_pattern, case=False, na=False, regex=True)
    )
    df_cleaned = df[~mask].copy()
    print(f"Removed {mask.sum()} test entries")
    return df_cleaned
'''

def normalize_domains(df):
    #Normalize domain fields by consolidating 'Other Domain' into main domain.
    df = df.copy()
    
    # If 'Software Portfolio Domain' is 'Other' and 'Other Domain' has a value, use 'Other Domain'
    mask = (df['Software Portfolio Domain'] == 'Other') & (df['Other Domain'].notna())
    df.loc[mask, 'Software Portfolio Domain'] = df.loc[mask, 'Other Domain']
    
    # Standardize domain names; will add more later
    domain_mapping = {
        'AI': 'AI',
        'ai': 'AI',
        'Data': 'Data',
        'data': 'Data',
        'Automation': 'Automation',
        'automation': 'Automation'
    }
    df['Software Portfolio Domain'] = df['Software Portfolio Domain'].map(
        lambda x: domain_mapping.get(x, x) if pd.notna(x) else x
    )
    
    df = df.drop(columns=['Other Domain'], errors='ignore')
    
    return df


def clean_product_names(df):
    # standardize product names
    df = df.copy()
    
    # Product name mappings for standardization; may add more later
    product_mapping = {
        'watsonx.ai': 'watsonx.ai',
        'wx.ai': 'watsonx.ai',
        'watsonx.orchestrate': 'watsonx.orchestrate',
        'gov': 'watsonx.governance',  # Assuming 'gov' refers to governance
    }
    
    df['Product﻿'] = df['Product﻿'].map(
        lambda x: product_mapping.get(x, x) if pd.notna(x) else x
    )
    
    # Rename column to remove special character
    df = df.rename(columns={'Product﻿': 'Product'})
    
    return df


def extract_and_validate_urls(df):
    """Extract and validate URLs from Asset Links field."""
    df = df.copy()
    
    # Rename column to remove special character
    df = df.rename(columns={'Asset ﻿Links': 'Asset_Links'})
    
    def extract_url(text):
        """Extract URL from text, handling markdown-style links."""
        if pd.isna(text) or text.strip() == '':
            return None
        
        # Pattern for markdown links: [text](url)
        markdown_pattern = r'\[.*?\]\((https?://[^\)]+)\)'
        markdown_match = re.search(markdown_pattern, text)
        if markdown_match:
            return markdown_match.group(1)
        
        # Pattern for plain URLs
        url_pattern = r'https?://[^\s]+'
        url_match = re.search(url_pattern, text)
        if url_match:
            return url_match.group(0)
        
        # Pattern for URLs without protocol
        domain_pattern = r'github\.com/[^\s]+'
        domain_match = re.search(domain_pattern, text)
        if domain_match:
            return f"https://{domain_match.group(0)}"
        
        return text
    
    df['Asset_Links'] = df['Asset_Links'].apply(extract_url)
    
    return df


def clean_text_fields(df):
    """Clean and standardize text fields."""
    df = df.copy()
    
    text_columns = ['Name', 'Idea Description']
    
    for col in text_columns:
        if col in df.columns:
            # Strip whitespace
            df[col] = df[col].str.strip()
            
            # Replace multiple spaces with single space
            df[col] = df[col].str.replace(r'\s+', ' ', regex=True)
    
    return df


def handle_missing_values(df):
    """Handle missing values appropriately."""
    df = df.copy()
    
    # Drop rows where critical fields are missing
    critical_fields = ['Name', 'Idea Description']
    df = df.dropna(subset=critical_fields, how='any')
    
    # Fill non-critical missing values
    df['Software Portfolio Domain'] = df['Software Portfolio Domain'].fillna('Unspecified')
    df['Product'] = df['Product'].fillna('Unspecified')
    df['Asset_Links'] = df['Asset_Links'].fillna('')
    
    # drop subitems column
    if 'Subitems' in df.columns and df['Subitems'].isna().all():
        df = df.drop(columns=['Subitems'])
    
    return df


def add_metadata(df):
    """Add useful metadata columns for analysis."""
    df = df.copy()
    
    # Add idea length (word count)
    df['Idea_Word_Count'] = df['Idea Description'].str.split().str.len()
    
    # Add flag for whether asset link is provided
    df['Has_Asset_Link'] = df['Asset_Links'].apply(lambda x: bool(x and x.strip()))
    
    # Add unique ID
    df['Idea_ID'] = range(1, len(df) + 1)
    
    return df


def clean_pipeline(input_file, output_file=None):
    """
    Main cleaning pipeline that orchestrates all cleaning steps.
    
    Args:
        input_file: Path to input Excel file
        output_file: Path to output CSV file (optional)
    
    Returns:
        Cleaned DataFrame
    """
    print("Starting data cleaning pipeline...")
    
    # Load data
    print(f"Loading data from {input_file}...")
    df = load_data(input_file)
    print(f"Loaded {len(df)} rows")
    
    # Apply cleaning steps
    #df = remove_test_entries(df)
    df = normalize_domains(df)
    df = clean_product_names(df)
    df = extract_and_validate_urls(df)
    df = clean_text_fields(df)
    df = handle_missing_values(df)
    df = add_metadata(df)
    
    print(f"Cleaning complete. Final dataset: {len(df)} rows")
    
    if output_file:
        df.to_csv(output_file, index=False)
        print(f"Cleaned data saved to {output_file}")
    
    return df


if __name__ == "__main__":
    input_file = Path("data/Horizon_Market_Idea_Sample_Data.xlsx")
    output_file = Path("data/cleaned_ideas.csv")
    
    cleaned_df = clean_pipeline(input_file, output_file)
    
    print("CLEANING SUMMARY")
    print(f"Total ideas: {len(cleaned_df)}")
    print(f"\nDomain distribution:")
    print(cleaned_df['Software Portfolio Domain'].value_counts())
    print(f"\nProduct distribution:")
    print(cleaned_df['Product'].value_counts())
    print(f"\nIdeas with asset links: {cleaned_df['Has_Asset_Link'].sum()}")
    print(f"\nAverage idea word count: {cleaned_df['Idea_Word_Count'].mean():.1f}")
