# Usage Examples

This document provides comprehensive examples of how to use the Job Description Parser in various scenarios.

## Table of Contents

- [Basic Usage](#basic-usage)
- [Batch Processing](#batch-processing)
- [Custom Configuration](#custom-configuration)
- [CLI Usage](#cli-usage)
- [Performance Optimization](#performance-optimization)
- [Integration Examples](#integration-examples)
- [Error Handling](#error-handling)

## Basic Usage

### Simple Job Parsing

```python
from job_parser import JobDescriptionParser

# Initialize parser
parser = JobDescriptionParser()

# Sample job description
job_text = """
Senior Software Engineer - Full Stack
TechCorp Inc.

We are seeking a Senior Software Engineer with 5+ years of experience 
to join our dynamic team. The ideal candidate will have expertise in 
Python, React, and cloud technologies.

Requirements:
• 5+ years of software development experience
• Proficiency in Python and JavaScript
• Experience with React.js and Node.js
• Knowledge of AWS cloud services
• Familiarity with Docker and Kubernetes
• Strong problem-solving skills
• Excellent communication abilities

Preferred:
• Experience with machine learning
• Knowledge of PostgreSQL
• CI/CD pipeline experience
"""

# Parse the job description
result = parser.parse_job_description(job_text)

# Display results
print("🎯 Extracted Skills:")
for skill in result.skills_required:
    confidence = result.confidence_scores.get(skill, 0)
    print(f"  • {skill} (confidence: {confidence:.3f})")

print(f"\n📊 Experience Level: {result.experience_level}")
print(f"🛠 Tools Mentioned: {', '.join(result.tools_mentioned)}")

print("\n📋 Skills by Category:")
for category, skills in result.categories.items():
    if skills:
        print(f"  {category.title()}: {', '.join(skills)}")
```

### Working with Results

```python
from job_parser import JobDescriptionParser

parser = JobDescriptionParser()
result = parser.parse_job_description(job_text)

# Access different result attributes
print("Skills Analysis:")
print(f"Total skills found: {len(result.skills_required)}")
print(f"Technical skills: {len(result.categories.get('technical', []))}")
print(f"Soft skills: {len(result.categories.get('soft', []))}")
print(f"Tools: {len(result.categories.get('tools', []))}")

# Get top skills by confidence
sorted_skills = sorted(
    result.confidence_scores.items(), 
    key=lambda x: x[1], 
    reverse=True
)

print("\nTop 5 Skills by Confidence:")
for skill, confidence in sorted_skills[:5]:
    print(f"  {skill}: {confidence:.3f}")

# Check metadata
metadata = result.metadata
print(f"\nProcessing Stats:")
print(f"  Input length: {metadata['processing_stats']['input_length_words']} words")
print(f"  Sentences processed: {metadata['processing_stats']['sentences_processed']}")
print(f"  Total skills found: {metadata['processing_stats']['total_skills_found']}")
```

## Batch Processing

### Processing Multiple Jobs

```python
from job_parser import JobDescriptionParser
import json

# Load multiple job descriptions
job_descriptions = [
    "Python developer with Django and PostgreSQL experience...",
    "React frontend developer with TypeScript and Node.js...",
    "DevOps engineer with AWS, Docker, and Kubernetes...",
    "Data scientist with Python, scikit-learn, and TensorFlow...",
    "Full-stack developer with MEAN stack experience..."
]

# Initialize parser and optimize for batch processing
parser = JobDescriptionParser()
parser.optimize_for_batch_processing(
    expected_batch_size=len(job_descriptions),
    max_memory_mb=512.0
)

# Process all jobs
print(f"Processing {len(job_descriptions)} job descriptions...")
results = parser.parse_job_descriptions_batch(job_descriptions)

# Analyze batch results
total_skills = sum(len(r.skills_required) for r in results)
experience_levels = [r.experience_level for r in results]

print(f"\nBatch Processing Results:")
print(f"  Jobs processed: {len(results)}")
print(f"  Total skills extracted: {total_skills}")
print(f"  Average skills per job: {total_skills/len(results):.1f}")

# Experience level distribution
from collections import Counter
exp_distribution = Counter(experience_levels)
print(f"  Experience level distribution:")
for level, count in exp_distribution.items():
    print(f"    {level}: {count}")

# Most common skills across all jobs
all_skills = []
for result in results:
    all_skills.extend(result.skills_required)

skill_counts = Counter(all_skills)
print(f"\nMost Common Skills:")
for skill, count in skill_counts.most_common(10):
    print(f"  {skill}: {count}")
```

### Loading from JSON File

```python
import json
from job_parser import JobDescriptionParser

# Load job descriptions from JSON file
with open('job_descriptions.json', 'r') as f:
    jobs_data = json.load(f)

# Extract job texts
job_texts = []
job_metadata = []

for job in jobs_data:
    job_texts.append(job['description'])
    job_metadata.append({
        'id': job.get('id'),
        'title': job.get('title'),
        'company': job.get('company')
    })

# Process jobs
parser = JobDescriptionParser()
results = parser.parse_job_descriptions_batch(job_texts)

# Combine results with metadata
combined_results = []
for result, metadata in zip(results, job_metadata):
    combined_result = {
        'job_id': metadata['id'],
        'title': metadata['title'],
        'company': metadata['company'],
        'skills': result.skills_required,
        'experience_level': result.experience_level,
        'confidence_scores': result.confidence_scores
    }
    combined_results.append(combined_result)

# Save results
with open('parsed_results.json', 'w') as f:
    json.dump(combined_results, f, indent=2)

print(f"Processed {len(combined_results)} jobs and saved results to parsed_results.json")
```

## Custom Configuration

### Adjusting Similarity Threshold

```python
from job_parser import JobDescriptionParser, ParserConfig

# Create custom configuration with lower threshold for more matches
config = ParserConfig()
config.similarity_threshold = 0.5  # Lower threshold = more matches
config.max_text_length = 1500
config.enable_ner = True
config.enable_semantic = True

parser = JobDescriptionParser(config)

job_text = "Looking for a developer with Python and machine learning experience"

# Parse with custom config
result = parser.parse_job_description(job_text)
print(f"Skills found with threshold 0.5: {result.skills_required}")

# Compare with default threshold
default_parser = JobDescriptionParser()
default_result = default_parser.parse_job_description(job_text)
print(f"Skills found with default threshold: {default_result.skills_required}")
```

### Environment-Based Configuration

```python
import os
from job_parser import JobDescriptionParser, ParserConfig

# Set environment variables
os.environ['SIMILARITY_THRESHOLD'] = '0.6'
os.environ['MAX_TEXT_LENGTH'] = '1800'
os.environ['ENABLE_NER'] = 'true'
os.environ['ENABLE_SEMANTIC'] = 'true'

# Configuration will automatically use environment variables
config = ParserConfig()
parser = JobDescriptionParser(config)

print(f"Using threshold: {config.similarity_threshold}")
print(f"Max text length: {config.max_text_length}")
```

### Custom Skills Ontology

```python
import csv
from job_parser import JobDescriptionParser, ParserConfig

# Create custom skills ontology
custom_skills = [
    ('technical', 'Python'),
    ('technical', 'JavaScript'),
    ('technical', 'React'),
    ('technical', 'Vue.js'),
    ('tools', 'Docker'),
    ('tools', 'Kubernetes'),
    ('tools', 'AWS'),
    ('tools', 'Azure'),
    ('soft', 'Leadership'),
    ('soft', 'Communication'),
]

# Save to CSV
with open('custom_skills.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['category', 'skill'])
    writer.writerows(custom_skills)

# Use custom ontology
config = ParserConfig()
config.ontology_path = 'custom_skills.csv'

parser = JobDescriptionParser(config)
result = parser.parse_job_description("React developer with AWS experience")
print(f"Skills found with custom ontology: {result.skills_required}")
```

## CLI Usage

### Basic CLI Commands

```bash
# Parse single job description file
python main.py job.json

# Batch process with summary output
python main.py jobs.json -f summary

# JSON output with pretty printing
python main.py jobs.json -f json --pretty -o results.json

# Custom configuration
python main.py jobs.json --threshold 0.6 --max-length 1500

# Performance optimized processing
python main.py jobs.json --batch-size 16 --max-memory 256 --performance-stats
```

### CLI with Shell Scripts

```bash
#!/bin/bash
# process_jobs.sh

echo "Processing job descriptions..."

# Process different job categories
python main.py tech_jobs.json -f json -o tech_results.json
python main.py marketing_jobs.json -f json -o marketing_results.json
python main.py sales_jobs.json -f json -o sales_results.json

echo "Generating summary reports..."

# Generate summary reports
python main.py tech_jobs.json -f summary > tech_summary.txt
python main.py marketing_jobs.json -f summary > marketing_summary.txt
python main.py sales_jobs.json -f summary > sales_summary.txt

echo "Processing complete!"
```

### PowerShell Script Example

```powershell
# process_jobs.ps1

Write-Host "Starting job description processing..." -ForegroundColor Green

# Define job files
$jobFiles = @(
    "data/software_jobs.json",
    "data/data_science_jobs.json",
    "data/devops_jobs.json"
)

# Process each file
foreach ($file in $jobFiles) {
    $basename = [System.IO.Path]::GetFileNameWithoutExtension($file)
    $outputFile = "results/${basename}_results.json"
    
    Write-Host "Processing $file..." -ForegroundColor Yellow
    
    python main.py $file -f json --pretty -o $outputFile --performance-stats
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Successfully processed $file" -ForegroundColor Green
    } else {
        Write-Host "✗ Failed to process $file" -ForegroundColor Red
    }
}

Write-Host "Processing complete!" -ForegroundColor Green
```

## Performance Optimization

### Memory-Efficient Processing

```python
from job_parser import JobDescriptionParser
from job_parser.performance import MemoryManager

# Large dataset processing
large_job_list = ["job description text..."] * 1000

# Estimate memory usage
memory_estimate = MemoryManager.estimate_memory_usage(large_job_list)
print(f"Estimated memory usage: {memory_estimate['total_estimated_mb']:.1f} MB")

# Optimize for memory constraints
parser = JobDescriptionParser()

if memory_estimate['total_estimated_mb'] > 512:
    # Process in smaller chunks
    chunk_size = 50
    all_results = []
    
    for i in range(0, len(large_job_list), chunk_size):
        chunk = large_job_list[i:i+chunk_size]
        print(f"Processing chunk {i//chunk_size + 1}...")
        
        # Optimize for this chunk
        parser.optimize_for_batch_processing(len(chunk), max_memory_mb=256)
        
        # Process chunk
        chunk_results = parser.parse_job_descriptions_batch(chunk)
        all_results.extend(chunk_results)
        
        # Clear caches between chunks
        parser.clear_caches()
    
    print(f"Processed {len(all_results)} jobs in chunks")
else:
    # Process all at once
    parser.optimize_for_batch_processing(len(large_job_list))
    all_results = parser.parse_job_descriptions_batch(large_job_list)
```

### Performance Monitoring

```python
import time
from job_parser import JobDescriptionParser

parser = JobDescriptionParser()

# Warm up the parser
parser.parse_job_description("Test job description")

# Benchmark performance
job_texts = ["Sample job description..."] * 20

start_time = time.time()
results = parser.parse_job_descriptions_batch(job_texts)
end_time = time.time()

processing_time = end_time - start_time
throughput = len(job_texts) / processing_time

print(f"Performance Metrics:")
print(f"  Jobs processed: {len(results)}")
print(f"  Total time: {processing_time:.2f} seconds")
print(f"  Throughput: {throughput:.1f} jobs/second")

# Get detailed performance stats
stats = parser.get_performance_stats()
print(f"  Cache hit rate: {stats['components']['semantic_matcher']['cache_hit_rate']:.3f}")
print(f"  Model loaded: {stats['components']['semantic_matcher']['model_loaded']}")
```

## Integration Examples

### Flask Web API

```python
from flask import Flask, request, jsonify
from job_parser import JobDescriptionParser
import logging

app = Flask(__name__)
parser = JobDescriptionParser()

# Optimize for web requests
parser.optimize_for_batch_processing(expected_batch_size=1, max_memory_mb=256)

@app.route('/parse', methods=['POST'])
def parse_job():
    try:
        data = request.get_json()
        job_text = data.get('job_description', '')
        
        if not job_text:
            return jsonify({'error': 'job_description is required'}), 400
        
        # Parse job description
        result = parser.parse_job_description(job_text)
        
        # Return structured response
        return jsonify({
            'success': True,
            'data': {
                'skills_required': result.skills_required,
                'experience_level': result.experience_level,
                'tools_mentioned': result.tools_mentioned,
                'confidence_scores': result.confidence_scores,
                'categories': result.categories
            }
        })
    
    except Exception as e:
        logging.error(f"Parsing error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/parse/batch', methods=['POST'])
def parse_jobs_batch():
    try:
        data = request.get_json()
        job_texts = data.get('job_descriptions', [])
        
        if not job_texts:
            return jsonify({'error': 'job_descriptions array is required'}), 400
        
        # Process batch
        results = parser.parse_job_descriptions_batch(job_texts)
        
        # Format response
        formatted_results = []
        for i, result in enumerate(results):
            formatted_results.append({
                'index': i,
                'skills_required': result.skills_required,
                'experience_level': result.experience_level,
                'tools_mentioned': result.tools_mentioned,
                'confidence_scores': result.confidence_scores
            })
        
        return jsonify({
            'success': True,
            'data': formatted_results,
            'summary': {
                'total_jobs': len(results),
                'total_skills': sum(len(r.skills_required) for r in results)
            }
        })
    
    except Exception as e:
        logging.error(f"Batch parsing error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
```

### Pandas Integration

```python
import pandas as pd
from job_parser import JobDescriptionParser

# Load job data
df = pd.read_csv('job_postings.csv')

# Initialize parser
parser = JobDescriptionParser()
parser.optimize_for_batch_processing(len(df))

# Parse all job descriptions
print("Parsing job descriptions...")
results = parser.parse_job_descriptions_batch(df['description'].tolist())

# Add results to dataframe
df['skills_extracted'] = [r.skills_required for r in results]
df['experience_level'] = [r.experience_level for r in results]
df['tools_mentioned'] = [r.tools_mentioned for r in results]
df['skill_count'] = [len(r.skills_required) for r in results]

# Analysis
print("\nDataset Analysis:")
print(f"Total jobs: {len(df)}")
print(f"Average skills per job: {df['skill_count'].mean():.1f}")

print("\nExperience Level Distribution:")
print(df['experience_level'].value_counts())

print("\nTop Companies by Average Skills:")
company_skills = df.groupby('company')['skill_count'].mean().sort_values(ascending=False)
print(company_skills.head(10))

# Save enhanced dataset
df.to_csv('job_postings_enhanced.csv', index=False)
print("\nEnhanced dataset saved to job_postings_enhanced.csv")
```

### Async Processing

```python
import asyncio
import aiofiles
import json
from concurrent.futures import ThreadPoolExecutor
from job_parser import JobDescriptionParser

class AsyncJobParser:
    def __init__(self):
        self.parser = JobDescriptionParser()
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    async def parse_job_async(self, job_text):
        """Parse a single job description asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor, 
            self.parser.parse_job_description, 
            job_text
        )
    
    async def parse_jobs_from_file(self, filename):
        """Load and parse jobs from a JSON file asynchronously."""
        # Load file asynchronously
        async with aiofiles.open(filename, 'r') as f:
            content = await f.read()
            jobs_data = json.loads(content)
        
        # Parse jobs concurrently
        tasks = []
        for job in jobs_data:
            task = self.parse_job_async(job['description'])
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        # Combine with metadata
        enhanced_results = []
        for job, result in zip(jobs_data, results):
            enhanced_results.append({
                'job_id': job.get('id'),
                'title': job.get('title'),
                'skills': result.skills_required,
                'experience_level': result.experience_level,
                'confidence_scores': result.confidence_scores
            })
        
        return enhanced_results

# Usage
async def main():
    parser = AsyncJobParser()
    results = await parser.parse_jobs_from_file('jobs.json')
    
    print(f"Processed {len(results)} jobs asynchronously")
    for result in results[:3]:  # Show first 3
        print(f"Job: {result['title']}")
        print(f"Skills: {', '.join(result['skills'][:5])}")
        print()

# Run async processing
asyncio.run(main())
```

## Error Handling

### Comprehensive Error Handling

```python
from job_parser import JobDescriptionParser
from job_parser.exceptions import (
    JobParserError, 
    InputValidationError, 
    ModelLoadError, 
    OntologyLoadError
)
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def robust_job_parsing(job_texts):
    """Parse jobs with comprehensive error handling."""
    results = []
    errors = []
    
    try:
        # Initialize parser
        parser = JobDescriptionParser()
        
    except ModelLoadError as e:
        logger.error(f"Failed to load models: {e}")
        return [], [f"Model loading failed: {e}"]
    
    except OntologyLoadError as e:
        logger.error(f"Failed to load ontology: {e}")
        return [], [f"Ontology loading failed: {e}"]
    
    # Process each job with individual error handling
    for i, job_text in enumerate(job_texts):
        try:
            # Validate input
            if not job_text or not job_text.strip():
                errors.append(f"Job {i+1}: Empty job description")
                results.append(None)
                continue
            
            if len(job_text.split()) > 2000:
                logger.warning(f"Job {i+1}: Text too long, truncating")
                job_text = ' '.join(job_text.split()[:2000])
            
            # Parse job
            result = parser.parse_job_description(job_text)
            results.append(result)
            
            # Validate results
            if not result.skills_required:
                logger.warning(f"Job {i+1}: No skills extracted")
            
        except InputValidationError as e:
            error_msg = f"Job {i+1}: Invalid input - {e}"
            logger.error(error_msg)
            errors.append(error_msg)
            results.append(None)
        
        except JobParserError as e:
            error_msg = f"Job {i+1}: Parsing failed - {e}"
            logger.error(error_msg)
            errors.append(error_msg)
            results.append(None)
        
        except Exception as e:
            error_msg = f"Job {i+1}: Unexpected error - {e}"
            logger.error(error_msg)
            errors.append(error_msg)
            results.append(None)
    
    # Summary
    successful = sum(1 for r in results if r is not None)
    logger.info(f"Processing complete: {successful}/{len(job_texts)} successful")
    
    return results, errors

# Example usage
job_texts = [
    "Python developer with Django experience",
    "",  # Empty job description
    "A" * 10000,  # Very long job description
    "React developer with TypeScript",
    None,  # Invalid input
]

results, errors = robust_job_parsing(job_texts)

print(f"Results: {len([r for r in results if r])}")
print(f"Errors: {len(errors)}")
for error in errors:
    print(f"  - {error}")
```

### Retry Logic

```python
import time
from job_parser import JobDescriptionParser
from job_parser.exceptions import JobParserError

def parse_with_retry(parser, job_text, max_retries=3, delay=1.0):
    """Parse job with retry logic for transient failures."""
    
    for attempt in range(max_retries):
        try:
            return parser.parse_job_description(job_text)
        
        except JobParserError as e:
            if attempt == max_retries - 1:
                # Last attempt failed
                raise e
            
            print(f"Attempt {attempt + 1} failed: {e}")
            print(f"Retrying in {delay} seconds...")
            time.sleep(delay)
            
            # Exponential backoff
            delay *= 2
    
    return None

# Usage
parser = JobDescriptionParser()
job_text = "Software engineer with Python experience"

try:
    result = parse_with_retry(parser, job_text)
    print(f"Successfully parsed: {result.skills_required}")
except JobParserError as e:
    print(f"Failed after all retries: {e}")
```

These examples demonstrate the flexibility and power of the Job Description Parser across various use cases and integration scenarios.