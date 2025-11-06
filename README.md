# Job Description Parser

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A powerful, AI-driven job description parser that extracts structured information from unstructured job postings. Built with advanced NLP techniques including semantic matching and named entity recognition (NER) to identify technical skills, experience levels, and tools mentioned in job descriptions.

## 🚀 Features

- **🧠 Dual Extraction Methods**: Combines semantic matching (Sentence-BERT) with Named Entity Recognition (spaCy)
- **⚡ High Performance**: Optimized batch processing with model caching and memory management
- **🎯 Accurate Skill Detection**: Identifies technical skills, soft skills, and tools with confidence scores
- **📊 Experience Level Inference**: Automatically determines job seniority (entry, mid, senior level)
- **🔧 Configurable**: Customizable similarity thresholds and processing parameters
- **📋 Multiple Output Formats**: JSON, summary, and structured data formats
- **🚀 CLI Interface**: Command-line tool for batch processing and automation
- **🔄 Extensible**: Modular architecture for easy customization and extension

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
- [CLI Usage](#cli-usage)
- [Configuration](#configuration)
- [Performance](#performance)
- [API Reference](#api-reference)
- [Contributing](#contributing)
- [License](#license)

## 🛠 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Install Dependencies

```bash
# Clone the repository
git clone https://github.com/yourusername/job-description-parser.git
cd job-description-parser

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

### Quick Install

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## 🚀 Quick Start

### Basic Usage

```python
from job_parser import JobDescriptionParser

# Initialize the parser
parser = JobDescriptionParser()

# Parse a job description
job_text = """
Senior Python Developer - Remote
We are looking for an experienced Python developer with 5+ years of experience.
Requirements: Python, Django, PostgreSQL, AWS, Docker, React.js
"""

result = parser.parse_job_description(job_text)

# Access results
print(f"Skills found: {result.skills_required}")
print(f"Experience level: {result.experience_level}")
print(f"Tools mentioned: {result.tools_mentioned}")
```

### Command Line Usage

```bash
# Parse a single job description file
python main.py job_description.json

# Batch process multiple job descriptions
python main.py jobs.json --batch -o results.json

# Get summary output
python main.py jobs.json -f summary
```

## 📖 Usage Examples

### 1. Single Job Processing

```python
from job_parser import JobDescriptionParser

parser = JobDescriptionParser()

job_description = """
Data Scientist Position
Looking for a data scientist with machine learning experience.
Required skills: Python, scikit-learn, pandas, SQL, TensorFlow
3-5 years of experience preferred.
"""

result = parser.parse_job_description(job_description)

print("Extracted Information:")
print(f"📊 Skills: {', '.join(result.skills_required)}")
print(f"🎯 Experience Level: {result.experience_level}")
print(f"🛠 Tools: {', '.join(result.tools_mentioned)}")
print(f"📈 Confidence Scores: {result.confidence_scores}")
```

### 2. Batch Processing

```python
from job_parser import JobDescriptionParser

parser = JobDescriptionParser()

# List of job descriptions
job_descriptions = [
    "Python developer with Django experience...",
    "React frontend developer with TypeScript...",
    "DevOps engineer with AWS and Docker..."
]

# Optimize for batch processing
parser.optimize_for_batch_processing(expected_batch_size=len(job_descriptions))

# Process all jobs
results = parser.parse_job_descriptions_batch(job_descriptions)

for i, result in enumerate(results):
    print(f"Job {i+1}: {len(result.skills_required)} skills found")
```

### 3. Custom Configuration

```python
from job_parser import JobDescriptionParser, ParserConfig

# Create custom configuration
config = ParserConfig()
config.similarity_threshold = 0.6  # Lower threshold for more matches
config.max_text_length = 1500
config.enable_ner = True
config.enable_semantic = True

# Initialize parser with custom config
parser = JobDescriptionParser(config)

result = parser.parse_job_description(job_text)
```

### 4. JSON Output

```python
from job_parser import JobDescriptionParser
import json

parser = JobDescriptionParser()

# Get structured JSON output
json_result = parser.parse_job_description_to_json(job_text)

# Parse and pretty print
parsed_result = json.loads(json_result)
print(json.dumps(parsed_result, indent=2))
```

## 💻 CLI Usage

The parser includes a comprehensive command-line interface for batch processing and automation.

### Basic Commands

```bash
# Parse single job description
python main.py input.json

# Batch process with JSON output
python main.py jobs.json -f json --pretty -o results.json

# Generate summary report
python main.py jobs.json -f summary

# Performance optimized processing
python main.py jobs.json --batch-size 16 --max-memory 512
```

### CLI Options

| Option | Description | Example |
|--------|-------------|---------|
| `-f, --format` | Output format (json/summary) | `-f summary` |
| `-o, --output` | Output file path | `-o results.json` |
| `--pretty` | Pretty-print JSON output | `--pretty` |
| `--batch` | Enable batch processing | `--batch` |
| `--threshold` | Similarity threshold (0.0-1.0) | `--threshold 0.6` |
| `--batch-size` | Processing batch size | `--batch-size 32` |
| `--max-memory` | Memory limit in MB | `--max-memory 256` |
| `--performance-stats` | Show performance statistics | `--performance-stats` |
| `-v, --verbose` | Enable verbose logging | `-v` |

### Input Format

The CLI expects JSON input files with job descriptions:

```json
[
  {
    "id": "job_001",
    "title": "Senior Python Developer",
    "company": "TechCorp",
    "description": "We are seeking a Senior Python Developer..."
  }
]
```

## ⚙️ Configuration

### Environment Variables

```bash
# Set custom configuration via environment variables
export SIMILARITY_THRESHOLD=0.7
export MAX_TEXT_LENGTH=2000
export ENABLE_NER=true
export ENABLE_SEMANTIC=true
export SENTENCE_BERT_MODEL=all-MiniLM-L6-v2
```

### Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `similarity_threshold` | 0.7 | Minimum similarity for semantic matches |
| `max_text_length` | 2000 | Maximum input text length (words) |
| `enable_ner` | true | Enable Named Entity Recognition |
| `enable_semantic` | true | Enable semantic matching |
| `model_name` | all-MiniLM-L6-v2 | Sentence-BERT model to use |

### Skills Ontology

The parser uses a customizable skills ontology (`skills_ontology.csv`):

```csv
category,skill
technical,Python
technical,JavaScript
technical,React
tools,Docker
tools,AWS
tools,PostgreSQL
soft,Communication
soft,Problem Solving
```

## 🚀 Performance

### Benchmarks

| Metric | Performance |
|--------|-------------|
| Single Job Processing | ~1.2 seconds |
| Batch Processing | 1.3-3.8 jobs/second |
| Embedding Generation | Up to 227 texts/second |
| Memory Usage | Optimized with chunking |
| Cache Hit Rate | Up to 100% with precomputed embeddings |

### Optimization Features

- **Model Caching**: Reuse loaded models across requests
- **Precomputed Embeddings**: Cache skill embeddings for faster matching
- **Memory Management**: Automatic chunking and garbage collection
- **Batch Processing**: Optimized vectorized operations
- **Adaptive Sizing**: Dynamic batch size adjustment

### Performance Tips

1. **Use Batch Processing**: Process multiple jobs together for better throughput
2. **Enable Caching**: Keep the same parser instance for multiple requests
3. **Optimize Memory**: Adjust `max_memory_mb` based on available RAM
4. **Precompute Embeddings**: Call `optimize_for_batch_processing()` before batch jobs

## 📚 API Reference

### JobDescriptionParser

Main parser class for extracting structured information from job descriptions.

#### Methods

##### `__init__(config: Optional[ParserConfig] = None)`
Initialize the parser with optional configuration.

##### `parse_job_description(job_desc: str) -> ParsedJobDescription`
Parse a single job description and return structured results.

**Parameters:**
- `job_desc` (str): Raw job description text

**Returns:**
- `ParsedJobDescription`: Structured parsing results

##### `parse_job_descriptions_batch(job_descriptions: List[str]) -> List[ParsedJobDescription]`
Process multiple job descriptions efficiently in batch.

**Parameters:**
- `job_descriptions` (List[str]): List of job description texts

**Returns:**
- `List[ParsedJobDescription]`: List of parsing results

##### `optimize_for_batch_processing(expected_batch_size: int, max_memory_mb: float = 512.0)`
Optimize parser for batch processing with performance enhancements.

##### `get_performance_stats() -> Dict[str, Any]`
Get comprehensive performance statistics and metrics.

### ParsedJobDescription

Data class containing structured parsing results.

#### Attributes

- `skills_required` (List[str]): List of identified skills
- `experience_level` (str): Inferred experience level (entry/mid/senior)
- `tools_mentioned` (List[str]): List of tools and technologies
- `confidence_scores` (Dict[str, float]): Confidence scores for each skill
- `categories` (Dict[str, List[str]]): Skills grouped by category
- `metadata` (Dict[str, Any]): Processing metadata and statistics

### ParserConfig

Configuration class for customizing parser behavior.

#### Attributes

- `similarity_threshold` (float): Minimum similarity for matches (0.0-1.0)
- `max_text_length` (int): Maximum input text length in words
- `enable_ner` (bool): Enable Named Entity Recognition
- `enable_semantic` (bool): Enable semantic matching
- `model_name` (str): Sentence-BERT model name

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Run all tests
python test_parser.py

# Run specific CLI tests
python main.py test_job_descriptions.json -f summary

# Test with custom configuration
python -c "
from job_parser import JobDescriptionParser
parser = JobDescriptionParser()
result = parser.parse_job_description('Python developer with React experience')
print(f'Skills: {result.skills_required}')
"
```

### Test Coverage

The test suite covers:
- ✅ Single job parsing
- ✅ Batch processing
- ✅ Performance optimization
- ✅ Custom configuration
- ✅ JSON output validation
- ✅ Error handling
- ✅ CLI interface

## 🏗 Architecture

### System Components

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Text Input    │───▶│  JobDescription  │───▶│ Structured Data │
│                 │    │     Parser       │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │   Components     │
                    └──────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Semantic   │    │     NER      │    │    Skill     │
│   Matching   │    │  Extraction  │    │Categorization│
│              │    │              │    │              │
└──────────────┘    └──────────────┘    └──────────────┘
```

### Key Components

1. **Text Preprocessor**: Cleans and tokenizes input text
2. **Semantic Matcher**: Uses Sentence-BERT for similarity matching
3. **NER Extractor**: Identifies entities using spaCy with custom patterns
4. **Skill Categorizer**: Groups and scores extracted skills
5. **Performance Optimizer**: Handles caching and batch processing


### Development Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/job-description-parser.git
cd job-description-parser

# Create development environment
python -m venv venv
source venv/bin/activate

# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run tests
python test_parser.py
```

### Code Style

This project uses:
- **Black** for code formatting
- **isort** for import sorting
- **flake8** for linting
- **mypy** for type checking

## 🙏 Acknowledgments

- [Sentence-Transformers](https://www.sbert.net/) for semantic similarity
- [spaCy](https://spacy.io/) for natural language processing
- [scikit-learn](https://scikit-learn.org/) for machine learning utilities

## 🗺 Roadmap

- [ ] Web API interface
- [ ] Support for multiple languages
- [ ] Integration with job boards
- [ ] Advanced skill taxonomy
- [ ] Machine learning model fine-tuning
- [ ] Real-time processing capabilities

---