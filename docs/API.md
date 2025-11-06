# API Reference

This document provides detailed API reference for the Job Description Parser.

## Core Classes

### JobDescriptionParser

The main parser class that orchestrates the entire parsing pipeline.

```python
from job_parser import JobDescriptionParser, ParserConfig
```

#### Constructor

```python
JobDescriptionParser(config: Optional[ParserConfig] = None)
```

**Parameters:**
- `config` (Optional[ParserConfig]): Custom configuration. Uses default if None.

**Raises:**
- `JobParserError`: If initialization fails

#### Methods

##### parse_job_description

```python
parse_job_description(job_desc: str) -> ParsedJobDescription
```

Parse a single job description into structured data.

**Parameters:**
- `job_desc` (str): Raw job description text

**Returns:**
- `ParsedJobDescription`: Structured parsing results

**Raises:**
- `InputValidationError`: If input is invalid
- `JobParserError`: If parsing fails

**Example:**
```python
parser = JobDescriptionParser()
result = parser.parse_job_description("Python developer with Django experience")
print(result.skills_required)  # ['Python', 'Django']
```

##### parse_job_descriptions_batch

```python
parse_job_descriptions_batch(job_descriptions: List[str]) -> List[ParsedJobDescription]
```

Process multiple job descriptions efficiently in batch.

**Parameters:**
- `job_descriptions` (List[str]): List of job description texts

**Returns:**
- `List[ParsedJobDescription]`: List of parsing results

**Example:**
```python
jobs = ["Python developer...", "React developer..."]
results = parser.parse_job_descriptions_batch(jobs)
```

##### optimize_for_batch_processing

```python
optimize_for_batch_processing(expected_batch_size: int, max_memory_mb: float = 512.0) -> None
```

Optimize parser for batch processing with performance enhancements.

**Parameters:**
- `expected_batch_size` (int): Expected number of jobs to process
- `max_memory_mb` (float): Maximum memory usage limit in MB

##### parse_job_description_to_json

```python
parse_job_description_to_json(job_desc: str) -> str
```

Parse job description and return JSON string output.

**Parameters:**
- `job_desc` (str): Raw job description text

**Returns:**
- `str`: JSON formatted parsing results

##### get_performance_stats

```python
get_performance_stats() -> Dict[str, Any]
```

Get comprehensive performance statistics.

**Returns:**
- `Dict[str, Any]`: Performance metrics and statistics

##### clear_caches

```python
clear_caches() -> None
```

Clear all performance caches to free memory.

### ParsedJobDescription

Data class containing structured parsing results.

#### Attributes

- `skills_required` (List[str]): List of identified skills
- `experience_level` (str): Inferred experience level
- `tools_mentioned` (List[str]): List of tools and technologies
- `confidence_scores` (Dict[str, float]): Confidence scores for each skill
- `categories` (Dict[str, List[str]]): Skills grouped by category
- `metadata` (Dict[str, Any]): Processing metadata

#### Methods

##### to_json

```python
to_json() -> Dict[str, Any]
```

Convert to JSON-serializable dictionary.

**Returns:**
- `Dict[str, Any]`: JSON-serializable representation

### ParserConfig

Configuration class for customizing parser behavior.

#### Attributes

- `ontology_path` (str): Path to skills ontology CSV file
- `similarity_threshold` (float): Minimum similarity for matches (0.0-1.0)
- `max_text_length` (int): Maximum input text length in words
- `enable_ner` (bool): Enable Named Entity Recognition
- `enable_semantic` (bool): Enable semantic matching
- `model_name` (str): Sentence-BERT model name
- `log_level` (str): Logging level
- `confidence_weighting` (Dict[str, float]): Weights for confidence sources

#### Methods

##### validate

```python
validate() -> None
```

Validate configuration parameters.

**Raises:**
- `ValueError`: If any parameter is invalid

## Component Interfaces

### SemanticMatcherInterface

Abstract interface for semantic matching components.

#### Methods

##### find_skill_matches

```python
find_skill_matches(sentences: List[str], ontology: Dict[str, List[str]], threshold: float) -> Dict[str, List[Tuple[str, float]]]
```

Find semantic matches between text and ontology skills.

### NERExtractorInterface

Abstract interface for Named Entity Recognition components.

#### Methods

##### extract_entities

```python
extract_entities(text: str) -> List[SkillMatch]
```

Extract named entities with confidence scores.

### SkillCategorizerInterface

Abstract interface for skill categorization components.

#### Methods

##### categorize_skills

```python
categorize_skills(skill_matches: List[SkillMatch], ontology: Dict[str, List[str]]) -> Dict[str, List[str]]
```

Categorize and group extracted skills.

## Data Classes

### SkillMatch

Represents a matched skill with metadata.

#### Attributes

- `skill` (str): The matched skill name
- `category` (str): Skill category
- `confidence` (float): Confidence score (0.0-1.0)
- `source` (str): Source of the match (semantic/ner)
- `context` (str): Context where skill was found

### JobDescriptionInput

Input validation class for job descriptions.

#### Attributes

- `text` (str): Job description text
- `max_length` (int): Maximum allowed length

#### Methods

##### validate

```python
validate() -> None
```

Validate input parameters.

## Exception Classes

### JobParserError

Base exception for parser-related errors.

### InputValidationError

Exception for input validation failures.

### ModelLoadError

Exception for model loading failures.

### OntologyLoadError

Exception for ontology loading failures.

## Utility Functions

### Performance Utilities

```python
from job_parser.performance import clear_all_caches, get_cached_model

# Clear all performance caches
clear_all_caches()

# Get cached model
model = get_cached_model("all-MiniLM-L6-v2")
```

### Configuration Utilities

```python
from job_parser.config import CONFIG

# Access global configuration
print(CONFIG.similarity_threshold)
```

## Usage Patterns

### Basic Usage

```python
from job_parser import JobDescriptionParser

parser = JobDescriptionParser()
result = parser.parse_job_description(job_text)
```

### Custom Configuration

```python
from job_parser import JobDescriptionParser, ParserConfig

config = ParserConfig()
config.similarity_threshold = 0.6
config.enable_ner = True

parser = JobDescriptionParser(config)
```

### Batch Processing

```python
parser = JobDescriptionParser()
parser.optimize_for_batch_processing(expected_batch_size=100)
results = parser.parse_job_descriptions_batch(job_texts)
```

### Performance Monitoring

```python
parser = JobDescriptionParser()
result = parser.parse_job_description(job_text)

# Get performance statistics
stats = parser.get_performance_stats()
print(f"Cache hit rate: {stats['components']['semantic_matcher']['cache_hit_rate']}")
```

### Error Handling

```python
from job_parser import JobDescriptionParser
from job_parser.exceptions import JobParserError, InputValidationError

parser = JobDescriptionParser()

try:
    result = parser.parse_job_description(job_text)
except InputValidationError as e:
    print(f"Invalid input: {e}")
except JobParserError as e:
    print(f"Parsing failed: {e}")
```