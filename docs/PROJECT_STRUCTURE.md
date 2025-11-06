# Project Structure

This document describes the organization and structure of the Job Description Parser project.

## Directory Structure

```
job-description-parser/
├── .github/                          # GitHub configuration
│   ├── workflows/                    # CI/CD workflows
│   │   ├── ci.yml                   # Continuous integration
│   │   └── release.yml              # Release automation
│   ├── ISSUE_TEMPLATE/              # Issue templates
│   │   ├── bug_report.md
│   │   ├── feature_request.md
│   │   └── question.md
│   └── pull_request_template.md     # PR template
├── docs/                            # Documentation
│   ├── API.md                       # API reference
│   ├── EXAMPLES.md                  # Usage examples
│   └── PROJECT_STRUCTURE.md         # This file
├── job_parser/                      # Main package
│   ├── __init__.py                  # Package initialization
│   ├── config.py                    # Configuration management
│   ├── exceptions.py                # Custom exceptions
│   ├── interfaces.py                # Abstract interfaces
│   ├── logging_config.py            # Logging configuration
│   ├── ontology.py                  # Skills ontology loader
│   ├── parser.py                    # Main parser class
│   ├── performance.py               # Performance optimizations
│   ├── preprocessing.py             # Text preprocessing
│   ├── semantic_matching.py         # Semantic similarity matching
│   ├── ner_extraction.py            # Named entity recognition
│   └── skill_categorization.py      # Skill categorization
├── tests/                           # Test files (optional)
│   ├── __init__.py
│   ├── test_parser.py
│   ├── test_semantic_matching.py
│   └── test_ner_extraction.py
├── .gitignore                       # Git ignore rules
├── .pre-commit-config.yaml          # Pre-commit hooks
├── CHANGELOG.md                     # Version history
├── CONTRIBUTING.md                  # Contribution guidelines
├── LICENSE                          # MIT license
├── README.md                        # Main documentation
├── install.sh                       # Linux/Mac installation script
├── install.bat                      # Windows installation script
├── main.py                          # CLI interface
├── pyproject.toml                   # Modern Python packaging
├── requirements.txt                 # Production dependencies
├── requirements-dev.txt             # Development dependencies
├── setup.py                         # Package setup
├── skills_ontology.csv              # Default skills ontology
├── test_job_descriptions.json       # Test data
└── test_parser.py                   # Comprehensive test suite
```

## Core Components

### 1. Main Package (`job_parser/`)

#### Core Classes

- **`JobDescriptionParser`** (`parser.py`): Main orchestrator class
  - Coordinates all components
  - Provides high-level API
  - Handles error management
  - Implements performance optimizations

- **`SemanticMatcher`** (`semantic_matching.py`): Semantic similarity engine
  - Uses Sentence-BERT embeddings
  - Implements cosine similarity matching
  - Provides caching and batch processing

- **`NERExtractor`** (`ner_extraction.py`): Named entity recognition
  - Uses spaCy NER models
  - Custom patterns for technical skills
  - Entity merging and deduplication

- **`SkillCategorizer`** (`skill_categorization.py`): Skill classification
  - Groups skills by category
  - Computes confidence scores
  - Infers experience levels

#### Supporting Modules

- **`interfaces.py`**: Abstract base classes and data structures
- **`config.py`**: Configuration management with environment variables
- **`exceptions.py`**: Custom exception hierarchy
- **`performance.py`**: Performance optimization utilities
- **`preprocessing.py`**: Text cleaning and tokenization
- **`ontology.py`**: Skills ontology loading and management
- **`logging_config.py`**: Centralized logging configuration

### 2. CLI Interface (`main.py`)

Command-line interface providing:
- Single job processing
- Batch processing
- Multiple output formats
- Performance optimization options
- Comprehensive argument parsing

### 3. Configuration Files

#### Package Configuration
- **`pyproject.toml`**: Modern Python packaging configuration
- **`setup.py`**: Traditional setup script for compatibility
- **`requirements.txt`**: Production dependencies
- **`requirements-dev.txt`**: Development dependencies

#### Code Quality
- **`.pre-commit-config.yaml`**: Pre-commit hooks for code quality
- **`.gitignore`**: Git ignore patterns
- **`CONTRIBUTING.md`**: Development guidelines

#### CI/CD
- **`.github/workflows/ci.yml`**: Continuous integration
- **`.github/workflows/release.yml`**: Release automation

### 4. Data Files

- **`skills_ontology.csv`**: Default skills taxonomy
- **`test_job_descriptions.json`**: Sample job descriptions for testing

### 5. Documentation

- **`README.md`**: Main project documentation
- **`docs/API.md`**: Detailed API reference
- **`docs/EXAMPLES.md`**: Comprehensive usage examples
- **`CHANGELOG.md`**: Version history and changes

## Architecture Overview

### Data Flow

```
Input Text → Preprocessing → Parallel Processing → Skill Categorization → Output
                                    ↓
                            ┌─────────────────┐
                            │ Semantic        │
                            │ Matching        │
                            └─────────────────┘
                                    ↓
                            ┌─────────────────┐
                            │ Named Entity    │
                            │ Recognition     │
                            └─────────────────┘
                                    ↓
                            ┌─────────────────┐
                            │ Result Merging  │
                            │ & Scoring       │
                            └─────────────────┘
```

### Component Interactions

1. **Parser** orchestrates the entire pipeline
2. **Preprocessor** cleans and tokenizes input text
3. **SemanticMatcher** and **NERExtractor** work in parallel
4. **SkillCategorizer** merges and scores results
5. **Performance** module provides caching and optimization

### Design Patterns

- **Strategy Pattern**: Different extraction methods (semantic, NER)
- **Factory Pattern**: Component initialization
- **Observer Pattern**: Logging and monitoring
- **Singleton Pattern**: Model caching
- **Template Method**: Processing pipeline

## Development Workflow

### 1. Setup Development Environment

```bash
# Clone repository
git clone https://github.com/yourusername/job-description-parser.git
cd job-description-parser

# Install development dependencies
pip install -r requirements-dev.txt
python -m spacy download en_core_web_sm

# Install pre-commit hooks
pre-commit install
```

### 2. Code Organization Guidelines

#### File Naming
- Use snake_case for Python files
- Use descriptive names that indicate purpose
- Group related functionality in modules

#### Class Organization
- One main class per file
- Related helper classes in the same file
- Abstract interfaces in `interfaces.py`

#### Import Organization
- Standard library imports first
- Third-party imports second
- Local imports last
- Use absolute imports for clarity

### 3. Testing Structure

#### Test Organization
```
tests/
├── unit/                    # Unit tests
│   ├── test_parser.py
│   ├── test_semantic_matching.py
│   └── test_ner_extraction.py
├── integration/             # Integration tests
│   ├── test_full_pipeline.py
│   └── test_cli.py
└── fixtures/                # Test data
    ├── sample_jobs.json
    └── expected_results.json
```

#### Test Categories
- **Unit Tests**: Test individual components
- **Integration Tests**: Test component interactions
- **Performance Tests**: Benchmark and regression tests
- **CLI Tests**: Command-line interface tests

### 4. Performance Considerations

#### Optimization Areas
- **Model Caching**: Reuse loaded models
- **Embedding Caching**: Cache computed embeddings
- **Batch Processing**: Vectorized operations
- **Memory Management**: Chunking and garbage collection

#### Monitoring
- Processing time per job
- Memory usage patterns
- Cache hit rates
- Throughput metrics

## Extension Points

### Adding New Components

1. **Create Interface**: Define abstract base class
2. **Implement Component**: Create concrete implementation
3. **Add Tests**: Write comprehensive tests
4. **Update Parser**: Integrate with main pipeline
5. **Document**: Add API documentation and examples

### Custom Skill Extraction Methods

```python
from job_parser.interfaces import SkillExtractorInterface

class CustomExtractor(SkillExtractorInterface):
    def extract_skills(self, text: str) -> List[SkillMatch]:
        # Your implementation here
        pass
```

### Custom Output Formats

```python
from job_parser.interfaces import OutputFormatterInterface

class CustomFormatter(OutputFormatterInterface):
    def format_results(self, result: ParsedJobDescription) -> str:
        # Your implementation here
        pass
```

## Deployment Considerations

### Package Distribution
- PyPI package for easy installation
- Docker container for containerized deployment
- GitHub releases with pre-built binaries

### Configuration Management
- Environment variables for production settings
- Configuration files for complex setups
- Runtime configuration updates

### Monitoring and Logging
- Structured logging for production
- Performance metrics collection
- Error tracking and alerting

## Future Enhancements

### Planned Features
- Web API interface
- Multi-language support
- Advanced skill taxonomy
- Real-time processing
- Cloud deployment options

### Architecture Improvements
- Plugin system for extensibility
- Microservices architecture
- Event-driven processing
- Distributed computing support