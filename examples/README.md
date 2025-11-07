# Examples

This directory contains practical examples demonstrating how to use the Resume-Job Matcher System.

## 📁 Directory Structure

```
examples/
├── basic_usage.py          # Simple resume-job comparison
├── batch_processing.py     # Processing multiple resumes
├── custom_configuration.py # Advanced configuration options
├── api_integration.py      # API integration examples
├── performance_tuning.py   # Performance optimization
├── sample_data/           # Sample resumes and job descriptions
│   ├── resumes/
│   └── jobs/
└── notebooks/             # Jupyter notebooks with detailed examples
    ├── getting_started.ipynb
    ├── advanced_features.ipynb
    └── model_training.ipynb
```

## 🚀 Quick Examples

### Basic Usage
```python
# See basic_usage.py for complete example
from resume_parser import ResumeParser
from job_parser import JobDescriptionParser
from resume_parser.similarity_engine import SimilarityEngine

# Initialize components
resume_parser = ResumeParser()
job_parser = JobDescriptionParser()
similarity_engine = SimilarityEngine()

# Parse and compare
resume = resume_parser.parse_resume("resume.pdf")
job = job_parser.parse_job_description(job_text)
result = similarity_engine.calculate_comprehensive_similarity(
    resume=resume, job_description=job,
    resume_text=resume_text, job_text=job_text
)

print(f"Match Score: {result.overall_score:.1f}%")
```

### Batch Processing
```python
# See batch_processing.py for complete example
resumes = ["resume1.pdf", "resume2.pdf", "resume3.pdf"]
results = []

for resume_file in resumes:
    resume = resume_parser.parse_resume(resume_file)
    result = similarity_engine.calculate_comprehensive_similarity(...)
    results.append((resume_file, result.overall_score))

# Sort by match score
results.sort(key=lambda x: x[1], reverse=True)
```

## 📊 Sample Data

The `sample_data/` directory contains:
- **Sample Resumes**: Various formats and experience levels
- **Job Descriptions**: Different roles and industries
- **Expected Results**: Benchmark results for validation

## 📓 Jupyter Notebooks

Interactive notebooks with step-by-step tutorials:
- **Getting Started**: Basic usage and concepts
- **Advanced Features**: Custom configurations and optimization
- **Model Training**: Fine-tuning SBERT on your data

## 🔧 Running Examples

```bash
# Run basic example
python examples/basic_usage.py

# Run batch processing example
python examples/batch_processing.py

# Start Jupyter notebook
jupyter notebook examples/notebooks/
```

## 💡 Use Cases

Each example demonstrates specific use cases:
- **HR Screening**: Automated candidate screening
- **Job Matching**: Finding best-fit positions
- **Resume Optimization**: Improving resume effectiveness
- **Skill Gap Analysis**: Identifying training needs
- **Performance Benchmarking**: System performance testing