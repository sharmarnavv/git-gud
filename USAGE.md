# Resume-Job Matcher - Usage Guide

## Quick Start

### 1. Setup

Ensure you have Python 3.8+ and install dependencies:

```bash
pip install -r requirements.txt
```

### 2. Prepare Job Description

Create a file named `job_description.txt` in the project root with the job posting:

```txt
Senior Python Developer

Requirements:
- 5+ years of Python development experience
- Experience with Django or Flask
- AWS cloud platform knowledge
- Docker and Kubernetes experience
- Strong problem-solving skills
...
```

### 3. Run Analysis

```bash
python main.py <path_to_your_resume.pdf>
```

**Examples:**
```bash
# Basic usage
python main.py resume.pdf

# With custom job description file
python main.py resume.pdf --job custom_job.txt

# Save detailed results to JSON
python main.py resume.pdf --output results.json

# Skip suggestions for faster analysis
python main.py resume.pdf --no-suggestions
```

## What You Get

### 1. Overall Match Score (0-100%)
- 🟢 80-100%: Excellent Match
- 🟡 60-79%: Good Match  
- 🟠 40-59%: Moderate Match
- 🔴 0-39%: Weak Match

### 2. Component Breakdown
- **Skills**: Technical and soft skills alignment
- **Experience**: Years and relevance
- **Education**: Degree and field match
- **Hybrid**: Semantic similarity (TF-IDF + SBERT)

### 3. Skills Analysis
- ✅ Matched skills you have
- ❌ Missing skills to acquire
- Match rate percentage

### 4. Experience Analysis
- Total years of experience
- Position history
- Required level comparison

### 5. Education & Certifications
- Degrees and institutions
- Certifications earned
- Graduation dates and GPAs

### 6. Quick Wins
- Easy, high-impact improvements
- Immediate actions you can take
- Low effort, high return suggestions

### 7. Top Priority Improvements
- Ranked by impact and feasibility
- Specific action items
- Rationale for each suggestion

### 8. Improvement Potential
- Current score
- Projected score after improvements
- Total suggestions generated

## Command-Line Options

```
usage: main.py [-h] [--job JOB] [-o OUTPUT] [--no-suggestions] resume

positional arguments:
  resume                Path to resume file (PDF, DOCX, or TXT)

optional arguments:
  -h, --help            show this help message and exit
  --job JOB             Path to job description file (default: job_description.txt)
  -o OUTPUT, --output OUTPUT
                        Save detailed results to JSON file
  --no-suggestions      Skip generating improvement suggestions (faster)
```

## Supported Resume Formats

- **PDF** (`.pdf`) - Recommended
- **Word Document** (`.docx`, `.doc`)
- **Text File** (`.txt`)

## Tips for Best Results

1. **Use a well-formatted resume** with clear sections
2. **List skills explicitly** in a dedicated section
3. **Include complete job posting** in job_description.txt
4. **Update resume** based on suggestions and re-run
5. **Focus on quick wins** first for immediate impact

## Example Output

```
================================================================================
                  🚀 RESUME-JOB MATCHER - AI-POWERED ANALYSIS
================================================================================

📄 Resume: resume.pdf
💼 Job Description: job_description.txt

🔄 Loading AI models and initializing parsers...
🔄 Parsing resume...
🔄 Parsing job description...
🔄 Calculating comprehensive similarity score...

================================================================================
                        📊 MATCH ANALYSIS RESULTS
================================================================================

🟢 Overall Match Score: 75.3% - GOOD MATCH

📈 COMPONENT BREAKDOWN
────────────────────────────────────────────────────────────────────────────
Skills              ████████████████░░░░  78.5%
Experience          ███████████████░░░░░  72.0%
Education           ████████████████████  95.0%
Hybrid              ███████████████░░░░░  76.2%

🛠️  SKILLS ANALYSIS
────────────────────────────────────────────────────────────────────────────
Skills Match Rate: 70.0% (14/20)

✅ Matched Skills (14):
    1. Python
    2. Django
    3. PostgreSQL
    4. Docker
   ...

❌ Missing Skills (6):
    1. Kubernetes
    2. AWS
    3. Redis
   ...

🚀 QUICK WINS (Easy & High Impact)
────────────────────────────────────────────────────────────────────────────

1. Better highlight your existing skills
   Impact: 60% | Effort: easy | Time: immediate
   Ensure relevant skills are prominently displayed.
   Actions:
   • Move skills section near top
   • Use specific skill names matching job
   ...

⭐ TOP PRIORITY IMPROVEMENTS
────────────────────────────────────────────────────────────────────────────

1. [HIGH] Add Kubernetes to your skillset
   Category: Skills | Impact: 80% | Feasibility: 60%
   The job requires Kubernetes, missing from your resume.
   Key Actions:
   • Learn Kubernetes through online courses
   • Practice Kubernetes with projects
   ...

📋 IMPROVEMENT SUMMARY
────────────────────────────────────────────────────────────────────────────
Current Match Score: 75.3%
Improvement Potential: +15%
Projected Score: 90.3%

Total Suggestions Generated: 12
Quick Wins Identified: 5
Long-term Improvements: 7

💾 Detailed results saved to: results.json

================================================================================
                            ✨ ANALYSIS COMPLETE
================================================================================
Thank you for using Resume-Job Matcher!
Good luck with your application! 🍀
```

## Troubleshooting

### "File not found" error
- Ensure `job_description.txt` exists in the project root
- Check resume file path is correct
- Use absolute paths if needed

### "Module not found" error
- Activate virtual environment: `.venv\Scripts\activate` (Windows)
- Install dependencies: `pip install -r requirements.txt`

### Slow first run
- First run downloads AI models (~500MB)
- Subsequent runs are much faster
- Models are cached locally

### PDF parsing issues
- Ensure PDF is text-based (not scanned image)
- Try converting to DOCX or TXT
- Check PDF is not password-protected

## Advanced Usage

### Custom Job Description File

```bash
python main.py resume.pdf --job path/to/custom_job.txt
```

### Save Results for Later Review

```bash
python main.py resume.pdf --output analysis_results.json
```

### Quick Analysis (No Suggestions)

```bash
python main.py resume.pdf --no-suggestions
```

This skips the detailed suggestion generation for faster results.

## Understanding the Results

### Impact Score
- Represents expected improvement to match score
- Higher = more important to address
- Range: 0-100%

### Feasibility Score
- How easy the improvement is to implement
- Higher = easier to achieve
- Range: 0-100%

### Priority Levels
- **CRITICAL**: Must address immediately
- **HIGH**: Very important, address soon
- **MEDIUM**: Important, address when possible
- **LOW**: Nice to have, address if time permits

### Suggestion Categories
- **Skills**: Technical/soft skills to acquire
- **Experience**: How to present experience better
- **Education**: Certifications/degrees to pursue
- **ATS Optimization**: Resume formatting for ATS
- **Formatting**: General resume improvements

## Next Steps

1. Review your match score and component breakdown
2. Address quick wins first (easy, high-impact)
3. Work on high-priority missing skills
4. Update resume based on suggestions
5. Re-run analysis to see improvement
6. Iterate until you reach your target score

## Support

For issues or questions:
- Check the main README.md
- Review examples in the `examples/` directory
- Check documentation in `docs/` folder
