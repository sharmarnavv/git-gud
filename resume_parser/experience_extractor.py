"""
Work experience extractor for resume parsing.

This module provides functionality to extract work experience information
from resume text using pattern matching and NER techniques.
"""

import re
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass

from .resume_interfaces import ExperienceExtractorInterface, WorkExperience
from .resume_exceptions import ExperienceExtractionError
from job_parser.logging_config import get_logger


@dataclass
class ExperienceAnalysis:
    """Analysis results for work experience."""
    total_years_experience: float
    years_by_category: Dict[str, float]
    career_progression: List[Dict[str, Any]]
    seniority_level: str
    industry_experience: Dict[str, float]
    skills_by_experience: Dict[str, List[str]]
    experience_relevance_scores: Dict[str, float]


class ExperienceExtractor(ExperienceExtractorInterface):
    """Work experience extractor using pattern matching and NER."""
    
    def __init__(self):
        """Initialize the experience extractor."""
        self.logger = get_logger(__name__)
        
        # Compile regex patterns for better performance
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for experience extraction."""
        
        # Enhanced date patterns for various formats
        self.date_patterns = [
            re.compile(r'(\d{1,2})/(\d{1,2})/(\d{4})', re.IGNORECASE),  # MM/DD/YYYY or DD/MM/YYYY
            re.compile(r'(\d{1,2})/(\d{4})', re.IGNORECASE),            # MM/YYYY
            re.compile(r'(\d{1,2})-(\d{4})', re.IGNORECASE),            # MM-YYYY
            re.compile(r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+(\d{4})', re.IGNORECASE),  # Month YYYY
            re.compile(r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})', re.IGNORECASE),  # Full Month YYYY
            re.compile(r'(\d{4})', re.IGNORECASE),                      # YYYY only
            re.compile(r'Q[1-4]\s+(\d{4})', re.IGNORECASE),            # Q1 YYYY format
        ]
        
        # Date range patterns
        self.date_range_patterns = [
            re.compile(r'(\d{1,2}/\d{4})\s*[-–—]\s*(\d{1,2}/\d{4})', re.IGNORECASE),  # MM/YYYY - MM/YYYY
            re.compile(r'(\w+\s+\d{4})\s*[-–—]\s*(\w+\s+\d{4})', re.IGNORECASE),     # Month YYYY - Month YYYY
            re.compile(r'(\d{4})\s*[-–—]\s*(\d{4})', re.IGNORECASE),                 # YYYY - YYYY
            re.compile(r'(\w+\s+\d{4})\s*[-–—]\s*(present|current|now)', re.IGNORECASE),  # Month YYYY - Present
        ]
        
        # Experience section headers
        self.experience_headers = [
            re.compile(r'(?:work\s+)?experience', re.IGNORECASE),
            re.compile(r'employment\s+history', re.IGNORECASE),
            re.compile(r'professional\s+experience', re.IGNORECASE),
            re.compile(r'career\s+history', re.IGNORECASE),
            re.compile(r'work\s+history', re.IGNORECASE),
            re.compile(r'job\s+history', re.IGNORECASE),
        ]
        
        # Enhanced job title patterns with common titles
        self.job_title_indicators = [
            # Technical roles
            'engineer', 'developer', 'programmer', 'architect', 'analyst', 'scientist',
            'administrator', 'technician', 'specialist', 'consultant',
            # Management roles
            'manager', 'director', 'supervisor', 'lead', 'head', 'chief', 'president',
            'vice president', 'vp', 'ceo', 'cto', 'cfo', 'coo',
            # General roles
            'coordinator', 'assistant', 'associate', 'representative', 'officer',
            'executive', 'advisor', 'consultant', 'intern', 'trainee',
            # Seniority levels
            'senior', 'junior', 'principal', 'staff', 'lead', 'entry level'
        ]
        
        # Company name patterns (words that often appear in company names)
        self.company_indicators = [
            'inc', 'corp', 'corporation', 'company', 'co', 'ltd', 'limited',
            'llc', 'llp', 'technologies', 'tech', 'systems', 'solutions',
            'services', 'consulting', 'group', 'enterprises', 'international'
        ]
        
        # Current job indicators
        self.current_job_indicators = [
            'present', 'current', 'now', 'ongoing', 'today', 'till date', 'to date'
        ]
    
    def extract_experience(self, text: str) -> List[WorkExperience]:
        """Extract work experience from resume text.
        
        Args:
            text: Resume text content
            
        Returns:
            List of WorkExperience objects
        """
        try:
            self.logger.info("Extracting work experience from resume")
            
            # Find experience section
            experience_section = self._find_experience_section(text)
            
            if not experience_section:
                self.logger.warning("No experience section found")
                return []
            
            # Extract individual experience entries
            experiences = self._parse_experience_entries(experience_section)
            
            self.logger.info(f"Extracted {len(experiences)} work experience entries")
            return experiences
            
        except Exception as e:
            self.logger.error(f"Experience extraction failed: {e}")
            raise ExperienceExtractionError(f"Failed to extract experience: {e}", cause=e)
    
    def _find_experience_section(self, text: str) -> str:
        """Find and extract the experience section from resume text."""
        lines = text.split('\n')
        
        in_experience_section = False
        experience_lines = []
        
        for line in lines:
            line = line.strip()
            
            # Check if this line starts an experience section
            if any(pattern.search(line) for pattern in self.experience_headers):
                in_experience_section = True
                continue
            
            # If we're in experience section, collect lines until we hit another section
            if in_experience_section:
                if self._is_new_section_header(line):
                    break
                else:
                    experience_lines.append(line)
        
        return '\n'.join(experience_lines)
    
    def _is_new_section_header(self, line: str) -> bool:
        """Check if line is a new section header."""
        common_headers = [
            'education', 'skills', 'projects', 'certifications',
            'achievements', 'awards', 'publications', 'references'
        ]
        
        line_lower = line.lower().strip()
        
        # Check for common section headers
        for header in common_headers:
            if line_lower.startswith(header) and ':' in line:
                return True
        
        # Check for all caps headers
        if line.isupper() and len(line.split()) <= 3:
            return True
        
        return False
    
    def _parse_experience_entries(self, experience_text: str) -> List[WorkExperience]:
        """Parse individual experience entries from experience section text."""
        experiences = []
        
        # Split by potential job entries (look for patterns that indicate new jobs)
        entries = self._split_into_entries(experience_text)
        
        for entry in entries:
            experience = self._parse_single_experience(entry)
            if experience and experience.job_title:  # Only add if we found a job title
                experiences.append(experience)
        
        return experiences
    
    def _split_into_entries(self, text: str) -> List[str]:
        """Split experience text into individual job entries."""
        lines = text.split('\n')
        entries = []
        current_entry = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if this line looks like a new job entry
            if self._looks_like_job_header(line):
                # Save previous entry
                if current_entry:
                    entries.append('\n'.join(current_entry))
                    current_entry = []
            
            current_entry.append(line)
        
        # Add final entry
        if current_entry:
            entries.append('\n'.join(current_entry))
        
        return entries
    
    def _looks_like_job_header(self, line: str) -> bool:
        """Check if line looks like a job header (title, company, dates)."""
        line_lower = line.lower()
        
        # Check for job title indicators
        has_job_title = any(indicator in line_lower for indicator in self.job_title_indicators)
        
        # Check for date patterns
        has_dates = any(pattern.search(line) for pattern in self.date_patterns)
        
        # Check for company indicators (at, @, |, -)
        has_company_separator = any(sep in line for sep in ['at ', '@ ', ' | ', ' - '])
        
        return has_job_title or has_dates or has_company_separator
    
    def _parse_single_experience(self, entry_text: str) -> Optional[WorkExperience]:
        """Parse a single work experience entry."""
        try:
            experience = WorkExperience()
            lines = [line.strip() for line in entry_text.split('\n') if line.strip()]
            
            if not lines:
                return None
            
            # First line usually contains job title, company, and dates
            header_line = lines[0]
            
            # Extract job title and company
            job_title, company = self._extract_title_and_company(header_line)
            experience.job_title = job_title
            experience.company = company
            
            # Extract dates from entire entry (not just header)
            start_date, end_date = self._extract_dates(entry_text)
            experience.start_date = start_date
            experience.end_date = end_date
            
            # Calculate duration
            experience.duration_months = self._calculate_duration(start_date, end_date)
            
            # Extract and clean description
            description = self._extract_description(lines)
            experience.description = description
            
            # Extract skills mentioned in this experience
            experience.skills_used = self._extract_skills_from_description(description)
            
            return experience
            
        except Exception as e:
            self.logger.warning(f"Failed to parse experience entry: {e}")
            return None
    
    def _extract_description(self, lines: List[str]) -> str:
        """Extract and clean job description from lines."""
        if len(lines) <= 1:
            return ""
        
        description_lines = []
        
        # Skip the header line and process remaining lines
        for line in lines[1:]:
            # Skip lines that look like they contain only dates or company info
            if self._is_metadata_line(line):
                continue
            
            # Clean up bullet points and formatting
            cleaned_line = self._clean_description_line(line)
            if cleaned_line:
                description_lines.append(cleaned_line)
        
        return '\n'.join(description_lines)
    
    def _is_metadata_line(self, line: str) -> bool:
        """Check if line contains metadata (dates, location) rather than job description."""
        line_lower = line.lower().strip()
        
        # Check if line is mostly dates
        date_chars = sum(1 for c in line if c.isdigit() or c in '/-')
        if date_chars > len(line) * 0.3:  # More than 30% date characters
            return True
        
        # Check for location patterns
        location_indicators = ['location:', 'city:', 'state:', 'country:']
        if any(indicator in line_lower for indicator in location_indicators):
            return True
        
        # Check if line is very short and contains common metadata words
        metadata_words = ['duration:', 'period:', 'tenure:', 'location', 'remote', 'onsite']
        if len(line.split()) <= 3 and any(word in line_lower for word in metadata_words):
            return True
        
        return False
    
    def _clean_description_line(self, line: str) -> str:
        """Clean a single description line."""
        # Remove bullet points and numbering
        line = re.sub(r'^[\s\-\*\•\d\.\)]+', '', line)
        
        # Remove excessive whitespace
        line = re.sub(r'\s+', ' ', line)
        
        return line.strip()
    
    def _extract_skills_from_description(self, description: str) -> List[str]:
        """Extract potential skills mentioned in job description."""
        if not description:
            return []
        
        skills = []
        description_lower = description.lower()
        
        # Common technical skills patterns
        tech_skills = [
            'python', 'java', 'javascript', 'c++', 'c#', 'sql', 'html', 'css',
            'react', 'angular', 'vue', 'node.js', 'django', 'flask', 'spring',
            'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'git',
            'machine learning', 'data analysis', 'project management', 'agile',
            'scrum', 'leadership', 'team management', 'communication'
        ]
        
        for skill in tech_skills:
            if skill in description_lower:
                skills.append(skill)
        
        return skills
    
    def analyze_experience(self, experiences: List[WorkExperience]) -> ExperienceAnalysis:
        """Analyze work experience for career progression and patterns.
        
        Args:
            experiences: List of work experiences to analyze
            
        Returns:
            ExperienceAnalysis object with detailed analysis
        """
        try:
            self.logger.info("Analyzing work experience patterns")
            
            # Calculate total years of experience
            total_years = self._calculate_total_experience_years(experiences)
            
            # Calculate years by category
            years_by_category = self._calculate_years_by_category(experiences)
            
            # Analyze career progression
            career_progression = self._analyze_career_progression(experiences)
            
            # Determine seniority level
            seniority_level = self._determine_seniority_level(experiences, total_years)
            
            # Calculate industry experience
            industry_experience = self._calculate_industry_experience(experiences)
            
            # Extract skills by experience
            skills_by_experience = self._extract_skills_by_experience(experiences)
            
            # Calculate relevance scores
            relevance_scores = self._calculate_experience_relevance(experiences)
            
            analysis = ExperienceAnalysis(
                total_years_experience=total_years,
                years_by_category=years_by_category,
                career_progression=career_progression,
                seniority_level=seniority_level,
                industry_experience=industry_experience,
                skills_by_experience=skills_by_experience,
                experience_relevance_scores=relevance_scores
            )
            
            self.logger.info(f"Experience analysis completed: {total_years:.1f} years total, {seniority_level} level")
            return analysis
            
        except Exception as e:
            self.logger.error(f"Experience analysis failed: {e}")
            raise ExperienceExtractionError(f"Failed to analyze experience: {e}", cause=e)
    
    def _calculate_total_experience_years(self, experiences: List[WorkExperience]) -> float:
        """Calculate total years of work experience."""
        total_months = sum(exp.duration_months for exp in experiences)
        return round(total_months / 12.0, 1)
    
    def _calculate_years_by_category(self, experiences: List[WorkExperience]) -> Dict[str, float]:
        """Calculate years of experience by job category."""
        categories = {
            'technical': 0,
            'management': 0,
            'consulting': 0,
            'sales': 0,
            'marketing': 0,
            'operations': 0,
            'other': 0
        }
        
        # Define category keywords
        category_keywords = {
            'technical': ['engineer', 'developer', 'programmer', 'architect', 'analyst', 'scientist', 'technician'],
            'management': ['manager', 'director', 'supervisor', 'lead', 'head', 'chief', 'president', 'vp'],
            'consulting': ['consultant', 'advisor', 'specialist'],
            'sales': ['sales', 'account', 'business development', 'revenue'],
            'marketing': ['marketing', 'brand', 'communications', 'pr', 'advertising'],
            'operations': ['operations', 'logistics', 'supply chain', 'coordinator']
        }
        
        for experience in experiences:
            job_title_lower = experience.job_title.lower()
            years = experience.duration_months / 12.0
            
            # Find matching category
            matched_category = 'other'
            for category, keywords in category_keywords.items():
                if any(keyword in job_title_lower for keyword in keywords):
                    matched_category = category
                    break
            
            categories[matched_category] += years
        
        # Round values
        return {k: round(v, 1) for k, v in categories.items() if v > 0}
    
    def _analyze_career_progression(self, experiences: List[WorkExperience]) -> List[Dict[str, Any]]:
        """Analyze career progression patterns."""
        if not experiences:
            return []
        
        # Sort experiences by start date (most recent first)
        sorted_experiences = self._sort_experiences_by_date(experiences)
        
        progression = []
        
        for i, exp in enumerate(sorted_experiences):
            entry = {
                'position': exp.job_title,
                'company': exp.company,
                'duration_months': exp.duration_months,
                'seniority_score': self._calculate_seniority_score(exp.job_title),
                'is_promotion': False,
                'career_change': False
            }
            
            # Compare with previous position
            if i > 0:
                prev_exp = sorted_experiences[i - 1]
                
                # Check for promotion (same company, higher seniority)
                if (exp.company.lower() == prev_exp.company.lower() and 
                    entry['seniority_score'] > self._calculate_seniority_score(prev_exp.job_title)):
                    entry['is_promotion'] = True
                
                # Check for career change (different industry/role type)
                if self._is_career_change(exp, prev_exp):
                    entry['career_change'] = True
            
            progression.append(entry)
        
        return progression
    
    def _sort_experiences_by_date(self, experiences: List[WorkExperience]) -> List[WorkExperience]:
        """Sort experiences by start date (most recent first)."""
        def date_sort_key(exp):
            try:
                if exp.start_date:
                    # Extract year for sorting
                    year_match = re.search(r'(\d{4})', exp.start_date)
                    if year_match:
                        return int(year_match.group(1))
                return 0
            except:
                return 0
        
        return sorted(experiences, key=date_sort_key, reverse=True)
    
    def _calculate_seniority_score(self, job_title: str) -> int:
        """Calculate seniority score based on job title."""
        title_lower = job_title.lower()
        
        # Senior level indicators
        if any(word in title_lower for word in ['chief', 'president', 'vp', 'vice president']):
            return 5
        elif any(word in title_lower for word in ['director', 'head of', 'principal']):
            return 4
        elif any(word in title_lower for word in ['senior', 'lead', 'staff']):
            return 3
        elif any(word in title_lower for word in ['manager', 'supervisor']):
            return 3
        elif any(word in title_lower for word in ['junior', 'entry', 'intern', 'trainee']):
            return 1
        else:
            return 2  # Mid-level
    
    def _determine_seniority_level(self, experiences: List[WorkExperience], total_years: float) -> str:
        """Determine overall seniority level."""
        if total_years < 2:
            return "Entry Level"
        elif total_years < 5:
            return "Junior"
        elif total_years < 8:
            return "Mid-Level"
        elif total_years < 12:
            return "Senior"
        else:
            return "Executive/Principal"
    
    def _calculate_industry_experience(self, experiences: List[WorkExperience]) -> Dict[str, float]:
        """Calculate years of experience by industry."""
        industries = {}
        
        # Simple industry classification based on company names and job descriptions
        industry_keywords = {
            'technology': ['tech', 'software', 'systems', 'digital', 'data', 'ai', 'ml'],
            'finance': ['bank', 'financial', 'investment', 'capital', 'trading'],
            'healthcare': ['health', 'medical', 'hospital', 'pharma', 'biotech'],
            'consulting': ['consulting', 'advisory', 'services'],
            'retail': ['retail', 'commerce', 'shopping', 'store'],
            'manufacturing': ['manufacturing', 'production', 'industrial'],
            'education': ['education', 'university', 'school', 'academic']
        }
        
        for experience in experiences:
            years = experience.duration_months / 12.0
            company_lower = experience.company.lower()
            description_lower = experience.description.lower()
            
            # Find matching industry
            matched_industry = 'other'
            for industry, keywords in industry_keywords.items():
                if any(keyword in company_lower or keyword in description_lower for keyword in keywords):
                    matched_industry = industry
                    break
            
            industries[matched_industry] = industries.get(matched_industry, 0) + years
        
        return {k: round(v, 1) for k, v in industries.items() if v > 0}
    
    def _extract_skills_by_experience(self, experiences: List[WorkExperience]) -> Dict[str, List[str]]:
        """Extract skills mentioned in each work experience."""
        skills_by_exp = {}
        
        for experience in experiences:
            exp_key = f"{experience.job_title} at {experience.company}"
            skills_by_exp[exp_key] = experience.skills_used
        
        return skills_by_exp
    
    def _calculate_experience_relevance(self, experiences: List[WorkExperience]) -> Dict[str, float]:
        """Calculate relevance scores for each experience."""
        relevance_scores = {}
        
        for experience in experiences:
            exp_key = f"{experience.job_title} at {experience.company}"
            
            # Base score from duration (longer = more relevant)
            duration_score = min(experience.duration_months / 24.0, 1.0)  # Max 2 years = 1.0
            
            # Recency score (more recent = more relevant)
            recency_score = self._calculate_recency_score(experience.start_date)
            
            # Seniority score
            seniority_score = self._calculate_seniority_score(experience.job_title) / 5.0
            
            # Combined relevance score
            relevance = (duration_score * 0.4 + recency_score * 0.4 + seniority_score * 0.2)
            relevance_scores[exp_key] = round(relevance, 2)
        
        return relevance_scores
    
    def _calculate_recency_score(self, start_date: str) -> float:
        """Calculate recency score (0-1) based on how recent the experience is."""
        try:
            if not start_date:
                return 0.5
            
            # Extract year
            year_match = re.search(r'(\d{4})', start_date)
            if not year_match:
                return 0.5
            
            start_year = int(year_match.group(1))
            current_year = datetime.now().year
            years_ago = current_year - start_year
            
            # Score decreases with age (5+ years ago = 0.1, current year = 1.0)
            if years_ago <= 0:
                return 1.0
            elif years_ago >= 5:
                return 0.1
            else:
                return 1.0 - (years_ago * 0.18)  # Linear decrease
        
        except:
            return 0.5
    
    def _is_career_change(self, current_exp: WorkExperience, previous_exp: WorkExperience) -> bool:
        """Determine if there was a significant career change between positions."""
        # Simple heuristic: different company and significantly different job title keywords
        if current_exp.company.lower() == previous_exp.company.lower():
            return False
        
        current_keywords = set(current_exp.job_title.lower().split())
        previous_keywords = set(previous_exp.job_title.lower().split())
        
        # If less than 30% overlap in keywords, consider it a career change
        if len(current_keywords & previous_keywords) / len(current_keywords | previous_keywords) < 0.3:
            return True
        
        return False
    
    def _extract_title_and_company(self, header_line: str) -> tuple[str, str]:
        """Extract job title and company from header line."""
        job_title = ""
        company = ""
        
        # Clean the header line
        clean_line = self._remove_dates_from_text(header_line)
        
        # Try different separator patterns (in order of preference)
        separators = [' at ', ' @ ', ' | ', ' - ', ' – ', ' — ', ', ']
        
        for sep in separators:
            if sep in clean_line:
                parts = clean_line.split(sep, 1)
                if len(parts) == 2:
                    potential_title = parts[0].strip()
                    potential_company = parts[1].strip()
                    
                    # Validate that the first part looks like a job title
                    if self._looks_like_job_title(potential_title):
                        job_title = potential_title
                        company = self._clean_company_name(potential_company)
                        break
        
        # If no separator found, try to identify job title by keywords
        if not job_title:
            job_title = self._extract_job_title_by_keywords(clean_line)
            
            # Try to find company name in remaining text
            if job_title:
                remaining_text = clean_line.replace(job_title, '').strip()
                company = self._clean_company_name(remaining_text)
        
        # Final cleanup
        job_title = self._clean_job_title(job_title)
        
        return job_title, company
    
    def _looks_like_job_title(self, text: str) -> bool:
        """Check if text looks like a job title."""
        text_lower = text.lower()
        
        # Check for job title indicators
        has_job_indicator = any(indicator in text_lower for indicator in self.job_title_indicators)
        
        # Check length (job titles are usually not too long)
        reasonable_length = len(text.split()) <= 6
        
        # Check that it's not obviously a company name
        not_company = not any(indicator in text_lower for indicator in self.company_indicators)
        
        return has_job_indicator and reasonable_length and not_company
    
    def _extract_job_title_by_keywords(self, text: str) -> str:
        """Extract job title using keyword matching."""
        words = text.split()
        best_match = ""
        best_score = 0
        
        # Look for sequences of words that contain job title indicators
        for i in range(len(words)):
            for j in range(i + 1, min(i + 7, len(words) + 1)):  # Max 6 words for job title
                phrase = ' '.join(words[i:j])
                score = self._score_job_title_phrase(phrase)
                
                if score > best_score:
                    best_score = score
                    best_match = phrase
        
        return best_match if best_score > 0 else ""
    
    def _score_job_title_phrase(self, phrase: str) -> int:
        """Score a phrase for how likely it is to be a job title."""
        phrase_lower = phrase.lower()
        score = 0
        
        # Points for job title indicators
        for indicator in self.job_title_indicators:
            if indicator in phrase_lower:
                score += 2
        
        # Bonus for seniority levels
        seniority_levels = ['senior', 'junior', 'lead', 'principal', 'staff', 'entry']
        for level in seniority_levels:
            if level in phrase_lower:
                score += 1
        
        # Penalty for company indicators
        for indicator in self.company_indicators:
            if indicator in phrase_lower:
                score -= 3
        
        return score
    
    def _clean_job_title(self, title: str) -> str:
        """Clean and normalize job title."""
        if not title:
            return ""
        
        # Remove common prefixes/suffixes that aren't part of the title
        title = title.strip()
        
        # Remove bullet points, numbers, etc.
        title = re.sub(r'^[\d\.\-\*\•]+\s*', '', title)
        
        # Remove trailing punctuation
        title = title.rstrip('.,;:')
        
        return title.strip()
    
    def _clean_company_name(self, company: str) -> str:
        """Clean and normalize company name."""
        if not company:
            return ""
        
        company = company.strip()
        
        # Remove common prefixes
        company = re.sub(r'^(at|@)\s+', '', company, flags=re.IGNORECASE)
        
        # Remove trailing punctuation and dates
        company = re.sub(r'[,\.\-\|]+$', '', company)
        
        # Remove any remaining date patterns
        company = self._remove_dates_from_text(company)
        
        return company.strip()
    
    def _extract_dates(self, text: str) -> tuple[str, Optional[str]]:
        """Extract and normalize start and end dates from text."""
        # First try to find date ranges
        for pattern in self.date_range_patterns:
            match = pattern.search(text)
            if match:
                start_date = self._normalize_date(match.group(1))
                end_date = match.group(2).lower()
                
                if end_date in [indicator.lower() for indicator in self.current_job_indicators]:
                    return start_date, None  # Current job
                else:
                    return start_date, self._normalize_date(match.group(2))
        
        # If no range found, extract individual dates
        dates = []
        for pattern in self.date_patterns:
            matches = pattern.findall(text)
            for match in matches:
                if isinstance(match, tuple):
                    if len(match) == 3:  # MM/DD/YYYY format
                        dates.append(f"{match[0]}/{match[2]}")
                    elif len(match) == 2:  # MM/YYYY or Month YYYY format
                        dates.append(f"{match[0]} {match[1]}")
                else:
                    # Handle YYYY format
                    dates.append(match)
        
        # Check for current job indicators
        has_current = any(indicator.lower() in text.lower() for indicator in self.current_job_indicators)
        
        if len(dates) >= 2:
            start_date = self._normalize_date(dates[0])
            end_date = self._normalize_date(dates[1]) if not has_current else None
            return start_date, end_date
        elif len(dates) == 1:
            start_date = self._normalize_date(dates[0])
            return start_date, None if has_current else start_date
        else:
            return "", None
    
    def _normalize_date(self, date_str: str) -> str:
        """Normalize date string to consistent format (MM/YYYY)."""
        if not date_str:
            return ""
        
        date_str = date_str.strip()
        
        # Handle full month names
        month_mapping = {
            'january': '01', 'february': '02', 'march': '03', 'april': '04',
            'may': '05', 'june': '06', 'july': '07', 'august': '08',
            'september': '09', 'october': '10', 'november': '11', 'december': '12',
            'jan': '01', 'feb': '02', 'mar': '03', 'apr': '04',
            'jun': '06', 'jul': '07', 'aug': '08', 'sep': '09',
            'oct': '10', 'nov': '11', 'dec': '12'
        }
        
        # Try to parse different formats
        for month_name, month_num in month_mapping.items():
            if month_name in date_str.lower():
                year_match = re.search(r'(\d{4})', date_str)
                if year_match:
                    return f"{month_num}/{year_match.group(1)}"
        
        # Handle MM/YYYY or MM-YYYY format
        mm_yyyy_match = re.match(r'(\d{1,2})[-/](\d{4})', date_str)
        if mm_yyyy_match:
            month = mm_yyyy_match.group(1).zfill(2)
            year = mm_yyyy_match.group(2)
            return f"{month}/{year}"
        
        # Handle YYYY only
        yyyy_match = re.match(r'(\d{4})', date_str)
        if yyyy_match:
            return f"01/{yyyy_match.group(1)}"  # Default to January
        
        return date_str
    
    def _remove_dates_from_text(self, text: str) -> str:
        """Remove date patterns from text."""
        for pattern in self.date_patterns:
            text = pattern.sub('', text)
        
        # Remove common date-related words
        date_words = ['present', 'current', 'now', 'ongoing', '-', '–', '|']
        for word in date_words:
            text = text.replace(word, '')
        
        return text.strip()
    
    def _calculate_duration(self, start_date: str, end_date: Optional[str]) -> int:
        """Calculate duration in months between start and end dates."""
        try:
            if not start_date:
                return 0
            
            start_month, start_year = self._parse_date_components(start_date)
            
            if end_date:
                end_month, end_year = self._parse_date_components(end_date)
            else:
                # Current job - use current date
                now = datetime.now()
                end_month, end_year = now.month, now.year
            
            if start_year and end_year:
                # Calculate total months
                total_months = (end_year - start_year) * 12 + (end_month - start_month)
                return max(1, total_months)  # Minimum 1 month
            
            return 0
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate duration: {e}")
            return 0
    
    def _parse_date_components(self, date_str: str) -> tuple[int, int]:
        """Parse date string to extract month and year components."""
        if not date_str:
            return 1, datetime.now().year
        
        # Handle MM/YYYY format
        mm_yyyy_match = re.match(r'(\d{1,2})/(\d{4})', date_str)
        if mm_yyyy_match:
            month = int(mm_yyyy_match.group(1))
            year = int(mm_yyyy_match.group(2))
            return month, year
        
        # Handle YYYY only
        yyyy_match = re.search(r'(\d{4})', date_str)
        if yyyy_match:
            year = int(yyyy_match.group(1))
            return 1, year  # Default to January
        
        return 1, datetime.now().year
    
    def _extract_year(self, date_str: str) -> Optional[int]:
        """Extract year from date string."""
        try:
            # Look for 4-digit year
            year_match = re.search(r'(\d{4})', date_str)
            if year_match:
                return int(year_match.group(1))
            return None
        except Exception:
            return None