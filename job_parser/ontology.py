"""
Ontology loading system for skills categorization.

This module handles loading and validation of the CSV-based skills ontology
that provides the knowledge base for skill extraction and categorization.
"""

import csv
import os
from typing import Dict, List

from .interfaces import OntologyLoaderInterface
from .exceptions import OntologyLoadError
from .logging_config import get_logger


class OntologyLoader(OntologyLoaderInterface):
    """Handles loading and validation of skills ontology from CSV files.
    
    This class implements the ontology loading interface and provides
    functionality to parse CSV files into structured skill categories.
    """
    
    # Minimal built-in ontology as fallback
    FALLBACK_ONTOLOGY = {
        "technical": [
            "Python", "JavaScript", "Java", "C++", "SQL", "HTML", "CSS",
            "React", "Node.js", "Django", "Flask", "Spring", "Angular",
            "Machine Learning", "Deep Learning", "Data Science", "AI",
            "TensorFlow", "PyTorch", "Pandas", "NumPy", "Scikit-learn"
        ],
        "soft": [
            "Communication", "Teamwork", "Problem Solving", "Leadership",
            "Time Management", "Critical Thinking", "Adaptability",
            "Collaboration", "Analytical Thinking", "Creativity"
        ],
        "tools": [
            "Git", "Docker", "Kubernetes", "AWS", "Azure", "GCP",
            "Jenkins", "Jira", "Confluence", "Slack", "VS Code",
            "IntelliJ", "Postman", "Tableau", "Power BI", "Excel"
        ]
    }
    
    def __init__(self):
        """Initialize the ontology loader."""
        self.logger = get_logger(__name__)
    
    def load_ontology(self, csv_path: str) -> Dict[str, List[str]]:
        """Load skills ontology from CSV file.
        
        Args:
            csv_path: Path to the CSV ontology file
            
        Returns:
            Dictionary with categories as keys and skill lists as values
            Example: {'technical': ['Python', 'React'], 'soft': ['teamwork']}
            
        Raises:
            OntologyLoadError: If ontology loading fails
        """
        self.logger.info(f"Loading ontology from {csv_path}")
        
        try:
            # Check if file exists
            if not os.path.exists(csv_path):
                raise OntologyLoadError(f"Ontology file not found: {csv_path}")
            
            # Initialize ontology dictionary
            ontology = {}
            
            # Read CSV file using DictReader
            with open(csv_path, 'r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                
                # Validate CSV structure
                if 'category' not in reader.fieldnames or 'skill' not in reader.fieldnames:
                    raise OntologyLoadError(
                        f"CSV file must contain 'category' and 'skill' columns. "
                        f"Found columns: {reader.fieldnames}"
                    )
                
                # Process each row
                for row_num, row in enumerate(reader, start=2):  # Start at 2 since header is row 1
                    try:
                        category = row['category'].strip()
                        skill = row['skill'].strip()
                        
                        # Skip empty rows
                        if not category or not skill:
                            self.logger.warning(f"Skipping empty row {row_num}: category='{category}', skill='{skill}'")
                            continue
                        
                        # Initialize category if not exists
                        if category not in ontology:
                            ontology[category] = []
                        
                        # Add skill to category (avoid duplicates)
                        if skill not in ontology[category]:
                            ontology[category].append(skill)
                        else:
                            self.logger.debug(f"Duplicate skill '{skill}' in category '{category}' at row {row_num}")
                            
                    except KeyError as e:
                        raise OntologyLoadError(f"Missing required column in row {row_num}: {e}")
                    except Exception as e:
                        raise OntologyLoadError(f"Error processing row {row_num}: {e}")
            
            # Validate that we loaded some data
            if not ontology:
                raise OntologyLoadError(f"No valid skills loaded from {csv_path}")
            
            # Log summary
            total_skills = sum(len(skills) for skills in ontology.values())
            self.logger.info(f"Successfully loaded {total_skills} skills across {len(ontology)} categories")
            for category, skills in ontology.items():
                self.logger.debug(f"Category '{category}': {len(skills)} skills")
            
            return ontology
            
        except OntologyLoadError:
            # Re-raise our custom exceptions
            raise
        except FileNotFoundError as e:
            raise OntologyLoadError(f"Ontology file not found: {csv_path}") from e
        except PermissionError as e:
            raise OntologyLoadError(f"Permission denied accessing ontology file: {csv_path}") from e
        except csv.Error as e:
            raise OntologyLoadError(f"CSV parsing error in {csv_path}: {e}") from e
        except UnicodeDecodeError as e:
            raise OntologyLoadError(f"Encoding error reading {csv_path}: {e}") from e
        except Exception as e:
            raise OntologyLoadError(f"Unexpected error loading ontology from {csv_path}: {e}") from e
    
    def load_ontology_with_fallback(self, csv_path: str) -> Dict[str, List[str]]:
        """Load skills ontology from CSV file with fallback to built-in ontology.
        
        This method attempts to load the ontology from the specified CSV file,
        but falls back to a minimal built-in ontology if loading fails.
        
        Args:
            csv_path: Path to the CSV ontology file
            
        Returns:
            Dictionary with categories as keys and skill lists as values
            
        Note:
            This method logs warnings but does not raise exceptions,
            making it suitable for production environments where
            graceful degradation is preferred.
        """
        try:
            return self.load_ontology(csv_path)
        except OntologyLoadError as e:
            self.logger.warning(f"Failed to load ontology from {csv_path}: {e}")
            self.logger.warning("Falling back to built-in minimal ontology")
            return self.get_fallback_ontology()
    
    def get_fallback_ontology(self) -> Dict[str, List[str]]:
        """Get the minimal built-in ontology as fallback.
        
        Returns:
            Dictionary containing minimal skills ontology
        """
        self.logger.info("Using fallback ontology with built-in skills")
        
        # Return a deep copy to prevent modification of the class constant
        fallback = {}
        for category, skills in self.FALLBACK_ONTOLOGY.items():
            fallback[category] = skills.copy()
        
        total_skills = sum(len(skills) for skills in fallback.values())
        self.logger.info(f"Fallback ontology contains {total_skills} skills across {len(fallback)} categories")
        
        return fallback