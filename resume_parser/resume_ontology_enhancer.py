"""
Resume-specific ontology enhancement module.

This module extends the existing skills ontology with resume-specific
skills, variations, and context-aware categorization.
"""

from typing import Dict, List, Set
import logging

from job_parser.logging_config import get_logger


class ResumeOntologyEnhancer:
    """Enhances skills ontology for resume-specific context."""
    
    def __init__(self):
        """Initialize the ontology enhancer."""
        self.logger = get_logger(__name__)
        
        # Resume-specific skill variations and synonyms
        self.skill_variations = {
            # Programming languages variations
            'JavaScript': ['JS', 'Javascript', 'ECMAScript', 'ES6', 'ES2015', 'Node.js'],
            'Python': ['Python3', 'Python 3', 'Py', 'CPython'],
            'C++': ['CPP', 'C Plus Plus', 'Cpp'],
            'C#': ['CSharp', 'C Sharp', 'dotnet', '.NET'],
            'TypeScript': ['TS', 'Typescript'],
            'SQL': ['MySQL', 'PostgreSQL', 'SQLite', 'T-SQL', 'PL/SQL'],
            
            # Frameworks and libraries
            'React': ['ReactJS', 'React.js', 'React Native'],
            'Angular': ['AngularJS', 'Angular 2+', 'Angular2'],
            'Vue.js': ['Vue', 'VueJS', 'Vuejs'],
            'Django': ['Django REST', 'DRF'],
            'Flask': ['Flask-RESTful', 'Flask API'],
            'Spring': ['Spring Boot', 'Spring Framework', 'SpringBoot'],
            'Express': ['Express.js', 'ExpressJS'],
            
            # Cloud platforms
            'AWS': ['Amazon Web Services', 'Amazon AWS', 'EC2', 'S3', 'Lambda'],
            'Azure': ['Microsoft Azure', 'Azure Cloud'],
            'GCP': ['Google Cloud Platform', 'Google Cloud', 'GCE'],
            
            # Tools and technologies
            'Docker': ['Containerization', 'Docker Compose'],
            'Kubernetes': ['K8s', 'Container Orchestration'],
            'Git': ['Version Control', 'GitHub', 'GitLab', 'Bitbucket'],
            'Jenkins': ['CI/CD', 'Continuous Integration'],
            
            # Data science and ML
            'Machine Learning': ['ML', 'Artificial Intelligence', 'AI'],
            'Deep Learning': ['Neural Networks', 'DL'],
            'TensorFlow': ['TF', 'Tensorflow'],
            'PyTorch': ['Torch'],
            'Pandas': ['Data Analysis', 'Data Manipulation'],
            'NumPy': ['Numpy', 'Numerical Computing'],
            'Scikit-learn': ['sklearn', 'scikit learn'],
            
            # Databases
            'MongoDB': ['Mongo', 'NoSQL'],
            'PostgreSQL': ['Postgres', 'PostGIS'],
            'Redis': ['Caching', 'In-memory Database'],
            'Elasticsearch': ['Elastic Search', 'ELK Stack'],
            
            # Soft skills variations
            'Communication': ['Verbal Communication', 'Written Communication', 'Presentation Skills'],
            'Leadership': ['Team Leadership', 'Project Leadership', 'Management'],
            'Problem Solving': ['Analytical Thinking', 'Troubleshooting', 'Critical Thinking'],
            'Teamwork': ['Collaboration', 'Team Player', 'Cross-functional Teams']
        }
        
        # Resume-specific additional skills not in base ontology
        self.resume_specific_skills = {
            'technical': [
                # Web development
                'REST API', 'GraphQL', 'WebSocket', 'JSON', 'XML', 'AJAX',
                'Responsive Design', 'Progressive Web Apps', 'PWA',
                'Single Page Applications', 'SPA', 'Microservices',
                
                # Mobile development
                'iOS Development', 'Android Development', 'Swift', 'Kotlin',
                'React Native', 'Flutter', 'Xamarin', 'Ionic',
                
                # DevOps and Infrastructure
                'Infrastructure as Code', 'IaC', 'Terraform', 'Ansible',
                'Monitoring', 'Logging', 'APM', 'Prometheus', 'Grafana',
                'Load Balancing', 'Auto Scaling', 'Blue-Green Deployment',
                
                # Testing
                'Unit Testing', 'Integration Testing', 'End-to-End Testing',
                'Test Driven Development', 'TDD', 'Behavior Driven Development', 'BDD',
                'Jest', 'Mocha', 'Pytest', 'JUnit', 'Selenium',
                
                # Security
                'Cybersecurity', 'Information Security', 'OWASP', 'Penetration Testing',
                'Encryption', 'Authentication', 'Authorization', 'OAuth', 'JWT',
                
                # Data Engineering
                'ETL', 'Data Pipeline', 'Data Warehouse', 'Big Data',
                'Apache Spark', 'Hadoop', 'Kafka', 'Airflow',
                
                # Blockchain
                'Blockchain', 'Cryptocurrency', 'Smart Contracts', 'Ethereum',
                'Solidity', 'Web3', 'DeFi', 'NFT'
            ],
            
            'tools': [
                # IDEs and Editors
                'Visual Studio Code', 'VS Code', 'IntelliJ IDEA', 'PyCharm',
                'Eclipse', 'Sublime Text', 'Atom', 'Vim', 'Emacs',
                
                # Project Management
                'Agile', 'Scrum', 'Kanban', 'Jira', 'Confluence', 'Trello',
                'Asana', 'Monday.com', 'Slack', 'Microsoft Teams',
                
                # Design Tools
                'Figma', 'Sketch', 'Adobe XD', 'Photoshop', 'Illustrator',
                'Canva', 'InVision', 'Zeplin',
                
                # Analytics and BI
                'Google Analytics', 'Tableau', 'Power BI', 'Looker',
                'Mixpanel', 'Amplitude', 'Segment',
                
                # Operating Systems
                'Linux', 'Ubuntu', 'CentOS', 'Red Hat', 'Windows Server',
                'macOS', 'Unix', 'Shell Scripting', 'Bash', 'PowerShell',
                
                # Virtualization
                'VMware', 'VirtualBox', 'Hyper-V', 'Vagrant',
                
                # API Tools
                'Postman', 'Insomnia', 'Swagger', 'OpenAPI'
            ],
            
            'soft': [
                # Leadership and Management
                'Strategic Planning', 'Decision Making', 'Delegation',
                'Conflict Resolution', 'Mentoring', 'Coaching',
                
                # Communication
                'Public Speaking', 'Technical Writing', 'Documentation',
                'Stakeholder Management', 'Client Relations', 'Negotiation',
                
                # Personal Skills
                'Self-motivated', 'Detail-oriented', 'Multitasking',
                'Stress Management', 'Emotional Intelligence', 'Empathy',
                'Curiosity', 'Continuous Learning', 'Growth Mindset',
                
                # Work Style
                'Remote Work', 'Cross-cultural Communication', 'Flexibility',
                'Initiative', 'Proactive', 'Results-oriented',
                'Customer-focused', 'Quality-focused'
            ]
        }
        
        # Context-based skill categorization rules
        self.context_categorization_rules = {
            'programming_indicators': ['programming', 'coding', 'development', 'scripting'],
            'framework_indicators': ['framework', 'library', 'package', 'module'],
            'tool_indicators': ['tool', 'platform', 'software', 'application', 'system'],
            'soft_skill_indicators': ['skill', 'ability', 'communication', 'leadership', 'management']
        }
    
    def enhance_ontology(self, base_ontology: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Enhance the base ontology with resume-specific skills and variations.
        
        Args:
            base_ontology: Base skills ontology
            
        Returns:
            Enhanced ontology with additional skills and variations
        """
        self.logger.info("Enhancing ontology for resume context")
        
        enhanced_ontology = {}
        
        # Start with base ontology
        for category, skills in base_ontology.items():
            enhanced_ontology[category] = skills.copy()
        
        # Add resume-specific skills
        for category, additional_skills in self.resume_specific_skills.items():
            if category not in enhanced_ontology:
                enhanced_ontology[category] = []
            
            # Add skills that aren't already present
            for skill in additional_skills:
                if skill not in enhanced_ontology[category]:
                    enhanced_ontology[category].append(skill)
        
        # Add skill variations
        for base_skill, variations in self.skill_variations.items():
            # Find which category the base skill belongs to
            skill_category = self._find_skill_category(base_skill, enhanced_ontology)
            
            if skill_category:
                for variation in variations:
                    if variation not in enhanced_ontology[skill_category]:
                        enhanced_ontology[skill_category].append(variation)
        
        # Log enhancement statistics
        original_count = sum(len(skills) for skills in base_ontology.values())
        enhanced_count = sum(len(skills) for skills in enhanced_ontology.values())
        
        self.logger.info(f"Ontology enhanced: {original_count} -> {enhanced_count} skills "
                        f"({enhanced_count - original_count} added)")
        
        return enhanced_ontology
    
    def get_skill_variations(self, skill: str) -> List[str]:
        """Get all variations of a given skill.
        
        Args:
            skill: Base skill name
            
        Returns:
            List of skill variations including the original
        """
        variations = [skill]  # Include the original skill
        
        # Check if this skill has defined variations
        if skill in self.skill_variations:
            variations.extend(self.skill_variations[skill])
        
        # Check if this skill is a variation of another skill
        for base_skill, skill_vars in self.skill_variations.items():
            if skill in skill_vars and base_skill not in variations:
                variations.append(base_skill)
                variations.extend([v for v in skill_vars if v not in variations])
        
        return variations
    
    def normalize_skill_name(self, skill: str) -> str:
        """Normalize a skill name to its canonical form.
        
        Args:
            skill: Skill name to normalize
            
        Returns:
            Canonical skill name
        """
        skill = skill.strip()
        
        # Check if this is a variation of a known skill
        for base_skill, variations in self.skill_variations.items():
            if skill.lower() in [v.lower() for v in variations]:
                return base_skill
            if skill.lower() == base_skill.lower():
                return base_skill
        
        return skill
    
    def categorize_unknown_skill(self, skill: str, context: str = "") -> str:
        """Categorize an unknown skill based on context and patterns.
        
        Args:
            skill: Skill to categorize
            context: Context where the skill was found
            
        Returns:
            Predicted category (technical, tools, soft)
        """
        skill_lower = skill.lower()
        context_lower = context.lower()
        
        # Check for programming language patterns
        if any(indicator in skill_lower for indicator in 
               ['++', '#', 'script', 'lang', 'programming']):
            return 'technical'
        
        # Check for framework/library patterns
        if any(indicator in skill_lower for indicator in 
               ['js', '.js', 'framework', 'library', 'api']):
            return 'technical'
        
        # Check for tool patterns
        if any(indicator in skill_lower for indicator in 
               ['tool', 'platform', 'software', 'app', 'system']):
            return 'tools'
        
        # Check context for categorization hints
        if context:
            if any(indicator in context_lower for indicator in 
                   self.context_categorization_rules['programming_indicators']):
                return 'technical'
            elif any(indicator in context_lower for indicator in 
                     self.context_categorization_rules['tool_indicators']):
                return 'tools'
            elif any(indicator in context_lower for indicator in 
                     self.context_categorization_rules['soft_skill_indicators']):
                return 'soft'
        
        # Default to technical for unknown skills
        return 'technical'
    
    def _find_skill_category(self, skill: str, ontology: Dict[str, List[str]]) -> str:
        """Find which category a skill belongs to in the ontology."""
        skill_lower = skill.lower()
        
        for category, skills in ontology.items():
            for ontology_skill in skills:
                if skill_lower == ontology_skill.lower():
                    return category
        
        return None
    
    def get_enhancement_statistics(self, base_ontology: Dict[str, List[str]], 
                                 enhanced_ontology: Dict[str, List[str]]) -> Dict[str, int]:
        """Get statistics about the ontology enhancement.
        
        Args:
            base_ontology: Original ontology
            enhanced_ontology: Enhanced ontology
            
        Returns:
            Dictionary with enhancement statistics
        """
        stats = {}
        
        for category in enhanced_ontology.keys():
            base_count = len(base_ontology.get(category, []))
            enhanced_count = len(enhanced_ontology[category])
            stats[f"{category}_original"] = base_count
            stats[f"{category}_enhanced"] = enhanced_count
            stats[f"{category}_added"] = enhanced_count - base_count
        
        stats['total_original'] = sum(len(skills) for skills in base_ontology.values())
        stats['total_enhanced'] = sum(len(skills) for skills in enhanced_ontology.values())
        stats['total_added'] = stats['total_enhanced'] - stats['total_original']
        
        return stats