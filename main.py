#!/usr/bin/env python3
"""
Job Description Parser CLI - Command-line interface for parsing job descriptions.

This script provides a command-line interface for the Job Description Parser,
supporting single file processing, batch processing, and various output formats.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

from job_parser import JobDescriptionParser, ParserConfig
from job_parser.exceptions import JobParserError, InputValidationError
from job_parser.logging_config import get_logger


def setup_cli_logging(verbose: bool = False) -> logging.Logger:
    """Set up logging for CLI operations.
    
    Args:
        verbose: Enable verbose logging output
        
    Returns:
        Configured logger instance
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stderr)]
    )
    return get_logger(__name__)


def load_job_descriptions_from_file(file_path: Path, logger: logging.Logger) -> List[Dict[str, Any]]:
    """Load job descriptions from JSON file.
    
    Args:
        file_path: Path to JSON file containing job descriptions
        logger: Logger instance for error reporting
        
    Returns:
        List of job description dictionaries
        
    Raises:
        FileNotFoundError: If input file doesn't exist
        json.JSONDecodeError: If file contains invalid JSON
    """
    try:
        logger.info(f"Loading job descriptions from: {file_path}")
        
        if not file_path.exists():
            raise FileNotFoundError(f"Input file not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle both single job description and list of job descriptions
        if isinstance(data, dict):
            # Single job description
            job_descriptions = [data]
        elif isinstance(data, list):
            # List of job descriptions
            job_descriptions = data
        else:
            raise ValueError(f"Invalid JSON format. Expected dict or list, got {type(data)}")
        
        logger.info(f"Loaded {len(job_descriptions)} job description(s)")
        return job_descriptions
        
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in file {file_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to load job descriptions from {file_path}: {e}")
        raise


def extract_job_text(job_data: Dict[str, Any], logger: logging.Logger) -> str:
    """Extract job description text from job data dictionary.
    
    Args:
        job_data: Dictionary containing job information
        logger: Logger instance for warnings
        
    Returns:
        Job description text string
        
    Raises:
        ValueError: If no valid job text is found
    """
    # Try different possible field names for job description text
    text_fields = ['description', 'job_description', 'text', 'content', 'body']
    
    for field in text_fields:
        if field in job_data and job_data[field]:
            text = str(job_data[field]).strip()
            if text:
                logger.debug(f"Found job text in field: {field}")
                return text
    
    # If no standard field found, try to construct from available fields
    if 'title' in job_data or 'company' in job_data:
        parts = []
        if 'title' in job_data:
            parts.append(f"Job Title: {job_data['title']}")
        if 'company' in job_data:
            parts.append(f"Company: {job_data['company']}")
        
        # Add any other text fields
        for key, value in job_data.items():
            if key not in ['title', 'company', 'id'] and isinstance(value, str) and len(value) > 20:
                parts.append(f"{key.title()}: {value}")
        
        if parts:
            constructed_text = "\n".join(parts)
            logger.warning(f"No 'description' field found, constructed text from available fields")
            return constructed_text
    
    raise ValueError(f"No valid job description text found in data: {list(job_data.keys())}")


def format_output(parsed_result: Dict[str, Any], output_format: str, pretty: bool = False) -> str:
    """Format parsing results according to specified output format.
    
    Args:
        parsed_result: Parsed job description result
        output_format: Output format ('json' or 'summary')
        pretty: Enable pretty printing for JSON output
        
    Returns:
        Formatted output string
    """
    if output_format == 'json':
        if pretty:
            return json.dumps(parsed_result, indent=2, ensure_ascii=False)
        else:
            return json.dumps(parsed_result, ensure_ascii=False)
    
    elif output_format == 'summary':
        # Create a human-readable summary
        skills = parsed_result.get('skills_required', [])
        tools = parsed_result.get('tools_mentioned', [])
        experience = parsed_result.get('experience_level', 'unknown')
        categories = parsed_result.get('categories', {})
        
        summary_lines = [
            f"=== Job Description Parsing Summary ===",
            f"Experience Level: {experience}",
            f"Total Skills Found: {len(skills)}",
            f"Tools Mentioned: {len(tools)}",
            ""
        ]
        
        if categories:
            summary_lines.append("Skills by Category:")
            for category, category_skills in categories.items():
                if category_skills:
                    summary_lines.append(f"  {category.title()}: {', '.join(category_skills)}")
            summary_lines.append("")
        
        if tools:
            summary_lines.append(f"Tools/Technologies: {', '.join(tools)}")
            summary_lines.append("")
        
        # Add confidence scores if available
        confidence_scores = parsed_result.get('confidence_scores', {})
        if confidence_scores:
            summary_lines.append("Top Skills by Confidence:")
            sorted_skills = sorted(confidence_scores.items(), key=lambda x: x[1], reverse=True)[:10]
            for skill, confidence in sorted_skills:
                summary_lines.append(f"  {skill}: {confidence:.2f}")
        
        return "\n".join(summary_lines)
    
    else:
        raise ValueError(f"Unsupported output format: {output_format}")


def process_single_job(parser: JobDescriptionParser, 
                      job_data: Dict[str, Any], 
                      job_id: Optional[str],
                      logger: logging.Logger) -> Dict[str, Any]:
    """Process a single job description.
    
    Args:
        parser: JobDescriptionParser instance
        job_data: Job description data dictionary
        job_id: Optional job ID for identification
        logger: Logger instance
        
    Returns:
        Dictionary with parsing results and metadata
        
    Raises:
        JobParserError: If parsing fails
    """
    try:
        # Extract job description text
        job_text = extract_job_text(job_data, logger)
        
        # Parse the job description
        logger.info(f"Parsing job {job_id or 'unknown'}")
        parsed_result = parser.parse_job_description(job_text)
        
        # Convert to JSON-serializable format
        result_dict = parsed_result.to_json()
        
        # Add job metadata if available
        if job_id:
            result_dict['job_id'] = job_id
        
        # Add original job metadata
        metadata_fields = ['title', 'company', 'source', 'experience_level']
        original_metadata = {}
        for field in metadata_fields:
            if field in job_data:
                original_metadata[field] = job_data[field]
        
        if original_metadata:
            result_dict['original_metadata'] = original_metadata
        
        logger.info(f"Successfully parsed job {job_id or 'unknown'}: "
                   f"{len(result_dict['skills_required'])} skills found")
        
        return result_dict
        
    except Exception as e:
        logger.error(f"Failed to parse job {job_id or 'unknown'}: {e}")
        # Return error result
        return {
            'job_id': job_id,
            'error': str(e),
            'error_type': type(e).__name__,
            'skills_required': [],
            'experience_level': 'unknown',
            'tools_mentioned': [],
            'confidence_scores': {},
            'categories': {},
            'metadata': {
                'parsing_failed': True,
                'error_message': str(e)
            }
        }


def process_batch(parser: JobDescriptionParser, 
                 job_descriptions: List[Dict[str, Any]], 
                 logger: logging.Logger) -> List[Dict[str, Any]]:
    """Process multiple job descriptions in optimized batch mode.
    
    Args:
        parser: JobDescriptionParser instance
        job_descriptions: List of job description dictionaries
        logger: Logger instance
        
    Returns:
        List of parsing results
    """
    results = []
    total_jobs = len(job_descriptions)
    
    logger.info(f"Starting optimized batch processing of {total_jobs} job descriptions")
    
    try:
        # Optimize parser for batch processing
        parser.optimize_for_batch_processing(total_jobs)
        
        # Extract job texts for batch processing
        job_texts = []
        job_metadata = []
        
        for i, job_data in enumerate(job_descriptions):
            job_id = job_data.get('id', f"job_{i+1}")
            
            try:
                job_text = extract_job_text(job_data, logger)
                job_texts.append(job_text)
                
                # Store metadata for later use
                metadata_fields = ['title', 'company', 'source', 'experience_level']
                original_metadata = {'job_id': job_id}
                for field in metadata_fields:
                    if field in job_data:
                        original_metadata[field] = job_data[field]
                
                job_metadata.append(original_metadata)
                
            except Exception as e:
                logger.error(f"Failed to extract text for job {job_id}: {e}")
                # Add placeholder for failed extraction
                job_texts.append("")
                job_metadata.append({
                    'job_id': job_id,
                    'extraction_failed': True,
                    'error_message': str(e)
                })
        
        # Process jobs in batch using optimized parser
        logger.info("Processing jobs using optimized batch mode")
        parsed_results = parser.parse_job_descriptions_batch(job_texts)
        
        # Combine results with metadata
        for i, (parsed_result, metadata) in enumerate(zip(parsed_results, job_metadata)):
            try:
                if metadata.get('extraction_failed'):
                    # Handle extraction failure
                    result_dict = {
                        'job_id': metadata['job_id'],
                        'error': metadata['error_message'],
                        'error_type': 'TextExtractionError',
                        'skills_required': [],
                        'experience_level': 'unknown',
                        'tools_mentioned': [],
                        'confidence_scores': {},
                        'categories': {},
                        'metadata': {
                            'parsing_failed': True,
                            'error_message': metadata['error_message']
                        }
                    }
                else:
                    # Convert successful result
                    result_dict = parsed_result.to_json()
                    
                    # Add original job metadata
                    result_dict.update({
                        'job_id': metadata['job_id'],
                        'original_metadata': {k: v for k, v in metadata.items() 
                                            if k not in ['job_id', 'extraction_failed', 'error_message']}
                    })
                
                results.append(result_dict)
                
            except Exception as e:
                logger.error(f"Failed to process result for job {i+1}: {e}")
                # Add error result
                error_result = {
                    'job_id': metadata.get('job_id', f'job_{i+1}'),
                    'error': str(e),
                    'error_type': type(e).__name__,
                    'skills_required': [],
                    'experience_level': 'unknown',
                    'tools_mentioned': [],
                    'confidence_scores': {},
                    'categories': {},
                    'metadata': {
                        'parsing_failed': True,
                        'error_message': str(e)
                    }
                }
                results.append(error_result)
        
        # Log batch summary with performance stats
        successful_parses = sum(1 for r in results if not r.get('error'))
        failed_parses = total_jobs - successful_parses
        
        logger.info(f"Optimized batch processing completed: {successful_parses} successful, {failed_parses} failed")
        
        # Log performance statistics
        try:
            perf_stats = parser.get_performance_stats()
            logger.info(f"Performance stats: {perf_stats.get('components', {}).get('semantic_matcher', {}).get('batch_size', 'unknown')} batch size used")
        except Exception as e:
            logger.debug(f"Could not retrieve performance stats: {e}")
        
        return results
        
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        # Fallback to individual processing
        logger.info("Falling back to individual job processing")
        return process_batch_fallback(parser, job_descriptions, logger)


def process_batch_fallback(parser: JobDescriptionParser, 
                          job_descriptions: List[Dict[str, Any]], 
                          logger: logging.Logger) -> List[Dict[str, Any]]:
    """Fallback batch processing using individual job processing.
    
    Args:
        parser: JobDescriptionParser instance
        job_descriptions: List of job description dictionaries
        logger: Logger instance
        
    Returns:
        List of parsing results
    """
    results = []
    total_jobs = len(job_descriptions)
    
    logger.info(f"Using fallback individual processing for {total_jobs} job descriptions")
    
    for i, job_data in enumerate(job_descriptions, 1):
        # Extract job ID if available
        job_id = job_data.get('id', f"job_{i}")
        
        logger.info(f"Processing job {i}/{total_jobs}: {job_id}")
        
        try:
            result = process_single_job(parser, job_data, job_id, logger)
            results.append(result)
            
        except Exception as e:
            logger.error(f"Unexpected error processing job {job_id}: {e}")
            # Add error result to maintain batch consistency
            error_result = {
                'job_id': job_id,
                'error': str(e),
                'error_type': type(e).__name__,
                'skills_required': [],
                'experience_level': 'unknown',
                'tools_mentioned': [],
                'confidence_scores': {},
                'categories': {},
                'metadata': {
                    'parsing_failed': True,
                    'error_message': str(e)
                }
            }
            results.append(error_result)
    
    # Log batch summary
    successful_parses = sum(1 for r in results if not r.get('error'))
    failed_parses = total_jobs - successful_parses
    
    logger.info(f"Fallback batch processing completed: {successful_parses} successful, {failed_parses} failed")
    
    return results


def save_output(output_data: Any, output_file: Optional[Path], logger: logging.Logger) -> None:
    """Save output data to file or print to stdout.
    
    Args:
        output_data: Data to output (string or dict/list for JSON)
        output_file: Optional output file path
        logger: Logger instance
    """
    if output_file:
        try:
            logger.info(f"Saving output to: {output_file}")
            
            # Ensure output directory exists
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                if isinstance(output_data, str):
                    f.write(output_data)
                else:
                    json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Output saved successfully to {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to save output to {output_file}: {e}")
            raise
    else:
        # Print to stdout
        if isinstance(output_data, str):
            print(output_data)
        else:
            print(json.dumps(output_data, indent=2, ensure_ascii=False))


def create_parser_config(args: argparse.Namespace) -> ParserConfig:
    """Create parser configuration from command line arguments.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        ParserConfig instance
    """
    config = ParserConfig()
    
    # Update configuration based on CLI arguments
    if hasattr(args, 'ontology') and args.ontology:
        config.ontology_path = args.ontology
    
    if hasattr(args, 'threshold') and args.threshold is not None:
        config.similarity_threshold = args.threshold
    
    if hasattr(args, 'max_length') and args.max_length:
        config.max_text_length = args.max_length
    
    if hasattr(args, 'disable_ner') and args.disable_ner:
        config.enable_ner = False
    
    if hasattr(args, 'disable_semantic') and args.disable_semantic:
        config.enable_semantic = False
    
    return config


def main():
    """Main CLI entry point with command-line argument parsing."""
    
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Job Description Parser - Extract skills and requirements from job descriptions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Parse single job description from JSON file
  python main.py input.json
  
  # Parse with custom output format and file
  python main.py input.json -o output.json -f json --pretty
  
  # Batch process multiple job descriptions
  python main.py jobs.json -o results.json --batch
  
  # Parse with custom configuration
  python main.py input.json --threshold 0.8 --ontology custom_skills.csv
  
  # Generate summary output
  python main.py input.json -f summary
        """
    )
    
    # Input arguments
    parser.add_argument(
        'input_file',
        type=Path,
        help='Input JSON file containing job description(s)'
    )
    
    # Output arguments
    parser.add_argument(
        '-o', '--output',
        type=Path,
        help='Output file path (default: stdout)'
    )
    
    parser.add_argument(
        '-f', '--format',
        choices=['json', 'summary'],
        default='json',
        help='Output format (default: json)'
    )
    
    parser.add_argument(
        '--pretty',
        action='store_true',
        help='Pretty-print JSON output with indentation'
    )
    
    # Processing arguments
    parser.add_argument(
        '--batch',
        action='store_true',
        help='Enable batch processing for multiple job descriptions'
    )
    
    # Parser configuration arguments
    parser.add_argument(
        '--ontology',
        type=Path,
        help='Path to custom skills ontology CSV file'
    )
    
    parser.add_argument(
        '--threshold',
        type=float,
        metavar='FLOAT',
        help='Similarity threshold for skill matching (0.0-1.0, default: 0.7)'
    )
    
    parser.add_argument(
        '--max-length',
        type=int,
        metavar='WORDS',
        help='Maximum input text length in words (default: 2000)'
    )
    
    parser.add_argument(
        '--disable-ner',
        action='store_true',
        help='Disable Named Entity Recognition'
    )
    
    parser.add_argument(
        '--disable-semantic',
        action='store_true',
        help='Disable semantic matching'
    )
    
    # Performance arguments
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        metavar='SIZE',
        help='Batch size for processing (default: 32)'
    )
    
    parser.add_argument(
        '--max-memory',
        type=float,
        default=512.0,
        metavar='MB',
        help='Maximum memory usage in MB (default: 512.0)'
    )
    
    parser.add_argument(
        '--disable-caching',
        action='store_true',
        help='Disable model and embedding caching'
    )
    
    parser.add_argument(
        '--performance-stats',
        action='store_true',
        help='Show detailed performance statistics'
    )
    
    # Logging arguments
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging output'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='Job Description Parser 1.0.0'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_cli_logging(args.verbose)
    
    try:
        # Validate arguments
        if args.threshold is not None and not (0.0 <= args.threshold <= 1.0):
            parser.error("Threshold must be between 0.0 and 1.0")
        
        if args.max_length is not None and args.max_length <= 0:
            parser.error("Max length must be positive")
        
        if args.disable_ner and args.disable_semantic:
            parser.error("Cannot disable both NER and semantic matching")
        
        # Load job descriptions from input file
        job_descriptions = load_job_descriptions_from_file(args.input_file, logger)
        
        # Create parser configuration
        config = create_parser_config(args)
        
        # Initialize parser
        logger.info("Initializing Job Description Parser")
        job_parser = JobDescriptionParser(config)
        
        # Process job descriptions
        if args.batch or len(job_descriptions) > 1:
            # Batch processing
            logger.info("Using batch processing mode")
            results = process_batch(job_parser, job_descriptions, logger)
            
            # Format output
            if args.format == 'json':
                output_data = results
            else:  # summary format
                # Create batch summary
                total_jobs = len(results)
                successful = sum(1 for r in results if not r.get('error'))
                failed = total_jobs - successful
                
                summary_lines = [
                    f"=== Batch Processing Summary ===",
                    f"Total Jobs Processed: {total_jobs}",
                    f"Successful Parses: {successful}",
                    f"Failed Parses: {failed}",
                    ""
                ]
                
                if successful > 0:
                    # Aggregate statistics
                    all_skills = []
                    all_tools = []
                    experience_levels = []
                    
                    for result in results:
                        if not result.get('error'):
                            all_skills.extend(result.get('skills_required', []))
                            all_tools.extend(result.get('tools_mentioned', []))
                            exp_level = result.get('experience_level')
                            if exp_level and exp_level != 'unknown':
                                experience_levels.append(exp_level)
                    
                    # Count unique skills and tools
                    unique_skills = list(set(all_skills))
                    unique_tools = list(set(all_tools))
                    
                    summary_lines.extend([
                        f"Unique Skills Found: {len(unique_skills)}",
                        f"Unique Tools Found: {len(unique_tools)}",
                        ""
                    ])
                    
                    if experience_levels:
                        from collections import Counter
                        exp_counts = Counter(experience_levels)
                        summary_lines.append("Experience Level Distribution:")
                        for level, count in exp_counts.most_common():
                            summary_lines.append(f"  {level}: {count}")
                        summary_lines.append("")
                    
                    # Show most common skills
                    if all_skills:
                        from collections import Counter
                        skill_counts = Counter(all_skills)
                        summary_lines.append("Most Common Skills:")
                        for skill, count in skill_counts.most_common(10):
                            summary_lines.append(f"  {skill}: {count}")
                
                output_data = "\n".join(summary_lines)
        
        else:
            # Single job processing
            logger.info("Using single job processing mode")
            job_data = job_descriptions[0]
            job_id = job_data.get('id', 'single_job')
            
            result = process_single_job(job_parser, job_data, job_id, logger)
            
            # Format output
            if args.format == 'json':
                output_data = result
            else:  # summary format
                output_data = format_output(result, args.format, args.pretty)
        
        # Save or print output
        save_output(output_data, args.output, logger)
        
        logger.info("Processing completed successfully")
        
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        sys.exit(1)
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        sys.exit(1)
        
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON format: {e}")
        sys.exit(1)
        
    except InputValidationError as e:
        logger.error(f"Input validation error: {e}")
        sys.exit(1)
        
    except JobParserError as e:
        logger.error(f"Parser error: {e}")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if args.verbose:
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)


if __name__ == '__main__':
    main()