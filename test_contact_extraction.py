#!/usr/bin/env python3
"""
Test script for updated contact extraction with Indian phone patterns.
"""

from resume_parser.contact_extractor import ContactExtractor


def test_indian_phone_extraction():
    """Test Indian phone number extraction."""
    extractor = ContactExtractor()
    
    test_cases = [
        # Indian phone formats
        ("Contact: +91-98765-43210", "+91-98765-43210"),
        ("Phone: +91 9876543210", "+91-98765-43210"),
        ("Mobile: 919876543210", "+91-98765-43210"),
        ("Call: 9876543210", "+91-98765-43210"),
        ("Tel: +91-8765432109", "+91-87654-32109"),
        ("Contact: 7654321098", "+91-76543-21098"),
        
        # US phone formats (fallback)
        ("Phone: (555) 123-4567", "5551234567"),
        ("Call: 555-123-4567", "5551234567"),
        
        # Invalid cases
        ("Phone: 1234567890", ""),  # Doesn't start with 6-9 for Indian
        ("Contact: 12345", ""),     # Too short
    ]
    
    print("Testing Indian phone number extraction:")
    print("=" * 50)
    
    for i, (text, expected) in enumerate(test_cases, 1):
        contact_info = extractor.extract_contact_info(text)
        actual = contact_info.phone
        
        status = "✓" if actual == expected else "✗"
        print(f"{status} Test {i}: '{text}' -> '{actual}' (expected: '{expected}')")
        
        if actual != expected:
            print(f"   MISMATCH: Got '{actual}', expected '{expected}'")


def test_pincode_extraction():
    """Test pin code/ZIP code extraction."""
    extractor = ContactExtractor()
    
    test_cases = [
        # Indian pin codes (6 digits)
        ("Address: Mumbai 400001", "400001"),
        ("Location: Delhi, 110001", "110001"),
        ("Pin: 560001 Bangalore", "560001"),
        
        # US ZIP codes (5 digits)
        ("Address: New York 10001", "10001"),
        ("ZIP: 90210 California", "90210"),
        ("Location: 12345-6789", "12345-6789"),
        
        # Invalid cases
        ("Address: 1234", ""),      # Too short
        ("Location: 1234567", ""),  # Too long (not standard)
        ("No numbers here", ""),    # No pin code
    ]
    
    print("\nTesting pin code/ZIP code extraction:")
    print("=" * 50)
    
    for i, (text, expected) in enumerate(test_cases, 1):
        contact_info = extractor.extract_contact_info(text)
        actual = contact_info.address
        
        status = "✓" if actual == expected else "✗"
        print(f"{status} Test {i}: '{text}' -> '{actual}' (expected: '{expected}')")
        
        if actual != expected:
            print(f"   MISMATCH: Got '{actual}', expected '{expected}'")


def test_complete_contact_extraction():
    """Test complete contact extraction with Indian resume."""
    extractor = ContactExtractor()
    
    sample_text = """
    Rajesh Kumar
    Email: rajesh.kumar@email.com
    Mobile: +91-9876543210
    Address: Mumbai, Maharashtra 400001
    LinkedIn: https://linkedin.com/in/rajeshkumar
    """
    
    print("\nTesting complete Indian contact extraction:")
    print("=" * 50)
    
    contact_info = extractor.extract_contact_info(sample_text)
    
    print(f"Name: '{contact_info.name}'")
    print(f"Email: '{contact_info.email}'")
    print(f"Phone: '{contact_info.phone}'")
    print(f"Address: '{contact_info.address}'")
    print(f"LinkedIn: '{contact_info.linkedin}'")
    
    # Validate
    try:
        contact_info.validate()
        print("✓ Contact validation passed")
    except Exception as e:
        print(f"✗ Contact validation failed: {e}")


if __name__ == "__main__":
    test_indian_phone_extraction()
    test_pincode_extraction()
    test_complete_contact_extraction()