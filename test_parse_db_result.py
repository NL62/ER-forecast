#!/usr/bin/env python3
"""Test script for parse_db_result method."""

import re
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Test parsing logic directly
def test_parse_line():
    """Test parsing a single line."""
    line = "(904841758, datetime.datetime(2023, 1, 3, 13, 54, 38), 'Anlänt', 'Mottagningsbes                                                ök akutmottagning', 'Medicinkliniken', 'Akutmottagning Västerås', datetime.datetime(2023, 1, 3, 18, 5, 10, 617000), '904841758')"
    
    # Pattern to extract ID (first number) and first datetime
    id_pattern = r'^\((\d+)'
    datetime_pattern = r'datetime\.datetime\((\d{4}),\s*(\d{1,2}),\s*(\d{1,2}),\s*(\d{1,2}),\s*(\d{1,2}),\s*(\d{1,2})(?:,\s*(\d+))?\)'
    
    # Extract ID
    id_match = re.search(id_pattern, line)
    if id_match:
        print(f"✓ Found ID: {id_match.group(1)}")
    else:
        print("✗ Could not find ID")
        return False
    
    # Extract first datetime
    datetime_match = re.search(datetime_pattern, line)
    if datetime_match:
        year, month, day, hour, minute, second = datetime_match.groups()[:6]
        microsecond = datetime_match.groups()[6] if datetime_match.groups()[6] else '0'
        
        dt = datetime(
            int(year), int(month), int(day),
            int(hour), int(minute), int(second),
            int(microsecond)
        )
        print(f"✓ Found datetime: {dt}")
    else:
        print("✗ Could not find datetime")
        return False
    
    return True


def test_parse_file():
    """Test parsing the actual file."""
    file_path = Path("data/raw/raw_real.csv")
    
    if not file_path.exists():
        print(f"✗ File not found: {file_path}")
        return False
    
    print(f"\nTesting parsing of {file_path}...")
    
    rows = []
    id_pattern = r'^\((\d+)'
    datetime_pattern = r'datetime\.datetime\((\d{4}),\s*(\d{1,2}),\s*(\d{1,2}),\s*(\d{1,2}),\s*(\d{1,2}),\s*(\d{1,2})(?:,\s*(\d+))?\)'
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Skip first line if it's just a closing parenthesis
    start_idx = 0
    if lines and lines[0].strip() == ')':
        start_idx = 1
    
    for line_num, line in enumerate(lines[start_idx:], start=start_idx + 1):
        line = line.strip()
        
        if not line or line == ')':
            continue
        
        try:
            # Extract ID
            id_match = re.search(id_pattern, line)
            if not id_match:
                continue
            
            surrogate_key = id_match.group(1)
            
            # Extract first datetime
            datetime_match = re.search(datetime_pattern, line)
            if not datetime_match:
                continue
            
            # Parse datetime
            year, month, day, hour, minute, second = datetime_match.groups()[:6]
            microsecond = datetime_match.groups()[6] if datetime_match.groups()[6] else '0'
            
            contact_start = datetime(
                int(year), int(month), int(day),
                int(hour), int(minute), int(second),
                int(microsecond)
            )
            
            rows.append({
                'Surrogate_Key': surrogate_key,
                'Contact_Start': contact_start
            })
            
        except Exception as e:
            print(f"Warning: Error on line {line_num}: {e}")
            continue
    
    print(f"✓ Successfully parsed {len(rows)} rows")
    
    if rows:
        print(f"\nFirst row: ID={rows[0]['Surrogate_Key']}, Date={rows[0]['Contact_Start']}")
        print(f"Last row: ID={rows[-1]['Surrogate_Key']}, Date={rows[-1]['Contact_Start']}")
        print(f"\nDate range: {min(r['Contact_Start'] for r in rows)} to {max(r['Contact_Start'] for r in rows)}")
    
    return len(rows) > 0


if __name__ == "__main__":
    print("=" * 70)
    print("Testing parse_db_result parsing logic")
    print("=" * 70)
    
    print("\n1. Testing single line parsing:")
    if test_parse_line():
        print("   ✓ Single line parsing works!")
    else:
        print("   ✗ Single line parsing failed!")
        sys.exit(1)
    
    print("\n2. Testing file parsing:")
    if test_parse_file():
        print("   ✓ File parsing works!")
    else:
        print("   ✗ File parsing failed!")
        sys.exit(1)
    
    print("\n" + "=" * 70)
    print("All tests passed! ✓")
    print("=" * 70)

