#!/usr/bin/env python3
"""Test the FaceX angle mapping function independently."""

import sys
sys.path.insert(0, '.')

from src.mqtt_bridge import angle_for
from types import SimpleNamespace


def test_angle_mapping():
    """Verify angle_for maps error_x correctly to needle angles."""
    
    test_cases = [
        ("Far LEFT", -1.0, 0.0),
        ("LEFT", -0.8, 18.0),
        ("Left", -0.4, 54.0),
        ("CENTER", 0.0, 90.0),
        ("Right", 0.4, 126.0),
        ("RIGHT", 0.8, 162.0),
        ("Far RIGHT", 1.0, 180.0),
    ]
    
    print("Testing angle_for(position) function")
    print("=" * 60)
    print(f"{'Position':<15} {'error_x':>8} {'Expected':>10} {'Actual':>10} {'Status':>8}")
    print("-" * 60)
    
    all_passed = True
    
    for label, error_x, expected_angle in test_cases:
        position = SimpleNamespace(error_x=error_x)
        actual_angle = angle_for(position)
        
        tolerance = 0.1
        passed = abs(actual_angle - expected_angle) < tolerance
        all_passed = all_passed and passed
        
        status = "PASS" if passed else "FAIL"
        print(f"{label:<15} {error_x:>8.1f} {expected_angle:>10.1f} {actual_angle:>10.1f} {status:>8}")
    
    print("-" * 60)
    
    # Test None position
    angle_none = angle_for(None)
    print(f"{'None position':<15} {'N/A':>8} {90.0:>10.1f} {angle_none:>10.1f} {'PASS' if angle_none == 90.0 else 'FAIL':>8}")
    
    print("=" * 60)
    
    if all_passed:
        print("\n✓ All angle mapping tests PASSED")
        print("\nExpected behavior:")
        print("  - Face moves LEFT  (error_x < 0) → angle decreases (< 90)")
        print("  - Face at CENTER   (error_x = 0) → angle = 90")
        print("  - Face moves RIGHT (error_x > 0) → angle increases (> 90)")
        return 0
    else:
        print("\n✗ Some tests FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(test_angle_mapping())
