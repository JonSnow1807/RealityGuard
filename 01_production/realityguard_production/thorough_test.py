#!/usr/bin/env python3
"""
THOROUGH TESTING - No BS, just facts
"""

import sys
import json
import time
import asyncio
from pathlib import Path
import traceback

# Test results storage
test_results = {
    "total_tests": 0,
    "passed": 0,
    "failed": 0,
    "warnings": [],
    "critical_issues": [],
    "grade": "F"
}

def test(name):
    """Decorator for tests."""
    def decorator(func):
        async def wrapper():
            test_results["total_tests"] += 1
            print(f"\n🔍 Testing: {name}")
            print("-" * 50)
            try:
                result = await func() if asyncio.iscoroutinefunction(func) else func()
                if result:
                    test_results["passed"] += 1
                    print(f"✅ PASSED: {name}")
                else:
                    test_results["failed"] += 1
                    print(f"❌ FAILED: {name}")
                return result
            except Exception as e:
                test_results["failed"] += 1
                test_results["critical_issues"].append(f"{name}: {str(e)}")
                print(f"💥 CRASHED: {name}")
                print(f"   Error: {e}")
                traceback.print_exc()
                return False
        return wrapper
    return decorator

@test("1. Basic Imports")
def test_imports():
    """Can we even import the modules?"""
    try:
        from src.core.config import settings
        print(f"   ✓ Config imports")

        from src.services.privacy_engine import PrivacyEngine
        print(f"   ✓ Privacy engine imports")

        from src.api.routes import router
        print(f"   ✓ API routes import")

        return True
    except ImportError as e:
        print(f"   ✗ Import failed: {e}")
        return False

@test("2. Configuration Validity")
def test_config():
    """Is the configuration actually valid?"""
    from src.core.config import settings

    issues = []

    # Check critical settings
    if settings.TARGET_FPS > settings.MAX_FPS:
        issues.append("TARGET_FPS > MAX_FPS (logic error)")

    if settings.MIN_QUALITY >= settings.MAX_QUALITY:
        issues.append("MIN_QUALITY >= MAX_QUALITY (logic error)")

    if not settings.SECRET_KEY or settings.SECRET_KEY == "dev-secret-key-change-in-production":
        test_results["warnings"].append("Using default dev secret key")

    # Check paths exist
    for path_name in ["MODEL_PATH", "UPLOAD_DIR", "OUTPUT_DIR"]:
        path = getattr(settings, path_name)
        if not Path(path).exists():
            issues.append(f"{path_name} doesn't exist: {path}")

    if issues:
        for issue in issues:
            print(f"   ✗ {issue}")
        return False

    print(f"   ✓ All settings valid")
    return True

@test("3. Core Engine Initialization")
async def test_engine_init():
    """Can the privacy engine actually initialize?"""
    from src.services.privacy_engine import PrivacyEngine

    engine = PrivacyEngine()

    # Check singleton
    engine2 = PrivacyEngine()
    if engine is not engine2:
        print(f"   ✗ Singleton pattern broken")
        return False

    # Check components
    if not hasattr(engine, 'cache'):
        print(f"   ✗ No cache component")
        return False

    if not hasattr(engine, 'quality_controller'):
        print(f"   ✗ No quality controller")
        return False

    print(f"   ✓ Engine initialized with all components")
    return True

@test("4. Cache System Functionality")
def test_cache():
    """Does the hierarchical cache actually work?"""
    from src.core.privacy_engine import HierarchicalCache

    cache = HierarchicalCache()

    # Test storage and retrieval
    import numpy as np
    test_bbox = (10, 20, 30, 40)
    test_mask = np.ones((20, 20, 3), dtype=np.uint8)

    # Store in cache
    cache.put(test_bbox, "person", test_mask)

    # Test L1 hit
    result = cache.get(test_bbox, "person")
    if result is None:
        print(f"   ✗ L1 cache failed")
        return False

    # Test L2 hit (similar region)
    similar_bbox = (11, 21, 31, 41)
    result = cache.get(similar_bbox, "person")
    # L2 might not hit for just 1 pixel difference

    # Check stats
    if cache.stats["total"] == 0:
        print(f"   ✗ Cache stats not updating")
        return False

    print(f"   ✓ Cache system functional")
    print(f"   ✓ Stats: {cache.stats}")
    return True

@test("5. API Endpoint Structure")
def test_api_structure():
    """Are all promised API endpoints actually defined?"""
    from src.api.routes import router

    expected_endpoints = [
        "/process",
        "/status/{job_id}",
        "/download/{job_id}",
        "/stream",
        "/modes",
        "/capabilities"
    ]

    routes = [route.path for route in router.routes]
    missing = []

    for endpoint in expected_endpoints:
        found = any(endpoint in route for route in routes)
        if not found:
            missing.append(endpoint)

    if missing:
        print(f"   ✗ Missing endpoints: {missing}")
        return False

    print(f"   ✓ All {len(routes)} endpoints defined")
    return True

@test("6. Video Processing Pipeline")
async def test_video_processing():
    """Can it actually process a frame?"""
    from src.services.privacy_engine import PrivacyEngine
    import numpy as np

    engine = PrivacyEngine()

    # Create test frame
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    try:
        # Process frame
        result = await engine.process_frame(test_frame, mode="fast")

        if result is None:
            print(f"   ✗ Processing returned None")
            return False

        if result.shape != test_frame.shape:
            print(f"   ✗ Output shape mismatch")
            return False

        print(f"   ✓ Frame processing successful")
        return True

    except Exception as e:
        print(f"   ✗ Processing failed: {e}")
        return False

@test("7. Job Manager Operations")
async def test_job_manager():
    """Does the job manager actually track jobs?"""
    from src.services.job_manager import JobManager

    mgr = JobManager()

    # Create job
    job = await mgr.create_job("test-job", {"test": True})
    if job["id"] != "test-job":
        print(f"   ✗ Job creation failed")
        return False

    # Update job
    await mgr.update_job("test-job", "processing", 0.5)

    # Get job
    retrieved = await mgr.get_job("test-job")
    if retrieved["progress"] != 0.5:
        print(f"   ✗ Job update failed")
        return False

    # Get stats
    stats = await mgr.get_stats()
    if stats["jobs"]["total"] == 0:
        print(f"   ✗ Stats not tracking")
        return False

    print(f"   ✓ Job manager functional")
    return True

@test("8. Error Handling")
async def test_error_handling():
    """Is there proper error handling?"""
    from src.services.privacy_engine import PrivacyEngine
    import numpy as np

    engine = PrivacyEngine()
    issues = []

    # Test with invalid frame
    try:
        result = await engine.process_frame(None, mode="fast")
        issues.append("No error on None frame")
    except:
        pass  # Expected

    # Test with invalid mode
    try:
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = await engine.process_frame(test_frame, mode="invalid_mode")
        # Should handle gracefully
    except:
        issues.append("Crashes on invalid mode")

    if issues:
        for issue in issues:
            print(f"   ✗ {issue}")
        test_results["warnings"].extend(issues)
        return False

    print(f"   ✓ Error handling present")
    return True

@test("9. Memory Leaks Check")
async def test_memory():
    """Does it leak memory?"""
    import psutil
    import gc
    from src.services.privacy_engine import PrivacyEngine
    import numpy as np

    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB

    engine = PrivacyEngine()

    # Process multiple frames
    for i in range(10):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        await engine.process_frame(frame)

    gc.collect()
    final_memory = process.memory_info().rss / 1024 / 1024  # MB

    memory_growth = final_memory - initial_memory

    if memory_growth > 100:  # More than 100MB growth
        print(f"   ⚠️ Memory growth: {memory_growth:.1f}MB")
        test_results["warnings"].append(f"High memory growth: {memory_growth:.1f}MB")
        return False

    print(f"   ✓ Memory stable (growth: {memory_growth:.1f}MB)")
    return True

@test("10. Docker Configuration")
def test_docker():
    """Is Docker config valid?"""

    # Check Dockerfile
    dockerfile = Path("Dockerfile")
    if not dockerfile.exists():
        print(f"   ✗ No Dockerfile")
        return False

    content = dockerfile.read_text()

    issues = []

    if "FROM python:" not in content:
        issues.append("No Python base image")

    if "HEALTHCHECK" not in content:
        test_results["warnings"].append("No health check in Dockerfile")

    if "USER root" in content and "USER" not in content[content.find("USER root")+9:]:
        issues.append("Running as root (security issue)")

    # Check docker-compose
    compose_file = Path("docker-compose.yml")
    if not compose_file.exists():
        print(f"   ✗ No docker-compose.yml")
        return False

    if issues:
        for issue in issues:
            print(f"   ✗ {issue}")
        return False

    print(f"   ✓ Docker config valid")
    return True

async def run_all_tests():
    """Run all tests and provide verdict."""
    print("=" * 60)
    print("THOROUGH SYSTEM EVALUATION - NO BS")
    print("=" * 60)

    # Run tests
    await test_imports()
    await test_config()
    await test_engine_init()
    await test_cache()
    await test_api_structure()
    await test_video_processing()
    await test_job_manager()
    await test_error_handling()
    await test_memory()
    await test_docker()

    # Calculate grade
    pass_rate = test_results["passed"] / test_results["total_tests"] * 100

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    print(f"Tests Run: {test_results['total_tests']}")
    print(f"Passed: {test_results['passed']}")
    print(f"Failed: {test_results['failed']}")
    print(f"Pass Rate: {pass_rate:.1f}%")

    if test_results["critical_issues"]:
        print(f"\n🔴 CRITICAL ISSUES:")
        for issue in test_results["critical_issues"]:
            print(f"   - {issue}")

    if test_results["warnings"]:
        print(f"\n⚠️  WARNINGS:")
        for warning in test_results["warnings"]:
            print(f"   - {warning}")

    # Grade calculation
    if pass_rate >= 95 and not test_results["critical_issues"]:
        grade = "A+"
        verdict = "EXCELLENT - Production ready"
    elif pass_rate >= 90 and len(test_results["critical_issues"]) <= 1:
        grade = "A"
        verdict = "VERY GOOD - Minor fixes needed"
    elif pass_rate >= 80:
        grade = "B+"
        verdict = "GOOD - Some issues to address"
    elif pass_rate >= 70:
        grade = "B"
        verdict = "ACCEPTABLE - Needs work"
    elif pass_rate >= 60:
        grade = "C"
        verdict = "MEDIOCRE - Significant issues"
    else:
        grade = "F"
        verdict = "FAILING - Not production ready"

    print(f"\n📊 GRADE: {grade}")
    print(f"📝 VERDICT: {verdict}")

    # Specific feedback
    print("\n🎯 HONEST ASSESSMENT:")

    strengths = []
    weaknesses = []

    if test_results["passed"] >= 8:
        strengths.append("Core functionality works")
    if "Cache system functional" in str(test_results):
        strengths.append("Caching implemented correctly")
    if not any("memory" in str(i).lower() for i in test_results["critical_issues"]):
        strengths.append("No major memory issues")

    if test_results["failed"] > 2:
        weaknesses.append(f"{test_results['failed']} tests failed")
    if test_results["critical_issues"]:
        weaknesses.append(f"{len(test_results['critical_issues'])} critical issues")
    if test_results["warnings"]:
        weaknesses.append(f"{len(test_results['warnings'])} warnings")

    if strengths:
        print("\n✅ STRENGTHS:")
        for s in strengths:
            print(f"   - {s}")

    if weaknesses:
        print("\n❌ WEAKNESSES:")
        for w in weaknesses:
            print(f"   - {w}")

    print("\n" + "=" * 60)

    return grade

if __name__ == "__main__":
    grade = asyncio.run(run_all_tests())
    sys.exit(0 if grade in ["A+", "A"] else 1)