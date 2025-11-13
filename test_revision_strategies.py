"""
Test script to demonstrate each revision strategy with max_revision_iterations=2.
Shows how RE_RETRIEVAL, CONSTRAINED_GENERATION, and CLAIM_BY_CLAIM work.

Usage:
    python test_revision_strategies.py

This script tests:
1. RE_RETRIEVAL strategy (triggered when entailment_rate < 0.5)
   - Expands query with failed claims
   - Re-retrieves with expanded query
   - Re-generates with new contexts

2. CONSTRAINED_GENERATION strategy (triggered when 0.5 <= entailment_rate < 0.8)
   - Uses verified claims as constraints in the prompt
   - Generates with verified facts included

3. CLAIM_BY_CLAIM strategy (triggered when entailment_rate >= 0.8)
   - Regenerates only unverified claims
   - Uses focused queries for each unverified claim
   - Reconstructs with verified + revised claims

4. Forced strategy tests (manually triggers each strategy)

Output:
    - Console output showing each strategy's behavior
    - JSON file: test_revision_strategies_results.json with detailed results
"""

import sys
from pathlib import Path
import json

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.pipeline import SelfVerificationRAGPipeline
from src.revision import AdaptiveRevisionStrategy, RevisionStrategy


def create_test_corpus():
    """Create a simple test corpus."""
    return [
        "The University of Florida was founded in 1853. It is located in Gainesville, Florida.",
        "UF is a public research university with over 50,000 students enrolled.",
        "The university's mascot is Albert the Alligator and the school colors are orange and blue.",
        "UF offers over 100 undergraduate degree programs and 200 graduate programs.",
        "The university is known for its strong programs in engineering, business, and agriculture.",
        "Gainesville is a city in north-central Florida with a population of approximately 141,000.",
        "The city is home to several museums, parks, and cultural attractions.",
        "Florida is a state in the southeastern United States, bordered by the Gulf of Mexico and the Atlantic Ocean."
    ]


def test_re_retrieval_strategy():
    """Test RE_RETRIEVAL strategy (triggered when entailment_rate < 0.5)."""
    print("\n" + "="*80)
    print("TEST 1: RE_RETRIEVAL STRATEGY")
    print("="*80)
    print("Strategy: Expands query with failed claims, re-retrieves, then re-generates")
    print("Triggered when: entailment_rate < 0.5")
    print("-"*80)
    
    corpus = create_test_corpus()
    
    # Initialize pipeline with revision enabled
    pipeline = SelfVerificationRAGPipeline(
        corpus=corpus,
        device="cuda",
        enable_revision=True,
        max_revision_iterations=2,
        use_qlora=False  # Use base model for testing
    )
    
    # Query that will likely have low entailment (hallucinated answer)
    query = "What is the population of Gainesville and when was UF founded?"
    
    print(f"\nQuery: {query}")
    print("\nRunning pipeline with RE_RETRIEVAL strategy...")
    
    result = pipeline.generate(query, top_k_retrieve=5, top_k_rerank=3)
    
    print(f"\n✓ Generated Text: {result['generated_text']}")
    print(f"✓ Revision Iterations: {result['revision_iterations']}")
    print(f"✓ Verified: {result['verified']}")
    print(f"✓ Abstained: {result.get('abstained', False)}")
    
    if result.get('revision_history'):
        print(f"\n📋 Revision History ({len(result['revision_history'])} iterations):")
        for i, rev in enumerate(result['revision_history'], 1):
            print(f"\n  Iteration {rev['iteration']}:")
            print(f"    Strategy: {rev.get('strategy', 'unknown')}")
            if rev.get('expanded_query'):
                print(f"    Expanded Query: {rev['expanded_query']}")
            if rev.get('failed_claims_used'):
                print(f"    Failed Claims Used: {rev['failed_claims_used']}")
            if rev.get('prompt_used'):
                print(f"    Prompt Used: {rev['prompt_used'][:150]}...")
            print(f"    Generation Before: {rev['generation_before'][:100]}...")
            print(f"    Generation After: {rev['generation_after'][:100]}...")
    
    return result


def test_constrained_generation_strategy():
    """Test CONSTRAINED_GENERATION strategy (triggered when 0.5 <= entailment_rate < 0.8)."""
    print("\n" + "="*80)
    print("TEST 2: CONSTRAINED_GENERATION STRATEGY")
    print("="*80)
    print("Strategy: Uses verified claims as constraints in the prompt")
    print("Triggered when: 0.5 <= entailment_rate < 0.8")
    print("-"*80)
    
    corpus = create_test_corpus()
    
    # Initialize pipeline with revision enabled
    pipeline = SelfVerificationRAGPipeline(
        corpus=corpus,
        device="cuda",
        enable_revision=True,
        max_revision_iterations=2,
        use_qlora=False
    )
    
    # Query that might have medium entailment (some claims verified, some not)
    query = "What are the school colors and mascot of UF?"
    
    print(f"\nQuery: {query}")
    print("\nRunning pipeline with CONSTRAINED_GENERATION strategy...")
    
    result = pipeline.generate(query, top_k_retrieve=5, top_k_rerank=3)
    
    print(f"\n✓ Generated Text: {result['generated_text']}")
    print(f"✓ Revision Iterations: {result['revision_iterations']}")
    print(f"✓ Verified: {result['verified']}")
    print(f"✓ Abstained: {result.get('abstained', False)}")
    
    if result.get('revision_history'):
        print(f"\n📋 Revision History ({len(result['revision_history'])} iterations):")
        for i, rev in enumerate(result['revision_history'], 1):
            print(f"\n  Iteration {rev['iteration']}:")
            print(f"    Strategy: {rev.get('strategy', 'unknown')}")
            if rev.get('verified_claims'):
                print(f"    Verified Claims Used: {rev['verified_claims']}")
            if rev.get('prompt_used'):
                print(f"    Prompt Used: {rev['prompt_used'][:200]}...")
            print(f"    Generation Before: {rev['generation_before'][:100]}...")
            print(f"    Generation After: {rev['generation_after'][:100]}...")
    
    return result


def test_claim_by_claim_strategy():
    """Test CLAIM_BY_CLAIM strategy (triggered when entailment_rate >= 0.8)."""
    print("\n" + "="*80)
    print("TEST 3: CLAIM_BY_CLAIM STRATEGY")
    print("="*80)
    print("Strategy: Regenerates only unverified claims with focused queries")
    print("Triggered when: entailment_rate >= 0.8")
    print("-"*80)
    
    corpus = create_test_corpus()
    
    # Initialize pipeline with revision enabled
    pipeline = SelfVerificationRAGPipeline(
        corpus=corpus,
        device="cuda",
        enable_revision=True,
        max_revision_iterations=2,
        use_qlora=False
    )
    
    # Query that might have high but not perfect entailment
    query = "Tell me about UF's location, founding year, and student enrollment."
    
    print(f"\nQuery: {query}")
    print("\nRunning pipeline with CLAIM_BY_CLAIM strategy...")
    
    result = pipeline.generate(query, top_k_retrieve=5, top_k_rerank=3)
    
    print(f"\n✓ Generated Text: {result['generated_text']}")
    print(f"✓ Revision Iterations: {result['revision_iterations']}")
    print(f"✓ Verified: {result['verified']}")
    print(f"✓ Abstained: {result.get('abstained', False)}")
    
    if result.get('revision_history'):
        print(f"\n📋 Revision History ({len(result['revision_history'])} iterations):")
        for i, rev in enumerate(result['revision_history'], 1):
            print(f"\n  Iteration {rev['iteration']}:")
            print(f"    Strategy: {rev.get('strategy', 'unknown')}")
            if rev.get('unverified_claims'):
                print(f"    Unverified Claims: {rev['unverified_claims']}")
            if rev.get('verified_claims'):
                print(f"    Verified Claims (preserved): {rev['verified_claims']}")
            if rev.get('claim_queries'):
                print(f"    Claim Queries Used: {rev['claim_queries']}")
            if rev.get('prompt_used'):
                if isinstance(rev['prompt_used'], list):
                    print(f"    Prompts Used ({len(rev['prompt_used'])}):")
                    for j, prompt in enumerate(rev['prompt_used'], 1):
                        print(f"      {j}. {prompt[:150]}...")
                else:
                    print(f"    Prompt Used: {rev['prompt_used'][:200]}...")
            print(f"    Generation Before: {rev['generation_before'][:100]}...")
            print(f"    Generation After: {rev['generation_after'][:100]}...")
    
    return result


def test_all_strategies_forced():
    """Test all strategies by manually creating scenarios that trigger each one."""
    print("\n" + "="*80)
    print("TEST 4: FORCING EACH STRATEGY (Manual Strategy Selection)")
    print("="*80)
    print("This test manually triggers each strategy to show their behavior")
    print("-"*80)
    
    corpus = create_test_corpus()
    
    strategies_to_test = [
        ("re_retrieval", "What is the exact date UF was established and its current enrollment?"),
        ("constrained_generation", "What are UF's colors and what is its mascot?"),
        ("claim_by_claim", "Where is UF located and when was it founded?")
    ]
    
    results = {}
    
    for strategy_name, query in strategies_to_test:
        print(f"\n{'='*80}")
        print(f"Testing with Query (may trigger {strategy_name.upper()}):")
        print(f"Query: {query}")
        print(f"{'='*80}")
        
        # Initialize pipeline for each test
        pipeline = SelfVerificationRAGPipeline(
            corpus=corpus,
            device="cuda",
            enable_revision=True,
            max_revision_iterations=2,
            use_qlora=False
        )
        
        # Temporarily override strategy selection to force specific strategy
        original_select = pipeline.revision_strategy._select_strategy
        
        def make_force_strategy(target_strategy):
            def force_strategy(verification_results, iteration):
                if target_strategy == "re_retrieval":
                    return RevisionStrategy.RE_RETRIEVAL
                elif target_strategy == "constrained_generation":
                    return RevisionStrategy.CONSTRAINED_GENERATION
                elif target_strategy == "claim_by_claim":
                    return RevisionStrategy.CLAIM_BY_CLAIM
                return original_select(verification_results, iteration)
            return force_strategy
        
        pipeline.revision_strategy._select_strategy = make_force_strategy(strategy_name)
        
        result = pipeline.generate(query, top_k_retrieve=5, top_k_rerank=3)
        results[strategy_name] = result
        
        print(f"\n✓ Generated: {result['generated_text'][:150]}...")
        if result.get('revision_history'):
            rev = result['revision_history'][0]
            print(f"✓ Strategy Used: {rev.get('strategy', 'none')}")
        else:
            print(f"✓ Strategy Used: none (no revision needed)")
        print(f"✓ Iterations: {result['revision_iterations']}")
        
        if result.get('revision_history'):
            rev = result['revision_history'][0]
            print(f"\n  Strategy Metadata:")
            print(f"    - Strategy: {rev.get('strategy')}")
            if rev.get('prompt_used'):
                prompt = rev['prompt_used']
                if isinstance(prompt, list):
                    print(f"    - Prompts: {len(prompt)} focused prompts")
                    for i, p in enumerate(prompt[:2], 1):  # Show first 2
                        print(f"      {i}. {p[:100]}...")
                else:
                    print(f"    - Prompt: {prompt[:150]}...")
            if rev.get('expanded_query'):
                print(f"    - Expanded Query: {rev['expanded_query']}")
            if rev.get('verified_claims'):
                print(f"    - Verified Claims: {rev['verified_claims']}")
            if rev.get('claim_queries'):
                print(f"    - Claim Queries: {rev['claim_queries']}")
    
    return results


def save_results_to_json(results_dict, filename="test_revision_strategies_results.json"):
    """Save test results to JSON file."""
    # Convert results to JSON-serializable format
    json_results = {}
    for key, result in results_dict.items():
        json_results[key] = {
            "query": result.get("query", ""),
            "generated_text": result.get("generated_text", ""),
            "revision_iterations": result.get("revision_iterations", 0),
            "verified": result.get("verified", False),
            "abstained": result.get("abstained", False),
            "revision_history": result.get("revision_history", [])
        }
    
    with open(filename, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\n✓ Results saved to {filename}")


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("REVISION STRATEGIES TEST SCRIPT")
    print("="*80)
    print("Testing all three revision strategies with max_revision_iterations=2")
    print("="*80)
    
    results = {}
    
    try:
        # Test each strategy (they will be auto-selected based on entailment rate)
        print("\n[Note: Strategies are auto-selected based on entailment rate]")
        print("[To force specific strategies, see test_all_strategies_forced()]\n")
        
        # Test 1: RE_RETRIEVAL (low entailment)
        results['re_retrieval'] = test_re_retrieval_strategy()
        
        # Test 2: CONSTRAINED_GENERATION (medium entailment)
        results['constrained_generation'] = test_constrained_generation_strategy()
        
        # Test 3: CLAIM_BY_CLAIM (high entailment)
        results['claim_by_claim'] = test_claim_by_claim_strategy()
        
        # Test 4: Force each strategy
        forced_results = test_all_strategies_forced()
        results['forced_strategies'] = forced_results
        
        # Save results
        save_results_to_json(results)
        
        print("\n" + "="*80)
        print("ALL TESTS COMPLETED")
        print("="*80)
        print("\nSummary:")
        for key, result in results.items():
            if isinstance(result, dict) and 'generated_text' in result:
                print(f"  {key}: {result['revision_iterations']} iterations, verified={result['verified']}")
            elif isinstance(result, dict):
                print(f"  {key}: {len(result)} forced strategy tests")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

