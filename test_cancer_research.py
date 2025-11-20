from quantum_bio_hybrid import QuantumBiologicalHybrid

print("🧬 TESTING QUANTUM-BIO CANCER RESEARCH")
print("=" * 50)

# Create quantum-bio AI
cancer_ai = QuantumBiologicalHybrid(
    name="CancerResearchAI", 
    research_duration=30  # Shorter for testing
)

# Start the 30-day cancer research journey!
research_results = cancer_ai.research_cancer_cure()

print(f"\n🎯 RESEARCH COMPLETE!")
print(f"   Breakthroughs: {len(research_results['major_breakthroughs'])}")
print(f"   Final Quantum Coherence: {cancer_ai.quantum_coherence:.1f}")