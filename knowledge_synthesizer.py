from brain_advanced import AdvancedBrain
import random
import json

class KnowledgeSynthesizer(AdvancedBrain):
    def __init__(self, name="Prometheus"):
        super().__init__(name)
        self.knowledge_base = {}
        self.cross_domain_insights = []
        self.scientific_domains = [
            "physics", "biology", "mathematics", "computer_science", 
            "chemistry", "psychology", "economics", "engineering"
        ]
        
    def ingest_knowledge(self, domain, concepts):
        """Feed knowledge from different scientific domains"""
        if domain not in self.knowledge_base:
            self.knowledge_base[domain] = []
        
        self.knowledge_base[domain].extend(concepts)
        print(f"📚 Ingested {len(concepts)} concepts from {domain}")
        
        # Convert knowledge to neural inputs and learn
        self._learn_from_knowledge(domain, concepts)
        
    def _learn_from_knowledge(self, domain, concepts):
        """Convert knowledge into neural learning experiences"""
        for concept in concepts:
            # Encode concept as neural input
            concept_input = self._encode_concept(domain, concept)
            output = self.think(concept_input)
            
            # Simulate understanding and integration
            understanding_score = random.uniform(0.6, 0.95)
            self.evolve_architecture(understanding_score)
    
    def _encode_concept(self, domain, concept):
        """Encode scientific concepts as neural inputs - IMPROVED VERSION"""
        input_vector = []
        
        # Safe domain encoding
        if domain in self.scientific_domains:
            domain_code = self.scientific_domains.index(domain) / len(self.scientific_domains)
        else:
            domain_code = 0.5  # Default for unknown domains
        
        input_vector.append(domain_code)
        
        # Better concept encoding
        concept_str = str(concept)
        
        # Use multiple encoding strategies
        input_vector.append(len(concept_str) / 100)  # Length encoding
        input_vector.append(sum(ord(c) for c in concept_str[:15]) / 1500)  # Character sum encoding
        
        # Word-based features (if concept has multiple words)
        words = concept_str.split()
        input_vector.append(len(words) / 10)  # Word count encoding
        
        # Add some semantic-like features based on common scientific terms
        science_indicators = {
            "quantum": 0.9, "evolution": 0.8, "algorithm": 0.7, 
            "calculus": 0.8, "neural": 0.7, "entanglement": 0.9
        }
        
        semantic_score = 0.0
        for term, score in science_indicators.items():
            if term in concept_str.lower():
                semantic_score = max(semantic_score, score)
        
        input_vector.append(semantic_score)
        
        # Add randomness for diversity but with domain consistency
        random.seed(hash(domain + concept_str) % 10000)  # Seeded for consistency
        for _ in range(5):
            input_vector.append(random.uniform(0, 0.3))
        
        # Pad to standard input size
        while len(input_vector) < 50:
            input_vector.append(random.uniform(0, 0.1))
            
        return input_vector
    
    def generate_cross_domain_insights(self):
        """Generate insights by connecting concepts across domains - IMPROVED"""
        insights = []
        
        print(f"   Available domains: {list(self.knowledge_base.keys())}")
        print(f"   Concepts per domain: {[len(concepts) for concepts in self.knowledge_base.values()]}")
        
        # Get domains with enough concepts
        domains_with_concepts = [domain for domain in self.knowledge_base 
                               if len(self.knowledge_base[domain]) >= 2]
        
        print(f"   Domains with enough concepts: {domains_with_concepts}")
        
        if len(domains_with_concepts) >= 2:
            for attempt in range(5):  # Try 5 times to generate insights
                domain1, domain2 = random.sample(domains_with_concepts, 2)
                concept1 = random.choice(self.knowledge_base[domain1])
                concept2 = random.choice(self.knowledge_base[domain2])
                
                print(f"   Trying connection: {concept1} ({domain1}) + {concept2} ({domain2})")
                
                # Use the brain to find connections
                connection_input = self._encode_connection(concept1, concept2)
                connection_output = self.think(connection_input)
                
                insight_quality = sum(connection_output) / len(connection_output)
                print(f"   Connection quality: {insight_quality:.2f}")
                
                if insight_quality > 0.6:  # Lower threshold for more insights
                    insight = self._formulate_insight(domain1, concept1, domain2, concept2, insight_quality)
                    insights.append(insight)
                    
                    if insight_quality > 0.8:
                        self.record_breakthrough(f"Cross-domain insight: {insight[:80]}...")
                    
                    # Don't generate too many at once
                    if len(insights) >= 3:
                        break
        
        return insights
    
    def _encode_connection(self, concept1, concept2):
        """Encode two concepts for connection finding"""
        input_vector = []
        
        # Encode both concepts using a default domain
        input_vector.extend(self._encode_concept("physics", concept1)[:10])  # Use physics as default
        input_vector.extend(self._encode_concept("physics", concept2)[:10])  # Use physics as default
        
        # Add connection-seeking patterns
        input_vector.extend([0.8, 0.6, 0.9])  # Connection-seeking codes
        
        # Add some random variation to differentiate the concepts
        input_vector.append(hash(concept1) % 1000 / 1000)
        input_vector.append(hash(concept2) % 1000 / 1000)
        
        # Pad to standard size
        while len(input_vector) < 50:
            input_vector.append(random.uniform(0, 0.2))
            
        return input_vector
    
    def _formulate_insight(self, domain1, concept1, domain2, concept2, quality):
        """Formulate a cross-domain insight"""
        insight_templates = [
            f"What if we apply {concept1} from {domain1} to solve problems in {domain2}?",
            f"The principle of {concept1} ({domain1}) might explain {concept2} in {domain2}",
            f"Combining {concept1} with {concept2} could create breakthrough innovations",
            f"{domain2} could benefit from the {concept1} approach used in {domain1}"
        ]
        
        template = random.choice(insight_templates)
        return f"{template} [Quality: {quality:.2f}]"
    
    def solve_complex_problem(self, problem_description, relevant_domains):
        """Solve complex problems using integrated knowledge"""
        print(f"🔍 Solving complex problem: {problem_description}")
        print(f"   Using domains: {relevant_domains}")
        
        # Gather relevant knowledge
        relevant_concepts = []
        for domain in relevant_domains:
            if domain in self.knowledge_base:
                relevant_concepts.extend(self.knowledge_base[domain][:5])  # Use first 5 concepts
        
        # Generate solutions using cross-domain thinking
        solutions = []
        for _ in range(3):  # Generate 3 potential solutions
            solution_input = self._encode_problem(problem_description, relevant_concepts)
            solution_output = self.think(solution_input)
            
            solution_quality = sum(solution_output) / len(solution_output)
            
            solution = {
                "approach": self._generate_solution_approach(relevant_concepts),
                "quality": solution_quality,
                "domains_used": relevant_domains,
                "innovation_score": max(solution_output) - min(solution_output)  # Measure of creative thinking
            }
            solutions.append(solution)
        
        # Evolve based on best solution quality
        best_quality = max(sol["quality"] for sol in solutions)
        self.evolve_architecture(best_quality)
        
        return solutions
    
    def _encode_problem(self, problem, concepts):
        """Encode problem and concepts for solution generation - IMPROVED"""
        input_vector = []
        
        # Encode problem description
        problem_str = str(problem)
        input_vector.append(len(problem_str) / 200)
        input_vector.append(hash(problem_str) % 1000 / 1000)
        
        # Encode problem complexity indicator
        complexity_words = ["design", "system", "efficient", "optimize", "complex"]
        complexity_score = sum(1 for word in complexity_words if word in problem_str.lower()) / 5
        input_vector.append(complexity_score)
        
        # Encode concepts (up to 5)
        concept_encodings = []
        for concept in concepts[:5]:
            concept_encodings.extend(self._encode_concept("physics", concept)[:4])  # Use physics as default
        
        input_vector.extend(concept_encodings)
        
        # Add solution-seeking patterns
        input_vector.extend([0.7, 0.5, 0.8, 0.6])  # Solution generation codes
        
        # Pad to standard size
        while len(input_vector) < 50:
            input_vector.append(random.uniform(0, 0.15))
            
        return input_vector
    
    def _generate_solution_approach(self, concepts):
        """Generate solution approach based on available concepts - MORE DIVERSE"""
        if not concepts:
            return "Apply systematic analysis and iterative testing"
        
        # Use different numbers of concepts for variety
        num_concepts_to_use = random.randint(1, min(3, len(concepts)))
        used_concepts = random.sample(concepts, num_concepts_to_use)
        
        approaches = [
            f"Combine {used_concepts[0]} with computational modeling",
            f"Use {used_concepts[0]} as foundation and apply optimization algorithms",
            f"Adapt the principles of {used_concepts[0]} to this domain",
            f"Create a hybrid approach using {used_concepts[0]} and systematic experimentation",
            f"Apply {used_concepts[0]} theory to develop novel architectures",
            f"Use {used_concepts[0]} as inspiration for biomimetic design",
            f"Integrate {used_concepts[0]} with machine learning for adaptive solutions"
        ]
        
        # Add multi-concept approaches if we have enough
        if len(used_concepts) >= 2:
            approaches.extend([
                f"Synthesize {used_concepts[0]} and {used_concepts[1]} for breakthrough innovation",
                f"Bridge {used_concepts[0]} from physics with {used_concepts[1]} from engineering",
                f"Use {used_concepts[0]} as framework and {used_concepts[1]} for implementation"
            ])
        
        return random.choice(approaches)

# 🧪 TEST THE KNOWLEDGE SYNTHESIZER
if __name__ == "__main__":
    synthesizer = KnowledgeSynthesizer("Prometheus")
    
    print("🧠 KNOWLEDGE SYNTHESIZER TEST - PHASE 3")
    print("=" * 60)
    # Ingest knowledge from different domains - EXPANDED VERSION
    scientific_knowledge = {
        "physics": ["quantum entanglement", "relativity", "thermodynamics", "electromagnetism", "superconductivity"],
        "biology": ["evolution", "DNA", "cellular respiration", "neural networks", "enzyme catalysis"],
        "computer_science": ["algorithms", "machine learning", "neural networks", "optimization", "data structures"],
        "mathematics": ["calculus", "probability", "graph theory", "linear algebra", "complex systems"],
        "engineering": ["systems design", "materials science", "energy efficiency", "control theory"]
    }

    
    for domain, concepts in scientific_knowledge.items():
        synthesizer.ingest_knowledge(domain, concepts)
    
    print(f"\n📊 After knowledge ingestion:")
    stats = synthesizer.get_brain_stats()
    print(f"   Generation: {stats['generation']}")
    print(f"   Neurons: {stats['total_neurons']}")
    print(f"   Connections: {stats['total_connections']}")
    print(f"   Skills: {synthesizer.skills_developed}")
    
    # Generate cross-domain insights
    print(f"\n💡 GENERATING CROSS-DOMAIN INSIGHTS:")
    insights = synthesizer.generate_cross_domain_insights()
    for i, insight in enumerate(insights, 1):
        print(f"   {i}. {insight}")
    
    # Solve complex problems
    print(f"\n🔧 SOLVING COMPLEX PROBLEMS:")
    complex_problem = "Design an efficient energy storage system"
    relevant_domains = ["physics", "engineering", "chemistry"]
    
    solutions = synthesizer.solve_complex_problem(complex_problem, relevant_domains)
    for i, solution in enumerate(solutions, 1):
        print(f"   Solution {i}: {solution['approach']}")
        print(f"      Quality: {solution['quality']:.2f}, Innovation: {solution['innovation_score']:.2f}")
    
    print(f"\n🎉 KNOWLEDGE SYNTHESIZER ACTIVE!")
    print(f"   Total breakthroughs: {len(synthesizer.breakthroughs)}")
    print(f"   Knowledge domains: {list(synthesizer.knowledge_base.keys())}")