from knowledge_synthesizer import KnowledgeSynthesizer
import random

class CrossDomainBrain(KnowledgeSynthesizer):
    def __init__(self, name="PrometheusV2"):
        super().__init__(name)
        self.connection_strengths = {}  # Track which domain pairs work well together
        self.insight_training_data = self._create_insight_examples()
        
    def _create_insight_examples(self):
        """Provide examples of good cross-domain insights to learn from"""
        return [
            {"domains": ["physics", "biology"], "concepts": ["quantum tunneling", "enzyme catalysis"], "quality": 0.9},
            {"domains": ["computer_science", "biology"], "concepts": ["neural networks", "neural networks"], "quality": 0.8},
            {"domains": ["mathematics", "physics"], "concepts": ["complex systems", "quantum mechanics"], "quality": 0.85},
            {"domains": ["engineering", "biology"], "concepts": ["control theory", "homeostasis"], "quality": 0.7}
        ]
    
    def train_cross_domain_thinking(self):
        """Train the brain to recognize good cross-domain connections"""
        print("🧠 TRAINING CROSS-DOMAIN THINKING...")
        
        for example in self.insight_training_data:
            domain1, domain2 = example["domains"]
            concept1, concept2 = example["concepts"]
            target_quality = example["quality"]
            
            # Encode the connection
            connection_input = self._encode_cross_domain_connection(domain1, concept1, domain2, concept2)
            connection_output = self.think(connection_input)
            
            current_quality = sum(connection_output) / len(connection_output)
            
            # Evolve to improve connection recognition
            performance = 1.0 - abs(current_quality - target_quality)  # Closer to target = better
            self.evolve_architecture(performance)
            
            print(f"   Training: {concept1} + {concept2} -> Quality: {current_quality:.2f} (Target: {target_quality})")
    
    def _encode_cross_domain_connection(self, domain1, concept1, domain2, concept2):
        """Advanced encoding specifically for cross-domain connections"""
        input_vector = []
        
        # Domain compatibility encoding
        domain_compatibility = self._calculate_domain_compatibility(domain1, domain2)
        input_vector.append(domain_compatibility)
        
        # Concept similarity features
        similarity_features = self._calculate_concept_similarity(concept1, concept2)
        input_vector.extend(similarity_features)
        
        # Domain-specific encodings
        domain1_features = self._encode_domain_specific(domain1, concept1)
        domain2_features = self._encode_domain_specific(domain2, concept2)
        input_vector.extend(domain1_features)
        input_vector.extend(domain2_features)
        
        # Cross-domain interaction patterns
        interaction_patterns = self._generate_interaction_patterns(domain1, domain2)
        input_vector.extend(interaction_patterns)
        
        # Pad to standard size
        while len(input_vector) < 50:
            input_vector.append(random.uniform(0, 0.1))
            
        return input_vector
    
    def _calculate_domain_compatibility(self, domain1, domain2):
        """Calculate how compatible two domains are for cross-pollination"""
        compatibility_scores = {
            ("physics", "engineering"): 0.9,
            ("biology", "computer_science"): 0.8,
            ("mathematics", "physics"): 0.95,
            ("biology", "engineering"): 0.7,
            ("computer_science", "mathematics"): 0.9,
            ("physics", "biology"): 0.6
        }
        
        key = tuple(sorted([domain1, domain2]))
        return compatibility_scores.get(key, 0.5)
    
    def _calculate_concept_similarity(self, concept1, concept2):
        """Calculate various similarity measures between concepts"""
        features = []
        
        concept1_str = str(concept1).lower()
        concept2_str = str(concept2).lower()
        
        # Length similarity
        len_similarity = 1.0 - (abs(len(concept1_str) - len(concept2_str)) / max(len(concept1_str), len(concept2_str)))
        features.append(len_similarity)
        
        # Word overlap
        words1 = set(concept1_str.split())
        words2 = set(concept2_str.split())
        if words1 and words2:
            word_overlap = len(words1.intersection(words2)) / len(words1.union(words2))
        else:
            word_overlap = 0.0
        features.append(word_overlap)
        
        # Common scientific prefixes/suffixes
        science_terms = ["quantum", "neural", "system", "theory", "network", "optimization"]
        common_terms = sum(1 for term in science_terms if term in concept1_str and term in concept2_str)
        features.append(common_terms / len(science_terms))
        
        return features
    
    def _encode_domain_specific(self, domain, concept):
        """Domain-specific feature encoding"""
        features = []
        concept_str = str(concept).lower()
        
        # Domain characteristic features
        if domain == "physics":
            physics_terms = ["quantum", "relativity", "energy", "field", "particle"]
            physics_score = sum(1 for term in physics_terms if term in concept_str) / len(physics_terms)
            features.append(physics_score)
        elif domain == "biology":
            bio_terms = ["cell", "dna", "evolution", "enzyme", "organism"]
            bio_score = sum(1 for term in bio_terms if term in concept_str) / len(bio_terms)
            features.append(bio_score)
        elif domain == "computer_science":
            cs_terms = ["algorithm", "network", "data", "optimization", "machine"]
            cs_score = sum(1 for term in cs_terms if term in concept_str) / len(cs_terms)
            features.append(cs_score)
        else:
            features.append(0.5)  # Default
        
        # Abstractness measure
        abstract_terms = ["theory", "system", "principle", "concept", "model"]
        abstractness = sum(1 for term in abstract_terms if term in concept_str) / len(abstract_terms)
        features.append(abstractness)
        
        return features
    
    def _generate_interaction_patterns(self, domain1, domain2):
        """Generate patterns that encourage cross-domain thinking"""
        patterns = []
        
        # Complementary domain patterns
        domain_pairs = {
            ("physics", "engineering"): [0.9, 0.1, 0.8],  # Theory + Application
            ("biology", "computer_science"): [0.2, 0.9, 0.3],  # Natural + Artificial
            ("mathematics", "physics"): [0.8, 0.7, 0.9],  # Abstract + Physical
        }
        
        key = tuple(sorted([domain1, domain2]))
        if key in domain_pairs:
            patterns.extend(domain_pairs[key])
        else:
            patterns.extend([0.5, 0.5, 0.5])  # Neutral pattern
        
        # Innovation potential indicator
        innovation_potential = random.uniform(0.6, 0.9) if domain1 != domain2 else 0.3
        patterns.append(innovation_potential)
        
        return patterns
    
    def enhanced_generate_insights(self):
        """Generate insights using trained cross-domain thinking"""
        print("💡 ENHANCED CROSS-DOMAIN INSIGHT GENERATION...")
        
        self.train_cross_domain_thinking()
        
        insights = []
        domains_with_concepts = [domain for domain in self.knowledge_base 
                               if len(self.knowledge_base[domain]) >= 2]
        
        for attempt in range(10):  # More attempts
            domain1, domain2 = random.sample(domains_with_concepts, 2)
            concept1 = random.choice(self.knowledge_base[domain1])
            concept2 = random.choice(self.knowledge_base[domain2])
            
            # Use enhanced encoding
            connection_input = self._encode_cross_domain_connection(domain1, concept1, domain2, concept2)
            connection_output = self.think(connection_input)
            
            insight_quality = sum(connection_output) / len(connection_output)
            
            print(f"   {concept1} ({domain1}) + {concept2} ({domain2}) -> Quality: {insight_quality:.2f}")
            
            if insight_quality > 0.5:  # More reasonable threshold
                insight = self._formulate_enhanced_insight(domain1, concept1, domain2, concept2, insight_quality)
                insights.append(insight)
                
                if insight_quality > 0.7:
                    self.record_breakthrough(f"Cross-domain insight: {insight[:80]}...")
            
            if len(insights) >= 5:  # Get more insights
                break
        
        return insights
    
    def _formulate_enhanced_insight(self, domain1, concept1, domain2, concept2, quality):
        """Create more sophisticated cross-domain insights"""
        insight_templates = [
            f"Applying {concept1} from {domain1} could revolutionize approaches to {concept2} in {domain2}",
            f"The principles of {concept1} ({domain1}) might provide novel solutions to challenges in {domain2} involving {concept2}",
            f"Combining {concept1} with {concept2} could lead to breakthrough innovations at the intersection of {domain1} and {domain2}",
            f"{domain2} could be transformed by adapting the {concept1} framework from {domain1} to address {concept2}",
            f"What if we used {concept1} to solve the fundamental problems of {concept2} in {domain2}?",
            f"The {concept1} approach from {domain1} might unlock new understanding of {concept2} in {domain2}"
        ]
        
        template = random.choice(insight_templates)
        return f"{template} [Confidence: {quality:.2f}]"

# 🧪 TEST THE ENHANCED BRAIN
if __name__ == "__main__":
    enhanced_brain = CrossDomainBrain("PrometheusV2")
    
    print("🧠 ENHANCED CROSS-DOMAIN BRAIN TEST")
    print("=" * 60)
    
    # Ingest knowledge
    scientific_knowledge = {
        "physics": ["quantum entanglement", "relativity", "thermodynamics", "electromagnetism", "superconductivity"],
        "biology": ["evolution", "DNA", "cellular respiration", "neural networks", "enzyme catalysis"],
        "computer_science": ["algorithms", "machine learning", "neural networks", "optimization", "data structures"],
        "mathematics": ["calculus", "probability", "graph theory", "linear algebra", "complex systems"],
        "engineering": ["systems design", "materials science", "energy efficiency", "control theory"]
    }
    
    for domain, concepts in scientific_knowledge.items():
        enhanced_brain.ingest_knowledge(domain, concepts)
    
    # Generate enhanced insights
    insights = enhanced_brain.enhanced_generate_insights()
    
    print(f"\n🎉 ENHANCED INSIGHTS GENERATED:")
    for i, insight in enumerate(insights, 1):
        print(f"   {i}. {insight}")
    
    print(f"\n📊 FINAL BRAIN STATS:")
    stats = enhanced_brain.get_brain_stats()
    print(f"   Generation: {stats['generation']}")
    print(f"   Neurons: {stats['total_neurons']}")
    print(f"   Connections: {stats['total_connections']}")
    print(f"   Breakthroughs: {len(enhanced_brain.breakthroughs)}")