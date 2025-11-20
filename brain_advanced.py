from brain_core import EvolvingNeuralNetwork
import math
import random

class AdvancedBrain(EvolvingNeuralNetwork):
    def __init__(self, name="AdvancedAthena"):
        super().__init__(name)
        self.problem_types = {
            "math": self._solve_math,
            "pattern": self._solve_pattern,
            "logic": self._solve_logic,
            "creative": self._solve_creative
        }
        self.skills_developed = []
        
    def solve_problem(self, problem_type, difficulty=1.0):
        """Solve different types of problems"""
        if problem_type in self.problem_types:
            problem = self._generate_problem(problem_type, difficulty)
            solution = self.problem_types[problem_type](problem)
            
            # Evaluate performance
            performance = self._evaluate_solution(problem_type, problem, solution)
            
            # Evolve based on performance
            self.evolve_architecture(performance)
            
            # Record skill development
            if performance > 0.8 and problem_type not in self.skills_developed:
                self.skills_developed.append(problem_type)
                self.record_breakthrough(f"Mastered {problem_type} problems!")
            
            return solution, performance
        return None, 0.0
    
    def _generate_problem(self, problem_type, difficulty):
        """Generate problems of increasing difficulty"""
        if problem_type == "math":
            # Math problems: equations, sequences, calculations
            if difficulty < 0.3:
                return {"type": "addition", "a": random.randint(1, 10), "b": random.randint(1, 10)}
            elif difficulty < 0.6:
                return {"type": "sequence", "sequence": [2, 4, 6, 8], "next_count": 2}
            else:
                return {"type": "equation", "equation": "x² + 3x - 4 = 0", "solve_for": "x"}
        
        elif problem_type == "pattern":
            # Pattern recognition problems
            patterns = [
                [1, 2, 3, 4, 5],  # Linear
                [2, 4, 8, 16],    # Geometric
                [1, 1, 2, 3, 5],  # Fibonacci
                [1, 4, 9, 16]     # Squares
            ]
            return {"pattern": random.choice(patterns), "continue_to": 3}
        
        elif problem_type == "logic":
            # Logic puzzles
            return {"puzzle": "If A then B. Not B. Therefore?"}
        
        elif problem_type == "creative":
            # Creative problem solving
            return {"scenario": "Design a system to sort 1000 different objects by weight using only a balance scale"}
    
    def _solve_math(self, problem):
        """Math problem solving"""
        input_data = self._problem_to_input(problem)
        output = self.think(input_data)
        return self._interpret_math_output(output, problem)
    
    def _solve_pattern(self, problem):
        """Pattern recognition"""
        pattern = problem["pattern"]
        input_data = [x/10 for x in pattern] + [0] * (10 - len(pattern))
        output = self.think(input_data)
        return self._interpret_pattern_output(output, problem)
    
    def _solve_logic(self, problem):
        """Logic reasoning"""
        # Convert logic problem to neural input
        input_data = [random.random() for _ in range(50)]  # Placeholder
        output = self.think(input_data)
        return self._interpret_logic_output(output)
    
    def _solve_creative(self, problem):
        """Creative problem solving"""
        input_data = [random.random() for _ in range(50)]  # Placeholder
        output = self.think(input_data)
        return self._interpret_creative_output(output)
    
    def _interpret_math_output(self, output, problem):
        """Interpret neural output for math problems"""
        # For now, return a simple interpretation
        # In advanced version, this would decode the neural output into actual solutions
        avg_output = sum(output) / len(output)
        
        if problem["type"] == "addition":
            # Simulate addition solution
            result = problem["a"] + problem["b"]
            confidence = max(0.0, min(1.0, 1.0 - abs(avg_output - 0.5)))
            return {"answer": result, "confidence": confidence, "method": "neural_calculation"}
        
        elif problem["type"] == "sequence":
            # Simulate sequence continuation
            sequence = problem["sequence"]
            next_value = sequence[-1] + (sequence[-1] - sequence[-2])  # Linear continuation
            confidence = max(0.0, min(1.0, avg_output))
            return {"next_values": [next_value, next_value + 2], "confidence": confidence}
        
        else:  # equation
            # Simulate equation solving
            confidence = max(0.0, min(1.0, avg_output * 0.8))
            return {"solutions": [1, -4], "confidence": confidence, "method": "neural_analysis"}
    
    def _interpret_pattern_output(self, output, problem):
        """Interpret neural output for pattern recognition"""
        pattern = problem["pattern"]
        avg_output = sum(output) / len(output)
        
        # Simple pattern continuation logic
        if len(pattern) >= 2:
            diff = pattern[-1] - pattern[-2]
            next_value = pattern[-1] + diff
            confidence = max(0.0, min(1.0, avg_output))
            
            next_values = []
            for i in range(problem.get("continue_to", 2)):
                next_values.append(next_value + i * diff)
            
            return {"continued_pattern": next_values, "confidence": confidence}
        
        return {"continued_pattern": [], "confidence": 0.0}
    
    def _interpret_logic_output(self, output):
        """Interpret neural output for logic problems"""
        avg_output = sum(output) / len(output)
        
        # Simple logic interpretation
        if avg_output > 0.7:
            conclusion = "Therefore, not A"
            confidence = avg_output
        elif avg_output < 0.3:
            conclusion = "Insufficient information"
            confidence = 1.0 - avg_output
        else:
            conclusion = "The logic is inconsistent"
            confidence = 0.5
        
        return {"conclusion": conclusion, "confidence": confidence}
    
    def _interpret_creative_output(self, output):
        """Interpret neural output for creative problems"""
        avg_output = sum(output) / len(output)
        
        # Creative solutions based on neural activation patterns
        creativity_score = sum(abs(o - 0.5) for o in output) / len(output)  # Diversity of thought
        
        if creativity_score > 0.3:
            solution = "Use a binary search approach with the balance scale to efficiently sort objects by weight"
            innovation_level = "high"
        else:
            solution = "Weigh objects pairwise and sort them"
            innovation_level = "standard"
        
        return {
            "solution": solution,
            "creativity_score": creativity_score,
            "innovation_level": innovation_level,
            "confidence": avg_output
        }
    
    def _problem_to_input(self, problem):
        """Convert problems to neural network input - IMPROVED VERSION"""
        input_vector = []
        
        if problem["type"] == "addition":
            # Encode addition problem: [a/100, b/100, operation_code, ...]
            input_vector.extend([problem["a"] / 100, problem["b"] / 100, 0.1])
        
        elif problem["type"] == "sequence":
            # Encode sequence: normalize numbers and pad
            sequence = problem["sequence"]
            normalized_seq = [x / max(sequence) for x in sequence]
            input_vector.extend(normalized_seq)
        
        elif problem["type"] == "equation":
            # Encode equation type
            input_vector.extend([0.5, 0.3, 0.8])  # Math operation codes
        
        # Pad to 50 elements with problem-type indicators
        while len(input_vector) < 50:
            input_vector.append(random.uniform(0, 0.2))  # Background noise
        
        return input_vector
    
    def _problem_to_input(self, problem):
        """Convert problems to neural network input"""
        # This is where we'd encode problems numerically
        # For now, using random inputs to test evolution
        return [random.random() for _ in range(50)]
    
    def _evaluate_solution(self, problem_type, problem, solution):
        if problem_type == "math":
            if problem["type"] == "addition":
                correct_answer = problem["a"] + problem["b"]
                if solution.get("answer") == correct_answer:
                    return solution.get("confidence", 0.5)
                else:
                    return 0.1  # Wrong answer
                    
            elif problem["type"] == "sequence":
                # Check if pattern continuation makes sense
                actual_sequence = problem["sequence"]
                if len(actual_sequence) >= 2:
                    expected_diff = actual_sequence[-1] - actual_sequence[-2]
                    proposed_values = solution.get("next_values", [])
                    if len(proposed_values) >= 1:
                        expected_next = actual_sequence[-1] + expected_diff
                        if proposed_values[0] == expected_next:
                            return solution.get("confidence", 0.7)
                return 0.2
        
        # For other problem types, use confidence as performance metric
        return solution.get("confidence", 0.3)
    
    def _evaluate_solution(self, problem_type, problem, solution):
        """Evaluate how good the solution was"""
        # Simulated evaluation - in real implementation, this would check actual solutions
        base_performance = random.uniform(0.3, 0.9)
        
        # Bonus for developed skills
        if problem_type in self.skills_developed:
            base_performance += 0.2
        
        return min(1.0, base_performance)

# 🧪 TEST THE ADVANCED BRAIN
if __name__ == "__main__":
    advanced_brain = AdvancedBrain("AthenaV2")
    
    print("🧠 ADVANCED BRAIN TESTING - PROBLEM SOLVING EVOLUTION")
    print("=" * 60)
    
    problem_types = ["math", "pattern", "logic", "creative"]
    
    for cycle in range(20):
        problem_type = random.choice(problem_types)
        difficulty = min(1.0, cycle * 0.05)  # Increasing difficulty
        
        solution, performance = advanced_brain.solve_problem(problem_type, difficulty)
        
        if cycle % 5 == 0:
            stats = advanced_brain.get_brain_stats()
            print(f"\n📈 CYCLE {cycle} - Advanced Evolution:")
            print(f"   Generation: {stats['generation']}")
            print(f"   Neurons: {stats['total_neurons']}")
            print(f"   Connections: {stats['total_connections']}")
            print(f"   Skills: {advanced_brain.skills_developed}")
            print(f"   Last Performance: {performance:.2f}")
    
    print(f"\n🎉 FINAL BRAIN STATS:")
    final_stats = advanced_brain.get_brain_stats()
    for key, value in final_stats.items():
        if key not in ["architecture", "breakthroughs"]:
            print(f"   {key}: {value}")
    print(f"   Skills Developed: {advanced_brain.skills_developed}")
    print(f"   Total Breakthroughs: {len(advanced_brain.breakthroughs)}")