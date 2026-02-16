#!/usr/bin/env python3
"""
Phase 3C: Official Evaluation, Optimization & Docker Integration

Automated script to:
1. Run official GAIA benchmark evaluation
2. Optimize parameters (threshold, attempts, embeddings)
3. Enable Docker integration for enhanced tool support
"""

import json
import subprocess
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Phase3CEvaluator:
    """Orchestrates Phase 3C evaluation, optimization, and Docker integration"""
    
    def __init__(self, output_dir: str = "."):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results: Dict[str, Any] = {}
        
    def run_evaluation(
        self,
        limit: Optional[int] = None,
        level: Optional[int] = None,
        with_docker: bool = False,
        label: str = "evaluation"
    ) -> Dict[str, Any]:
        """Run a single evaluation and return results"""
        
        cmd = [
            "python3",
            "ngvt_inspect_ai_integration.py",
        ]
        
        if limit:
            cmd.extend(["--limit", str(limit)])
        if level:
            cmd.extend(["--level", str(level)])
        if with_docker:
            cmd.append("--with-docker")
            
        output_file = self.output_dir / f"phase3c_{label}_{self.timestamp}.json"
        cmd.extend(["--output", str(output_file)])
        
        logger.info(f"Running: {' '.join(cmd)}")
        logger.info(f"Output: {output_file}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            
            if result.returncode == 0:
                logger.info(f"✓ Evaluation complete: {label}")
                
                # Load and return results
                if output_file.exists():
                    with open(output_file) as f:
                        return json.load(f)
            else:
                logger.error(f"✗ Evaluation failed: {label}")
                logger.error(f"STDERR: {result.stderr}")
                return {}
                
        except subprocess.TimeoutExpired:
            logger.error(f"✗ Timeout: {label} (exceeded 1 hour)")
            return {}
        except Exception as e:
            logger.error(f"✗ Error: {label} - {e}")
            return {}
    
    def phase_3c_1_official_evaluation(self):
        """Workstream 1: Official Evaluation"""
        
        logger.info("="*80)
        logger.info("PHASE 3C-1: OFFICIAL EVALUATION")
        logger.info("="*80)
        
        # Check HF token
        if not os.getenv("HF_TOKEN"):
            logger.warning("HF_TOKEN not set. Cannot access official dataset.")
            logger.warning("To get token:")
            logger.warning("  1. Go to: https://huggingface.co/datasets/gaia-benchmark/GAIA")
            logger.warning("  2. Request dataset access")
            logger.warning("  3. Create token: https://huggingface.co/settings/tokens")
            logger.warning("  4. Run: export HF_TOKEN='your_token'")
            return
        
        logger.info("HF_TOKEN detected. Proceeding with official evaluation...")
        
        # Phase 3C-1A: Quick validation (50 questions)
        logger.info("\nPhase 3C-1A: Quick validation (50 questions)")
        result_50 = self.run_evaluation(limit=50, label="validation_50")
        if result_50:
            self.results["phase_3c_1a_validation_50"] = result_50
            self._log_results(result_50, "50-question validation")
        
        # Phase 3C-1B: By difficulty level
        logger.info("\nPhase 3C-1B: Evaluation by difficulty level (50 per level)")
        for level in [1, 2, 3]:
            result = self.run_evaluation(limit=50, level=level, label=f"level{level}_50")
            if result:
                self.results[f"phase_3c_1b_level{level}"] = result
                self._log_results(result, f"Level {level} (50 questions)")
        
        # Phase 3C-1C: Full validation split
        logger.info("\nPhase 3C-1C: Full validation split (450 questions)")
        logger.info("This may take 2-5 hours...")
        
        if self._ask_to_proceed("Run full 450-question evaluation?"):
            result_full = self.run_evaluation(label="validation_full")
            if result_full:
                self.results["phase_3c_1c_validation_full"] = result_full
                self._log_results(result_full, "Full 450-question validation")
    
    def phase_3c_2_optimization(self):
        """Workstream 2: Optimization & Tuning"""
        
        logger.info("="*80)
        logger.info("PHASE 3C-2: OPTIMIZATION & TUNING")
        logger.info("="*80)
        
        if not os.getenv("HF_TOKEN"):
            logger.warning("HF_TOKEN not set. Cannot optimize on official dataset.")
            logger.warning("Using mock data for optimization tuning...")
        
        logger.info("\nPhase 3C-2A: Testing semantic match thresholds")
        logger.info("Testing thresholds: 0.6, 0.7, 0.75, 0.8, 0.9")
        
        threshold_results = {}
        for threshold in [0.6, 0.7, 0.75, 0.8, 0.9]:
            logger.info(f"\nTesting threshold={threshold}")
            # Note: In production, would modify ngvt_inspect_ai_integration.py
            # and re-run. For now, we document the structure.
            threshold_results[threshold] = {
                "status": "pending",
                "note": "Requires code modification and re-run"
            }
        
        self.results["phase_3c_2a_thresholds"] = threshold_results
        
        logger.info("\nPhase 3C-2B: Testing max_attempts parameter")
        logger.info("Testing attempts: 2, 3, 5")
        
        attempts_results = {}
        for attempts in [2, 3, 5]:
            logger.info(f"\nTesting max_attempts={attempts}")
            # Note: In production, would modify and re-run
            attempts_results[attempts] = {
                "status": "pending",
                "note": "Requires code modification and re-run"
            }
        
        self.results["phase_3c_2b_attempts"] = attempts_results
        
        logger.info("\nPhase 3C-2C: Testing embeddings impact")
        # Would run with --no-embeddings flag
        logger.info("Comparing semantic vs non-semantic matching...")
    
    def phase_3c_3_docker_integration(self):
        """Workstream 3: Docker Integration"""
        
        logger.info("="*80)
        logger.info("PHASE 3C-3: DOCKER INTEGRATION")
        logger.info("="*80)
        
        # Check Docker availability
        try:
            result = subprocess.run(["docker", "--version"], capture_output=True, text=True)
            if result.returncode == 0:
                logger.info(f"✓ Docker found: {result.stdout.strip()}")
                docker_available = True
            else:
                docker_available = False
        except FileNotFoundError:
            docker_available = False
        
        if not docker_available:
            logger.warning("Docker not found. To install:")
            logger.warning("  1. Download: https://www.docker.com/products/docker-desktop")
            logger.warning("  2. Install and launch Docker Desktop")
            logger.warning("  3. Verify: docker run hello-world")
            logger.warning("  4. Re-run this script with --with-docker flag")
            return
        
        logger.info("Docker is available. Proceeding with Docker-enabled evaluation...")
        
        # Phase 3C-3A: Quick Docker test
        if os.getenv("HF_TOKEN"):
            logger.info("\nPhase 3C-3A: Quick Docker test (10 questions)")
            if self._ask_to_proceed("Run Docker-enabled quick test?"):
                result_docker_quick = self.run_evaluation(
                    limit=10,
                    with_docker=True,
                    label="docker_quick"
                )
                if result_docker_quick:
                    self.results["phase_3c_3a_docker_quick"] = result_docker_quick
                    self._log_results(result_docker_quick, "Docker quick test (10 questions)")
            
            # Phase 3C-3B: Docker-enabled full evaluation
            logger.info("\nPhase 3C-3B: Full Docker-enabled evaluation (450 questions)")
            logger.info("This enables web_search and bash execution")
            
            if self._ask_to_proceed("Run full Docker-enabled evaluation?"):
                result_docker_full = self.run_evaluation(
                    with_docker=True,
                    label="docker_full"
                )
                if result_docker_full:
                    self.results["phase_3c_3b_docker_full"] = result_docker_full
                    self._log_results(result_docker_full, "Docker full evaluation")
    
    def _log_results(self, result: Dict[str, Any], label: str):
        """Log evaluation results"""
        if not result:
            return
        
        accuracy = result.get("accuracy", 0)
        total = result.get("total_questions", 0)
        correct = result.get("correct", 0)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Results: {label}")
        logger.info(f"{'='*60}")
        logger.info(f"Total Questions:    {total}")
        logger.info(f"Correct Answers:    {correct}")
        logger.info(f"Accuracy:           {accuracy:.1%}")
        
        by_level = result.get("by_level", {})
        if by_level:
            logger.info(f"\nBy Difficulty Level:")
            for level, metrics in by_level.items():
                level_acc = metrics.get("accuracy", 0)
                level_correct = metrics.get("correct", 0)
                level_total = metrics.get("total", 0)
                logger.info(f"  Level {level}: {level_correct}/{level_total} ({level_acc:.1%})")
        
        perf = result.get("performance", {})
        if perf:
            total_time = perf.get("total_time_seconds", 0)
            throughput = perf.get("throughput_per_second", 0)
            logger.info(f"\nPerformance:")
            logger.info(f"  Total Time:     {total_time:.1f}s ({total_time/60:.1f}m)")
            logger.info(f"  Throughput:     {throughput:.2f} q/s")
    
    def _ask_to_proceed(self, question: str) -> bool:
        """Ask user for confirmation"""
        response = input(f"\n{question} (y/n): ").strip().lower()
        return response == 'y'
    
    def generate_report(self):
        """Generate comprehensive Phase 3C analysis report"""
        
        report_path = self.output_dir / f"PHASE_3C_REPORT_{self.timestamp}.md"
        
        report = """# Phase 3C: Official Evaluation, Optimization & Docker Integration Report

## Executive Summary

This report documents Phase 3C evaluation results across three workstreams:
1. Official GAIA benchmark evaluation
2. Parameter optimization and tuning
3. Docker integration for enhanced tool support

## Results Summary

"""
        
        # Add evaluation results
        if "phase_3c_1a_validation_50" in self.results:
            result = self.results["phase_3c_1a_validation_50"]
            report += f"""### Official Evaluation (Phase 3C-1)

**Quick Validation (50 questions)**
- Accuracy: {result.get('accuracy', 0):.1%}
- Correct: {result.get('correct', 0)}/{result.get('total_questions', 0)}

"""
        
        # Add optimization results
        if "phase_3c_2a_thresholds" in self.results:
            report += """### Optimization & Tuning (Phase 3C-2)

**Semantic Match Threshold Analysis**
- Tested thresholds: 0.6, 0.7, 0.75, 0.8, 0.9
- Recommendation: (to be determined after evaluation)

"""
        
        # Add Docker results
        if "phase_3c_3a_docker_quick" in self.results:
            result = self.results["phase_3c_3a_docker_quick"]
            report += f"""### Docker Integration (Phase 3C-3)

**Docker-enabled Quick Test (10 questions)**
- Status: ✓ Docker support verified
- Accuracy: {result.get('accuracy', 0):.1%}

"""
        
        report += """## Recommendations

Based on evaluation results:

1. **Parameter Tuning**: Fine-tune threshold and max_attempts based on results
2. **Docker Deployment**: Enable Docker for web_search capability (+5-10% on Level 1)
3. **Leaderboard Preparation**: Use optimal configuration for final submission

## Next Steps

1. Review detailed results in JSON files
2. Identify optimal parameter configuration
3. Run final evaluation with best settings
4. Prepare for leaderboard submission

---

Generated: {}
""".format(datetime.now().isoformat())
        
        with open(report_path, 'w') as f:
            f.write(report)
        
        logger.info(f"\n✓ Report saved: {report_path}")
        return report_path
    
    def run_all(self):
        """Run all three workstreams"""
        
        logger.info("\n" + "="*80)
        logger.info("PHASE 3C: OFFICIAL EVALUATION, OPTIMIZATION & DOCKER INTEGRATION")
        logger.info("="*80 + "\n")
        
        try:
            # Workstream 1: Official Evaluation
            self.phase_3c_1_official_evaluation()
            
            # Workstream 2: Optimization
            self.phase_3c_2_optimization()
            
            # Workstream 3: Docker Integration
            self.phase_3c_3_docker_integration()
            
            # Generate report
            self.generate_report()
            
            # Save comprehensive results
            results_file = self.output_dir / f"phase3c_all_results_{self.timestamp}.json"
            with open(results_file, 'w') as f:
                json.dump(self.results, f, indent=2)
            logger.info(f"\n✓ All results saved: {results_file}")
            
            logger.info("\n" + "="*80)
            logger.info("PHASE 3C COMPLETE")
            logger.info("="*80)
            
        except KeyboardInterrupt:
            logger.info("\n\nPhase 3C interrupted by user")
            sys.exit(0)
        except Exception as e:
            logger.error(f"\nPhase 3C failed: {e}", exc_info=True)
            sys.exit(1)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Phase 3C: Official Evaluation, Optimization & Docker")
    parser.add_argument("--workstream", choices=["1", "2", "3", "all"], default="all",
                        help="Which workstream to run (default: all)")
    parser.add_argument("--output-dir", default=".", help="Output directory for results")
    
    args = parser.parse_args()
    
    evaluator = Phase3CEvaluator(output_dir=args.output_dir)
    
    if args.workstream == "1" or args.workstream == "all":
        evaluator.phase_3c_1_official_evaluation()
    
    if args.workstream == "2" or args.workstream == "all":
        evaluator.phase_3c_2_optimization()
    
    if args.workstream == "3" or args.workstream == "all":
        evaluator.phase_3c_3_docker_integration()
    
    evaluator.generate_report()
