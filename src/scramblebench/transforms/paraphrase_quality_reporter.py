"""
Quality Reporting System for Paraphrase Pipeline.

This module provides comprehensive quality assessment and reporting capabilities
for the paraphrase control pipeline, ensuring academic-grade standards and
detailed analysis for contamination detection research.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)


class ParaphraseQualityReporter:
    """
    Comprehensive quality assessment and reporting for paraphrase pipeline.
    
    Generates detailed reports, visualizations, and recommendations for
    paraphrase quality, acceptance rates, and provider isolation compliance.
    """
    
    def __init__(self, output_dir: Path = None):
        """
        Initialize quality reporter.
        
        Args:
            output_dir: Directory for saving reports and visualizations
        """
        self.output_dir = output_dir or Path("results/paraphrase_reports")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure plotting style
        plt.style.use('default')
        sns.set_palette("husl")
    
    def generate_comprehensive_report(self, 
                                    generation_results: Dict[str, Any],
                                    save_artifacts: bool = True) -> Dict[str, Any]:
        """
        Generate comprehensive quality assessment report.
        
        Args:
            generation_results: Results from paraphrase generation pipeline
            save_artifacts: Whether to save report artifacts to disk
            
        Returns:
            Comprehensive report dictionary with quality metrics and recommendations
        """
        timestamp = datetime.now().isoformat()
        
        # Extract key statistics
        stats = generation_results.get("statistics", {})
        quality = generation_results.get("quality_assessment", {})
        rejection_analysis = generation_results.get("rejection_analysis", {})
        provider_isolation = generation_results.get("provider_isolation", {})
        
        # Calculate derived metrics
        total_items = stats.get("total_items", 0)
        acceptance_rate = stats.get("acceptance_rate", 0.0)
        cache_hit_rate = stats.get("cache_hit_rate", 0.0)
        
        # Assess quality dimensions
        quality_assessment = self._assess_quality_dimensions(
            acceptance_rate, rejection_analysis, provider_isolation, total_items
        )
        
        # Generate recommendations
        recommendations = self._generate_detailed_recommendations(
            quality_assessment, stats, rejection_analysis
        )
        
        # Create comprehensive report
        report = {
            "report_metadata": {
                "generated_at": timestamp,
                "report_version": "1.0",
                "pipeline_provider": generation_results.get("provider"),
                "total_items_processed": total_items
            },
            "executive_summary": {
                "overall_quality_score": quality_assessment["overall_score"],
                "acceptance_rate": acceptance_rate,
                "meets_academic_standards": quality_assessment["meets_standards"],
                "provider_isolation_status": "maintained" if provider_isolation.get("isolation_maintained") else "violated",
                "primary_concerns": quality_assessment["primary_concerns"]
            },
            "detailed_metrics": {
                "generation_statistics": stats,
                "quality_dimensions": quality_assessment,
                "rejection_breakdown": self._analyze_rejection_patterns(rejection_analysis),
                "provider_isolation_analysis": self._analyze_provider_isolation(provider_isolation)
            },
            "recommendations": {
                "immediate_actions": recommendations["immediate"],
                "process_improvements": recommendations["process"],
                "technical_optimizations": recommendations["technical"]
            },
            "visualizations_generated": []
        }
        
        # Generate visualizations
        if save_artifacts:
            viz_paths = self._generate_visualizations(generation_results, report)
            report["visualizations_generated"] = viz_paths
            
            # Save report
            report_path = self.output_dir / f"quality_report_{timestamp.replace(':', '-')}.json"
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            # Save human-readable summary
            summary_path = self.output_dir / f"quality_summary_{timestamp.replace(':', '-')}.md"
            self._save_markdown_summary(report, summary_path)
            
            logger.info(f"Quality report saved: {report_path}")
            logger.info(f"Summary saved: {summary_path}")
        
        return report
    
    def _assess_quality_dimensions(self, 
                                 acceptance_rate: float,
                                 rejection_analysis: Dict[str, int],
                                 provider_isolation: Dict[str, Any],
                                 total_items: int) -> Dict[str, Any]:
        """Assess multiple quality dimensions."""
        
        # Acceptance rate quality (target: ≥95%)
        acceptance_score = min(1.0, acceptance_rate / 0.95) if acceptance_rate > 0 else 0.0
        
        # Provider isolation quality (binary: maintained or violated)
        isolation_score = 1.0 if provider_isolation.get("isolation_maintained", False) else 0.0
        
        # Rejection pattern quality (prefer semantic fails over surface fails)
        rejection_score = self._score_rejection_patterns(rejection_analysis, total_items)
        
        # Sample size adequacy
        sample_size_score = min(1.0, total_items / 100) if total_items > 0 else 0.0  # Target: ≥100 items
        
        # Overall weighted score
        weights = {
            "acceptance": 0.4,    # Most important
            "isolation": 0.3,     # Critical for academic integrity
            "rejection_patterns": 0.2,  # Important for understanding
            "sample_size": 0.1    # Least important but still relevant
        }
        
        overall_score = (
            acceptance_score * weights["acceptance"] +
            isolation_score * weights["isolation"] +
            rejection_score * weights["rejection_patterns"] +
            sample_size_score * weights["sample_size"]
        )
        
        # Determine if meets academic standards
        meets_standards = (
            acceptance_score >= 0.95 and  # ≥95% acceptance rate
            isolation_score == 1.0 and    # Perfect isolation
            total_items >= 50             # Adequate sample size
        )
        
        # Identify primary concerns
        concerns = []
        if acceptance_score < 0.95:
            concerns.append(f"Acceptance rate ({acceptance_rate:.1%}) below target (95%)")
        if isolation_score < 1.0:
            concerns.append("Provider isolation violated - contamination risk")
        if total_items < 50:
            concerns.append(f"Sample size ({total_items}) too small for reliable results")
        if rejection_score < 0.7:
            concerns.append("Concerning rejection patterns detected")
        
        return {
            "overall_score": overall_score,
            "meets_standards": meets_standards,
            "scores": {
                "acceptance_rate": acceptance_score,
                "provider_isolation": isolation_score,
                "rejection_patterns": rejection_score,
                "sample_size": sample_size_score
            },
            "primary_concerns": concerns
        }
    
    def _score_rejection_patterns(self, rejection_analysis: Dict[str, int], total_items: int) -> float:
        """Score rejection patterns (prefer semantic over surface failures)."""
        if total_items == 0:
            return 1.0
        
        semantic_fails = rejection_analysis.get("semantic_failures", 0)
        surface_fails = rejection_analysis.get("surface_failures", 0)
        both_fails = rejection_analysis.get("both_failures", 0)
        generation_fails = rejection_analysis.get("generation_failures", 0)
        
        total_rejects = semantic_fails + surface_fails + both_fails + generation_fails
        
        if total_rejects == 0:
            return 1.0  # Perfect - no rejections
        
        # Scoring logic:
        # - Generation failures are worst (technical issues)
        # - Both failures are bad (fundamental quality issues)
        # - Surface failures are concerning (model not diverse enough)
        # - Semantic failures are most acceptable (model being careful)
        
        weights = {
            "semantic": 0.8,      # Most acceptable
            "surface": 0.4,       # Concerning
            "both": 0.1,          # Bad
            "generation": 0.0     # Worst
        }
        
        weighted_score = (
            semantic_fails * weights["semantic"] +
            surface_fails * weights["surface"] +
            both_fails * weights["both"] +
            generation_fails * weights["generation"]
        ) / total_rejects
        
        return weighted_score
    
    def _analyze_rejection_patterns(self, rejection_analysis: Dict[str, int]) -> Dict[str, Any]:
        """Analyze rejection patterns for insights."""
        total_rejects = sum(rejection_analysis.values())
        
        if total_rejects == 0:
            return {
                "total_rejections": 0,
                "patterns": "No rejections - excellent quality",
                "primary_failure_mode": "none"
            }
        
        # Find primary failure mode
        primary_mode = max(rejection_analysis.items(), key=lambda x: x[1])
        
        # Calculate percentages
        percentages = {
            key: (count / total_rejects * 100) if total_rejects > 0 else 0
            for key, count in rejection_analysis.items()
        }
        
        # Generate insights
        insights = []
        if percentages.get("generation_failures", 0) > 20:
            insights.append("High generation failure rate indicates technical/connectivity issues")
        if percentages.get("both_failures", 0) > 15:
            insights.append("High both-criteria failure rate suggests fundamental quality issues")
        if percentages.get("semantic_failures", 0) > percentages.get("surface_failures", 0):
            insights.append("More semantic failures than surface - model may be too conservative")
        if percentages.get("surface_failures", 0) > 30:
            insights.append("High surface failure rate - model not generating sufficiently diverse paraphrases")
        
        return {
            "total_rejections": total_rejects,
            "percentages": percentages,
            "primary_failure_mode": primary_mode[0],
            "insights": insights
        }
    
    def _analyze_provider_isolation(self, provider_isolation: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze provider isolation compliance."""
        isolation_maintained = provider_isolation.get("isolation_maintained", False)
        violations = provider_isolation.get("violations", [])
        
        analysis = {
            "status": "compliant" if isolation_maintained else "violated",
            "risk_level": "none" if isolation_maintained else "critical",
            "violation_count": len(violations),
            "academic_integrity": "maintained" if isolation_maintained else "compromised"
        }
        
        if not isolation_maintained:
            analysis["violations_summary"] = violations
            analysis["remediation_required"] = True
        
        return analysis
    
    def _generate_detailed_recommendations(self,
                                         quality_assessment: Dict[str, Any],
                                         stats: Dict[str, Any],
                                         rejection_analysis: Dict[str, int]) -> Dict[str, List[str]]:
        """Generate detailed recommendations based on quality assessment."""
        
        immediate = []
        process = []
        technical = []
        
        acceptance_rate = stats.get("acceptance_rate", 0.0)
        overall_score = quality_assessment["overall_score"]
        
        # Immediate actions
        if acceptance_rate < 0.95:
            immediate.append(
                f"URGENT: Acceptance rate ({acceptance_rate:.1%}) below academic standard (95%). "
                f"Review and adjust quality thresholds or improve paraphrase generation."
            )
        
        if not quality_assessment["scores"]["provider_isolation"]:
            immediate.append(
                "CRITICAL: Provider isolation violated. Immediately separate paraphrase and evaluation providers."
            )
        
        if stats.get("total_items", 0) < 50:
            immediate.append(
                "Increase sample size to at least 100 items for statistically significant results."
            )
        
        # Process improvements
        if overall_score < 0.8:
            process.append("Consider comprehensive review of paraphrase generation process and quality standards.")
        
        if rejection_analysis.get("generation_failures", 0) > stats.get("total_items", 0) * 0.1:
            process.append("High generation failure rate suggests need for more robust error handling and retry logic.")
        
        semantic_fails = rejection_analysis.get("semantic_failures", 0)
        surface_fails = rejection_analysis.get("surface_failures", 0)
        
        if surface_fails > semantic_fails:
            process.append(
                "Surface divergence failures exceed semantic failures. "
                "Consider prompt engineering to encourage more diverse paraphrasing."
            )
        
        # Technical optimizations
        cache_hit_rate = stats.get("cache_hit_rate", 0.0)
        if cache_hit_rate < 0.5:
            technical.append(
                f"Low cache hit rate ({cache_hit_rate:.1%}) suggests inefficient caching. "
                f"Review cache key generation and storage strategy."
            )
        
        if stats.get("generated_count", 0) < stats.get("total_items", 0) * 0.5:
            technical.append(
                "Low generation success rate. Consider improving model connectivity, "
                "timeout settings, or fallback mechanisms."
            )
        
        # Add positive reinforcement
        if overall_score >= 0.9:
            process.append("Excellent quality metrics achieved. Consider this as a template for future runs.")
        
        return {
            "immediate": immediate,
            "process": process,
            "technical": technical
        }
    
    def _generate_visualizations(self, 
                               generation_results: Dict[str, Any],
                               report: Dict[str, Any]) -> List[str]:
        """Generate quality assessment visualizations."""
        viz_paths = []
        
        try:
            # 1. Acceptance rate visualization
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            stats = generation_results.get("statistics", {})
            acceptance_rate = stats.get("acceptance_rate", 0.0)
            
            # Acceptance rate gauge
            categories = ['Accepted', 'Rejected']
            values = [stats.get("accepted_count", 0), stats.get("rejected_count", 0)]
            colors = ['#2ecc71', '#e74c3c']
            
            ax1.pie(values, labels=categories, colors=colors, autopct='%1.1f%%', startangle=90)
            ax1.set_title(f'Acceptance Rate: {acceptance_rate:.1%}')
            
            # Target comparison
            target_rate = 0.95
            rates = ['Actual', 'Target']
            values = [acceptance_rate, target_rate]
            colors = ['#3498db', '#95a5a6']
            
            bars = ax2.bar(rates, values, color=colors)
            ax2.set_ylabel('Acceptance Rate')
            ax2.set_title('Acceptance Rate vs Target')
            ax2.set_ylim(0, 1.0)
            
            # Add target line
            ax2.axhline(y=target_rate, color='red', linestyle='--', alpha=0.7, label='Target (95%)')
            ax2.legend()
            
            plt.tight_layout()
            
            viz_path = self.output_dir / "acceptance_rate_analysis.png"
            plt.savefig(viz_path, dpi=300, bbox_inches='tight')
            plt.close()
            viz_paths.append(str(viz_path))
            
            # 2. Rejection analysis visualization
            rejection_analysis = generation_results.get("rejection_analysis", {})
            if any(rejection_analysis.values()):
                fig, ax = plt.subplots(figsize=(10, 6))
                
                rejection_types = list(rejection_analysis.keys())
                rejection_counts = list(rejection_analysis.values())
                
                bars = ax.bar(rejection_types, rejection_counts, 
                             color=['#e74c3c', '#f39c12', '#9b59b6', '#34495e'])
                ax.set_title('Rejection Analysis by Failure Type')
                ax.set_ylabel('Number of Rejections')
                ax.tick_params(axis='x', rotation=45)
                
                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    if height > 0:
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{int(height)}', ha='center', va='bottom')
                
                plt.tight_layout()
                
                viz_path = self.output_dir / "rejection_analysis.png"
                plt.savefig(viz_path, dpi=300, bbox_inches='tight')
                plt.close()
                viz_paths.append(str(viz_path))
            
            # 3. Quality score radar chart
            quality_dims = report["detailed_metrics"]["quality_dimensions"]["scores"]
            
            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
            
            categories = list(quality_dims.keys())
            scores = list(quality_dims.values())
            
            # Add first point to close the circle
            categories += [categories[0]]
            scores += [scores[0]]
            
            ax.plot(np.linspace(0, 2*np.pi, len(categories)), scores, 'o-', linewidth=2)
            ax.fill(np.linspace(0, 2*np.pi, len(categories)), scores, alpha=0.25)
            ax.set_xticks(np.linspace(0, 2*np.pi, len(categories)))
            ax.set_xticklabels([cat.replace('_', ' ').title() for cat in categories[:-1]])
            ax.set_ylim(0, 1)
            ax.set_title('Quality Dimensions Radar Chart', pad=20)
            
            viz_path = self.output_dir / "quality_radar.png"
            plt.savefig(viz_path, dpi=300, bbox_inches='tight')
            plt.close()
            viz_paths.append(str(viz_path))
            
        except Exception as e:
            logger.warning(f"Failed to generate some visualizations: {e}")
        
        return viz_paths
    
    def _save_markdown_summary(self, report: Dict[str, Any], output_path: Path):
        """Save human-readable markdown summary."""
        
        summary_md = f"""# Paraphrase Quality Assessment Report

## Executive Summary

- **Overall Quality Score**: {report['executive_summary']['overall_quality_score']:.2f}/1.00
- **Acceptance Rate**: {report['executive_summary']['acceptance_rate']:.1%}
- **Meets Academic Standards**: {report['executive_summary']['meets_academic_standards']}
- **Provider Isolation**: {report['executive_summary']['provider_isolation_status']}

## Key Findings

### Primary Concerns
"""
        
        concerns = report['executive_summary']['primary_concerns']
        if concerns:
            for concern in concerns:
                summary_md += f"- ⚠️ {concern}\\n"
        else:
            summary_md += "- ✅ No major concerns identified\\n"
        
        summary_md += """

## Detailed Metrics

### Generation Statistics
"""
        
        stats = report['detailed_metrics']['generation_statistics']
        for key, value in stats.items():
            if isinstance(value, float):
                value_str = f"{value:.2%}" if key.endswith('_rate') else f"{value:.2f}"
            else:
                value_str = str(value)
            summary_md += f"- **{key.replace('_', ' ').title()}**: {value_str}\\n"
        
        summary_md += """

## Recommendations

### Immediate Actions
"""
        
        immediate = report['recommendations']['immediate_actions']
        if immediate:
            for rec in immediate:
                summary_md += f"- 🚨 {rec}\\n"
        else:
            summary_md += "- ✅ No immediate actions required\\n"
        
        summary_md += """

### Process Improvements
"""
        
        process = report['recommendations']['process_improvements']
        if process:
            for rec in process:
                summary_md += f"- 🔄 {rec}\\n"
        else:
            summary_md += "- ✅ No process improvements needed\\n"
        
        summary_md += f"""

---
*Report generated at: {report['report_metadata']['generated_at']}*
"""
        
        with open(output_path, 'w') as f:
            f.write(summary_md)


def generate_quality_report(generation_results: Dict[str, Any], 
                          output_dir: Path = None) -> Dict[str, Any]:
    """
    Convenience function to generate a comprehensive quality report.
    
    Args:
        generation_results: Results from paraphrase generation pipeline
        output_dir: Optional output directory for artifacts
        
    Returns:
        Comprehensive quality report
    """
    reporter = ParaphraseQualityReporter(output_dir)
    return reporter.generate_comprehensive_report(generation_results, save_artifacts=True)