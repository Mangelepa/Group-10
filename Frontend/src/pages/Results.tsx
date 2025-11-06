import { useLocation, useNavigate } from "react-router-dom";
import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import HealthScoreMeter from "@/components/results/HealthScoreMeter";
import MetricCard from "@/components/results/MetricCard";
import RecommendationCard from "@/components/results/RecommendationCard";
import { FinancialData, FinancialAnalysis } from "@/types/financial";
import { calculateFinancialHealth } from "@/lib/calculations";
import { Download, ArrowLeft, TrendingUp, AlertTriangle, CheckCircle } from "lucide-react";

const Results = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const [analysis, setAnalysis] = useState<FinancialAnalysis | null>(null);

  useEffect(() => {
    const data = location.state?.financialData as FinancialData;
    if (!data) {
      navigate("/");
      return;
    }

    const result = calculateFinancialHealth(data);
    setAnalysis(result);
  }, [location, navigate]);

  if (!analysis) {
    return null;
  }

  const getRiskColor = (risk: string) => {
    switch (risk) {
      case "Healthy": return "text-success";
      case "Watch": return "text-warning";
      case "At Risk": return "text-destructive";
      default: return "text-foreground";
    }
  };

  return (
    <div className="min-h-screen bg-gradient-hero">
      <div className="container mx-auto px-4 py-8 max-w-6xl">
        {/* Header */}
        <div className="flex justify-between items-center mb-8">
          <Button 
            variant="outline" 
            onClick={() => navigate("/assessment")}
            className="flex items-center gap-2"
          >
            <ArrowLeft className="w-4 h-4" />
            Back to Assessment
          </Button>
          <Button className="flex items-center gap-2 bg-gradient-primary">
            <Download className="w-4 h-4" />
            Export Report
          </Button>
        </div>

        {/* Executive Summary */}
        <Card className="p-8 mb-8 shadow-strong">
          <div className="text-center mb-6">
            <h1 className="text-3xl font-bold mb-2">Your Financial Health Report</h1>
            <p className="text-muted-foreground">Based on your submitted financial data</p>
          </div>

          <HealthScoreMeter 
            score={analysis.healthScore}
            zScore={analysis.zScore}
            riskLevel={analysis.riskLevel}
          />

          <div className={`text-center mt-6 text-2xl font-bold ${getRiskColor(analysis.riskLevel)}`}>
            {analysis.riskLevel.toUpperCase()}
          </div>
        </Card>

        {/* Key Metrics Overview */}
        <div className="grid md:grid-cols-4 gap-4 mb-8">
          <MetricCard 
            title="Profitability"
            value={analysis.ratios.profitability.netMargin}
            status={analysis.ratios.profitability.netMargin > 10 ? "good" : analysis.ratios.profitability.netMargin > 5 ? "moderate" : "poor"}
            icon={TrendingUp}
          />
          <MetricCard 
            title="Liquidity"
            value={analysis.ratios.liquidity.currentRatio}
            status={analysis.ratios.liquidity.currentRatio > 1.5 ? "good" : analysis.ratios.liquidity.currentRatio > 1 ? "moderate" : "poor"}
            suffix="x"
          />
          <MetricCard 
            title="Leverage"
            value={analysis.ratios.leverage.debtToEquity}
            status={analysis.ratios.leverage.debtToEquity < 1 ? "good" : analysis.ratios.leverage.debtToEquity < 2 ? "moderate" : "poor"}
            suffix="x"
          />
          <MetricCard 
            title="Efficiency"
            value={analysis.ratios.efficiency.returnOnAssets}
            status={analysis.ratios.efficiency.returnOnAssets > 8 ? "good" : analysis.ratios.efficiency.returnOnAssets > 4 ? "moderate" : "poor"}
          />
        </div>

        {/* Detailed Ratios */}
        <div className="grid md:grid-cols-2 gap-6 mb-8">
          <Card className="p-6 shadow-medium">
            <h3 className="text-xl font-semibold mb-4 flex items-center gap-2">
              <CheckCircle className="w-5 h-5 text-success" />
              Strengths
            </h3>
            <ul className="space-y-2">
              {analysis.strengths.map((strength, index) => (
                <li key={index} className="flex items-start gap-2">
                  <span className="text-success mt-1">•</span>
                  <span>{strength}</span>
                </li>
              ))}
            </ul>
          </Card>

          <Card className="p-6 shadow-medium">
            <h3 className="text-xl font-semibold mb-4 flex items-center gap-2">
              <AlertTriangle className="w-5 h-5 text-warning" />
              Areas for Improvement
            </h3>
            <ul className="space-y-2">
              {analysis.weaknesses.map((weakness, index) => (
                <li key={index} className="flex items-start gap-2">
                  <span className="text-warning mt-1">•</span>
                  <span>{weakness}</span>
                </li>
              ))}
            </ul>
          </Card>
        </div>

        {/* AI Insights */}
        <Card className="p-6 mb-8 shadow-medium bg-gradient-primary text-primary-foreground">
          <h3 className="text-xl font-semibold mb-4">AI Analysis & Insights</h3>
          <p className="mb-4">{analysis.aiInsights}</p>
        </Card>

        {/* Recommendations */}
        <div className="mb-8">
          <h2 className="text-2xl font-bold mb-6">Recommended Actions</h2>
          <div className="grid gap-4">
            {analysis.recommendations.immediate.map((rec, index) => (
              <RecommendationCard 
                key={`immediate-${index}`}
                priority="immediate"
                title={rec}
                timeframe="Next 30 days"
              />
            ))}
            {analysis.recommendations.quarterly.map((rec, index) => (
              <RecommendationCard 
                key={`quarterly-${index}`}
                priority="quarterly"
                title={rec}
                timeframe="Next 90 days"
              />
            ))}
          </div>
        </div>

        {/* CTA */}
        <Card className="p-8 text-center shadow-medium">
          <h3 className="text-2xl font-bold mb-4">Ready to Improve Your Financial Health?</h3>
          <p className="text-muted-foreground mb-6">
            Run a new assessment to track your progress or share this report with your financial advisor.
          </p>
          <div className="flex gap-4 justify-center">
            <Button onClick={() => navigate("/assessment")} className="bg-gradient-primary">
              Run New Assessment
            </Button>
            <Button variant="outline" className="flex items-center gap-2">
              <Download className="w-4 h-4" />
              Download Report
            </Button>
          </div>
        </Card>
      </div>
    </div>
  );
};

export default Results;
