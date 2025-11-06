import { Card } from "@/components/ui/card";

interface HealthScoreMeterProps {
  score: number;
  zScore: number;
  riskLevel: string;
}

const HealthScoreMeter = ({ score, zScore, riskLevel }: HealthScoreMeterProps) => {
  const getColorClass = (score: number) => {
    if (score >= 70) return "text-success";
    if (score >= 40) return "text-warning";
    return "text-destructive";
  };

  const getBarColor = (score: number) => {
    if (score >= 70) return "bg-gradient-success";
    if (score >= 40) return "bg-warning";
    return "bg-destructive";
  };

  return (
    <div className="space-y-6">
      <div className="text-center">
        <div className={`text-6xl font-bold mb-2 ${getColorClass(score)}`}>
          {score}/100
        </div>
        <div className="text-sm text-muted-foreground">Your Financial Health Score</div>
      </div>

      <div className="relative h-4 bg-muted rounded-full overflow-hidden">
        <div 
          className={`absolute top-0 left-0 h-full ${getBarColor(score)} transition-all duration-1000`}
          style={{ width: `${score}%` }}
        />
      </div>

      <div className="grid grid-cols-2 gap-4 pt-4">
        <Card className="p-4 bg-muted/50">
          <div className="text-sm text-muted-foreground mb-1">Altman Z-Score</div>
          <div className="text-2xl font-bold">{zScore.toFixed(2)}</div>
          <div className="text-xs text-muted-foreground mt-1">
            {zScore > 2.99 ? "Safe Zone" : zScore > 1.81 ? "Grey Zone" : "Distress Zone"}
          </div>
        </Card>

        <Card className="p-4 bg-muted/50">
          <div className="text-sm text-muted-foreground mb-1">Risk Assessment</div>
          <div className={`text-2xl font-bold ${getColorClass(score)}`}>{riskLevel}</div>
          <div className="text-xs text-muted-foreground mt-1">
            Current Status
          </div>
        </Card>
      </div>
    </div>
  );
};

export default HealthScoreMeter;
