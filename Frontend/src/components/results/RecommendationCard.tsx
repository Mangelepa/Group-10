import { Card } from "@/components/ui/card";
import { Clock, AlertCircle } from "lucide-react";

interface RecommendationCardProps {
  priority: "immediate" | "quarterly";
  title: string;
  timeframe: string;
}

const RecommendationCard = ({ priority, title, timeframe }: RecommendationCardProps) => {
  const isPriority = priority === "immediate";

  return (
    <Card className={`p-4 ${isPriority ? "border-warning bg-warning/5" : "border-border"}`}>
      <div className="flex items-start gap-3">
        <div className={`mt-0.5 ${isPriority ? "text-warning" : "text-muted-foreground"}`}>
          {isPriority ? <AlertCircle className="w-5 h-5" /> : <Clock className="w-5 h-5" />}
        </div>
        <div className="flex-1">
          <div className="flex items-center gap-2 mb-1">
            <span className={`text-xs font-bold uppercase ${isPriority ? "text-warning" : "text-muted-foreground"}`}>
              {priority === "immediate" ? "Immediate Action" : "Quarterly Goal"}
            </span>
            <span className="text-xs text-muted-foreground">• {timeframe}</span>
          </div>
          <p className="font-medium">{title}</p>
        </div>
      </div>
    </Card>
  );
};

export default RecommendationCard;
