import { Card } from "@/components/ui/card";
import { LucideIcon } from "lucide-react";

interface MetricCardProps {
  title: string;
  value: number;
  status: "good" | "moderate" | "poor";
  suffix?: string;
  icon?: LucideIcon;
}

const MetricCard = ({ title, value, status, suffix = "%", icon: Icon }: MetricCardProps) => {
  const getStatusColor = (status: string) => {
    switch (status) {
      case "good": return "text-success bg-success/10 border-success/20";
      case "moderate": return "text-warning bg-warning/10 border-warning/20";
      case "poor": return "text-destructive bg-destructive/10 border-destructive/20";
      default: return "text-foreground bg-muted border-border";
    }
  };

  const getStatusLabel = (status: string) => {
    switch (status) {
      case "good": return "Good";
      case "moderate": return "Moderate";
      case "poor": return "Needs Attention";
      default: return "";
    }
  };

  return (
    <Card className={`p-4 border-2 ${getStatusColor(status)} transition-all hover:shadow-medium`}>
      <div className="flex items-start justify-between mb-2">
        <div className="text-sm font-medium text-muted-foreground">{title}</div>
        {Icon && <Icon className="w-4 h-4 opacity-50" />}
      </div>
      <div className="text-3xl font-bold mb-1">
        {value.toFixed(suffix === "x" ? 2 : 1)}{suffix}
      </div>
      <div className="text-xs font-medium">{getStatusLabel(status)}</div>
    </Card>
  );
};

export default MetricCard;
