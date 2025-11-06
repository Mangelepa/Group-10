import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { HelpCircle } from "lucide-react";

interface FinancialInputProps {
  label: string;
  tooltip: string;
  value: number;
  onChange: (value: number) => void;
  placeholder?: string;
}

const FinancialInput = ({ label, tooltip, value, onChange, placeholder }: FinancialInputProps) => {
  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const rawValue = e.target.value.replace(/[^0-9.-]/g, "");
    const numValue = parseFloat(rawValue) || 0;
    onChange(numValue);
  };

  const formatValue = (num: number) => {
    if (num === 0) return "";
    return num.toLocaleString("en-US");
  };

  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2">
        <Label htmlFor={label} className="text-base font-medium">
          {label}
        </Label>
        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger asChild>
              <HelpCircle className="w-4 h-4 text-muted-foreground cursor-help" />
            </TooltipTrigger>
            <TooltipContent className="max-w-xs">
              <p>{tooltip}</p>
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>
      </div>
      <div className="relative">
        <span className="absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground">$</span>
        <Input
          id={label}
          type="text"
          value={formatValue(value)}
          onChange={handleChange}
          placeholder={placeholder}
          className="pl-7 text-lg"
        />
      </div>
    </div>
  );
};

export default FinancialInput;
