import { Card } from "@/components/ui/card";
import FinancialInput from "./FinancialInput";
import { CashFlow } from "@/types/financial";

interface CashFlowStepProps {
  data: CashFlow;
  onChange: (data: CashFlow) => void;
}

const CashFlowStep = ({ data, onChange }: CashFlowStepProps) => {
  const handleChange = (field: keyof CashFlow, value: number) => {
    onChange({ ...data, [field]: value });
  };

  return (
    <Card className="p-6 shadow-medium">
      <div className="mb-6">
        <h2 className="text-2xl font-bold mb-2">Cash Flow Statement</h2>
        <p className="text-muted-foreground">
          Enter your cash flow information. This shows how cash moves in and out of your business.
        </p>
      </div>

      <div className="space-y-6">
        <FinancialInput
          label="Operating Cash Flow"
          tooltip="Cash generated from your main business activities (sales minus operating expenses)"
          value={data.operatingCashFlow}
          onChange={(value) => handleChange("operatingCashFlow", value)}
          placeholder="250000"
        />

        <FinancialInput
          label="Capital Expenditures"
          tooltip="Money spent on long-term assets like equipment, buildings, or technology"
          value={data.capitalExpenditures}
          onChange={(value) => handleChange("capitalExpenditures", value)}
          placeholder="100000"
        />

        <FinancialInput
          label="Financing Activities"
          tooltip="Cash from loans, investments, or dividends paid (can be negative)"
          value={data.financingActivities}
          onChange={(value) => handleChange("financingActivities", value)}
          placeholder="50000"
        />
      </div>

      <div className="mt-6 p-4 bg-muted rounded-lg">
        <p className="text-sm text-muted-foreground">
          <strong>Important:</strong> Operating cash flow is often considered more important than net income 
          because it shows actual cash your business generates, not just accounting profits.
        </p>
      </div>

      <div className="mt-4 p-4 bg-success/10 border border-success/20 rounded-lg">
        <p className="text-sm font-medium text-success-foreground">
          Ready to see your results! Click "View Results" to get your financial health assessment.
        </p>
      </div>
    </Card>
  );
};

export default CashFlowStep;
