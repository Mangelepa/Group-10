import { Card } from "@/components/ui/card";
import FinancialInput from "./FinancialInput";
import { IncomeStatement } from "@/types/financial";

interface IncomeStatementStepProps {
  data: IncomeStatement;
  onChange: (data: IncomeStatement) => void;
}

const IncomeStatementStep = ({ data, onChange }: IncomeStatementStepProps) => {
  const handleChange = (field: keyof IncomeStatement, value: number) => {
    onChange({ ...data, [field]: value });
  };

  return (
    <Card className="p-6 shadow-medium">
      <div className="mb-6">
        <h2 className="text-2xl font-bold mb-2">Income Statement</h2>
        <p className="text-muted-foreground">
          Enter your revenue and profit information. All values should be for the same time period (typically annual or quarterly).
        </p>
      </div>

      <div className="space-y-6">
        <FinancialInput
          label="Total Revenue"
          tooltip="Your total sales for the period"
          value={data.totalRevenue}
          onChange={(value) => handleChange("totalRevenue", value)}
          placeholder="1000000"
        />

        <FinancialInput
          label="Gross Profit"
          tooltip="Revenue minus cost of goods sold (COGS)"
          value={data.grossProfit}
          onChange={(value) => handleChange("grossProfit", value)}
          placeholder="400000"
        />

        <FinancialInput
          label="Operating Income"
          tooltip="Profit from core business operations before interest and taxes (EBIT)"
          value={data.operatingIncome}
          onChange={(value) => handleChange("operatingIncome", value)}
          placeholder="200000"
        />

        <FinancialInput
          label="Net Income"
          tooltip="Final profit after all expenses, interest, and taxes"
          value={data.netIncome}
          onChange={(value) => handleChange("netIncome", value)}
          placeholder="150000"
        />
      </div>

      <div className="mt-6 p-4 bg-muted rounded-lg">
        <p className="text-sm text-muted-foreground">
          <strong>Tip:</strong> You can find these numbers on your income statement (also called profit & loss statement). 
          If you use accounting software like QuickBooks or Xero, these reports are readily available.
        </p>
      </div>
    </Card>
  );
};

export default IncomeStatementStep;
