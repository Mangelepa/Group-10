import { Card } from "@/components/ui/card";
import FinancialInput from "./FinancialInput";
import { BalanceSheet } from "@/types/financial";

interface BalanceSheetStepProps {
  data: BalanceSheet;
  onChange: (data: BalanceSheet) => void;
}

const BalanceSheetStep = ({ data, onChange }: BalanceSheetStepProps) => {
  const handleChange = (field: keyof BalanceSheet, value: number) => {
    onChange({ ...data, [field]: value });
  };

  return (
    <Card className="p-6 shadow-medium">
      <div className="mb-6">
        <h2 className="text-2xl font-bold mb-2">Balance Sheet</h2>
        <p className="text-muted-foreground">
          Enter your assets, liabilities, and equity. This shows what your business owns and owes at a specific point in time.
        </p>
      </div>

      <div className="space-y-6">
        <div>
          <h3 className="text-lg font-semibold mb-4">Assets</h3>
          <div className="space-y-4">
            <FinancialInput
              label="Total Assets"
              tooltip="Everything your business owns (current + long-term assets)"
              value={data.totalAssets}
              onChange={(value) => handleChange("totalAssets", value)}
              placeholder="2000000"
            />

            <FinancialInput
              label="Current Assets"
              tooltip="Assets that can be converted to cash within one year (cash, inventory, receivables)"
              value={data.currentAssets}
              onChange={(value) => handleChange("currentAssets", value)}
              placeholder="800000"
            />

            <FinancialInput
              label="Cash & Equivalents"
              tooltip="Cash in bank accounts and highly liquid investments"
              value={data.cashAndEquivalents}
              onChange={(value) => handleChange("cashAndEquivalents", value)}
              placeholder="200000"
            />
          </div>
        </div>

        <div>
          <h3 className="text-lg font-semibold mb-4">Liabilities</h3>
          <div className="space-y-4">
            <FinancialInput
              label="Total Liabilities"
              tooltip="Everything your business owes (current + long-term debt)"
              value={data.totalLiabilities}
              onChange={(value) => handleChange("totalLiabilities", value)}
              placeholder="1000000"
            />

            <FinancialInput
              label="Current Liabilities"
              tooltip="Debts due within one year (accounts payable, short-term loans)"
              value={data.currentLiabilities}
              onChange={(value) => handleChange("currentLiabilities", value)}
              placeholder="400000"
            />
          </div>
        </div>

        <div>
          <h3 className="text-lg font-semibold mb-4">Equity</h3>
          <FinancialInput
            label="Stockholders' Equity"
            tooltip="Net worth of your business (Assets - Liabilities)"
            value={data.stockholdersEquity}
            onChange={(value) => handleChange("stockholdersEquity", value)}
            placeholder="1000000"
          />
        </div>
      </div>

      <div className="mt-6 p-4 bg-muted rounded-lg">
        <p className="text-sm text-muted-foreground">
          <strong>Quick Check:</strong> Total Assets should equal Total Liabilities + Stockholders Equity. 
          This is the fundamental accounting equation.
        </p>
      </div>
    </Card>
  );
};

export default BalanceSheetStep;
