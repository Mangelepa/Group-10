import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import IncomeStatementStep from "@/components/assessment/IncomeStatementStep";
import BalanceSheetStep from "@/components/assessment/BalanceSheetStep";
import CashFlowStep from "@/components/assessment/CashFlowStep";
import { FinancialData } from "@/types/financial";
import { ChevronLeft, ChevronRight } from "lucide-react";

const Assessment = () => {
  const navigate = useNavigate();
  const [currentStep, setCurrentStep] = useState(1);
  const [financialData, setFinancialData] = useState<FinancialData>({
    incomeStatement: {
      totalRevenue: 0,
      grossProfit: 0,
      operatingIncome: 0,
      netIncome: 0,
    },
    balanceSheet: {
      totalAssets: 0,
      currentAssets: 0,
      totalLiabilities: 0,
      currentLiabilities: 0,
      stockholdersEquity: 0,
      cashAndEquivalents: 0,
    },
    cashFlow: {
      operatingCashFlow: 0,
      capitalExpenditures: 0,
      financingActivities: 0,
    },
  });

  const totalSteps = 3;
  const progress = (currentStep / totalSteps) * 100;

  const handleNext = () => {
    if (currentStep < totalSteps) {
      setCurrentStep(currentStep + 1);
    } else {
      navigate("/results", { state: { financialData } });
    }
  };

  const handleBack = () => {
    if (currentStep > 1) {
      setCurrentStep(currentStep - 1);
    } else {
      navigate("/");
    }
  };

  const updateFinancialData = (section: keyof FinancialData, data: any) => {
    setFinancialData((prev) => ({
      ...prev,
      [section]: data,
    }));
  };

  const stepTitles = [
    "Income Statement",
    "Balance Sheet",
    "Cash Flow",
  ];

  return (
    <div className="min-h-screen bg-background">
      <div className="container mx-auto px-4 py-8 max-w-3xl">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold mb-2">Financial Health Assessment</h1>
          <p className="text-muted-foreground">Step {currentStep} of {totalSteps}: {stepTitles[currentStep - 1]}</p>
        </div>

        {/* Progress Bar */}
        <div className="mb-8">
          <Progress value={progress} className="h-2" />
          <div className="flex justify-between mt-2 text-sm text-muted-foreground">
            {stepTitles.map((title, index) => (
              <span 
                key={index}
                className={currentStep > index + 1 ? "text-success font-medium" : currentStep === index + 1 ? "text-primary font-medium" : ""}
              >
                {title}
              </span>
            ))}
          </div>
        </div>

        {/* Step Content */}
        <div className="mb-8">
          {currentStep === 1 && (
            <IncomeStatementStep 
              data={financialData.incomeStatement}
              onChange={(data) => updateFinancialData("incomeStatement", data)}
            />
          )}
          {currentStep === 2 && (
            <BalanceSheetStep 
              data={financialData.balanceSheet}
              onChange={(data) => updateFinancialData("balanceSheet", data)}
            />
          )}
          {currentStep === 3 && (
            <CashFlowStep 
              data={financialData.cashFlow}
              onChange={(data) => updateFinancialData("cashFlow", data)}
            />
          )}
        </div>

        {/* Navigation */}
        <div className="flex justify-between items-center">
          <Button 
            variant="outline" 
            onClick={handleBack}
            className="flex items-center gap-2"
          >
            <ChevronLeft className="w-4 h-4" />
            {currentStep === 1 ? "Back to Home" : "Previous"}
          </Button>
          
          <Button 
            onClick={handleNext}
            className="bg-gradient-primary flex items-center gap-2"
          >
            {currentStep === totalSteps ? "View Results" : "Next"}
            <ChevronRight className="w-4 h-4" />
          </Button>
        </div>
      </div>
    </div>
  );
};

export default Assessment;
