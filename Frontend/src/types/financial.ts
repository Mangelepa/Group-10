export interface IncomeStatement {
  totalRevenue: number;
  grossProfit: number;
  operatingIncome: number;
  netIncome: number;
}

export interface BalanceSheet {
  totalAssets: number;
  currentAssets: number;
  totalLiabilities: number;
  currentLiabilities: number;
  stockholdersEquity: number;
  cashAndEquivalents: number;
}

export interface CashFlow {
  operatingCashFlow: number;
  capitalExpenditures: number;
  financingActivities: number;
}

export interface FinancialData {
  incomeStatement: IncomeStatement;
  balanceSheet: BalanceSheet;
  cashFlow: CashFlow;
}

export interface FinancialRatios {
  profitability: {
    grossMargin: number;
    operatingMargin: number;
    netMargin: number;
  };
  liquidity: {
    currentRatio: number;
    quickRatio: number;
  };
  leverage: {
    debtToEquity: number;
    debtToAssets: number;
  };
  efficiency: {
    returnOnAssets: number;
    returnOnEquity: number;
  };
}

export interface FinancialAnalysis {
  healthScore: number;
  zScore: number;
  riskLevel: "Healthy" | "Watch" | "At Risk";
  ratios: FinancialRatios;
  strengths: string[];
  weaknesses: string[];
  aiInsights: string;
  recommendations: {
    immediate: string[];
    quarterly: string[];
  };
}
