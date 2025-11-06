import { useMutation } from '@tanstack/react-query';
import { analyzeFinancials } from '@/lib/api';
import { FinancialData, FinancialAnalysis } from '@/types/financial';
import { useToast } from '@/hooks/use-toast';

export function useFinancialAnalysis() {
  const { toast } = useToast();

  return useMutation<FinancialAnalysis, Error, FinancialData>({
    mutationFn: analyzeFinancials,
    onError: (error) => {
      toast({
        title: "Analysis failed",
        description: error.message,
        variant: "destructive",
      });
    },
    onSuccess: () => {
      toast({
        title: "Analysis completed",
        description: "Your financial health has been analyzed successfully",
      });
    }
  });
}