import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { useNavigate } from "react-router-dom";
import { Activity, TrendingUp, Shield, Zap } from "lucide-react";

const Landing = () => {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen bg-gradient-hero">
      {/* Hero Section */}
      <div className="container mx-auto px-4 py-16">
        <div className="text-center max-w-3xl mx-auto mb-16">
          <div className="inline-flex items-center gap-2 bg-primary/10 text-primary px-4 py-2 rounded-full mb-6">
            <Activity className="w-4 h-4" />
            <span className="text-sm font-medium">Financial Health Assessment</span>
          </div>
          
          <h1 className="text-5xl font-bold text-foreground mb-6">
            Know Your Business's Financial Health in Minutes
          </h1>
          
          <p className="text-xl text-muted-foreground mb-8">
            Get an instant, AI-powered assessment of your company's financial health. 
            No complex jargon—just clear insights and actionable recommendations.
          </p>
          
          <Button 
            size="lg" 
            className="bg-gradient-primary shadow-strong hover:opacity-90 transition-all text-lg px-8 py-6"
            onClick={() => navigate("/assessment")}
          >
            Start Your Free Assessment
          </Button>
        </div>

        {/* Features Grid */}
        <div className="grid md:grid-cols-3 gap-6 max-w-5xl mx-auto mb-16">
          <Card className="p-6 shadow-soft hover:shadow-medium transition-all">
            <div className="w-12 h-12 bg-gradient-primary rounded-lg flex items-center justify-center mb-4">
              <Zap className="w-6 h-6 text-primary-foreground" />
            </div>
            <h3 className="text-lg font-semibold mb-2">Instant Analysis</h3>
            <p className="text-muted-foreground">
              Get your financial health score in under 15 minutes. Input your data and receive immediate insights.
            </p>
          </Card>

          <Card className="p-6 shadow-soft hover:shadow-medium transition-all">
            <div className="w-12 h-12 bg-gradient-success rounded-lg flex items-center justify-center mb-4">
              <TrendingUp className="w-6 h-6 text-success-foreground" />
            </div>
            <h3 className="text-lg font-semibold mb-2">Clear Recommendations</h3>
            <p className="text-muted-foreground">
              Receive prioritized action items to improve your financial position, explained in plain language.
            </p>
          </Card>

          <Card className="p-6 shadow-soft hover:shadow-medium transition-all">
            <div className="w-12 h-12 bg-accent/20 rounded-lg flex items-center justify-center mb-4">
              <Shield className="w-6 h-6 text-accent" />
            </div>
            <h3 className="text-lg font-semibold mb-2">Bank-Ready Reports</h3>
            <p className="text-muted-foreground">
              Professional reports you can share with lenders, investors, or financial advisors.
            </p>
          </Card>
        </div>

        {/* How It Works */}
        <div className="text-center max-w-4xl mx-auto">
          <h2 className="text-3xl font-bold mb-12">How It Works</h2>
          
          <div className="grid md:grid-cols-4 gap-8">
            <div className="relative">
              <div className="w-16 h-16 bg-primary text-primary-foreground rounded-full flex items-center justify-center text-2xl font-bold mx-auto mb-4">
                1
              </div>
              <h4 className="font-semibold mb-2">Input Your Data</h4>
              <p className="text-sm text-muted-foreground">
                Enter key financial figures from your statements
              </p>
            </div>

            <div className="relative">
              <div className="w-16 h-16 bg-primary text-primary-foreground rounded-full flex items-center justify-center text-2xl font-bold mx-auto mb-4">
                2
              </div>
              <h4 className="font-semibold mb-2">Get Your Score</h4>
              <p className="text-sm text-muted-foreground">
                Receive an instant financial health assessment
              </p>
            </div>

            <div className="relative">
              <div className="w-16 h-16 bg-primary text-primary-foreground rounded-full flex items-center justify-center text-2xl font-bold mx-auto mb-4">
                3
              </div>
              <h4 className="font-semibold mb-2">Review Insights</h4>
              <p className="text-sm text-muted-foreground">
                Understand your strengths and improvement areas
              </p>
            </div>

            <div className="relative">
              <div className="w-16 h-16 bg-primary text-primary-foreground rounded-full flex items-center justify-center text-2xl font-bold mx-auto mb-4">
                4
              </div>
              <h4 className="font-semibold mb-2">Take Action</h4>
              <p className="text-sm text-muted-foreground">
                Follow personalized recommendations
              </p>
            </div>
          </div>

          <div className="mt-12">
            <Button 
              size="lg"
              className="bg-gradient-primary shadow-medium hover:opacity-90 transition-all"
              onClick={() => navigate("/assessment")}
            >
              Get Started Now
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Landing;
