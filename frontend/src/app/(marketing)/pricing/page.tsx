import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Check } from "lucide-react";

export default function PricingPage() {
  return (
    <div className="py-20 px-4">
      <div className="container mx-auto">
        <div className="text-center mb-16">
          <h1 className="text-4xl font-bold mb-4">Simple, Transparent Pricing</h1>
          <p className="text-zinc-400 max-w-2xl mx-auto">
            Choose the plan that fits your needs. All plans include core features and community support.
          </p>
        </div>

        <div className="grid md:grid-cols-3 gap-8 max-w-6xl mx-auto">
          {/* Open Source */}
          <PricingCard
            title="Open Source"
            price="$0"
            description="Perfect for hobbyists and local development."
            features={[
              "Self-hosted via Docker",
              "Unlimited local usage",
              "Community support",
              "Basic vector search",
              "File system crawling"
            ]}
            buttonText="Download Now"
            buttonVariant="outline"
            href="https://github.com/contextpilot/contextpilot"
          />

          {/* Hosted Pro */}
          <PricingCard
            title="Hosted Cloud"
            price="$29"
            period="/month"
            description="For developers who want a managed solution."
            highlighted={true}
            features={[
              "Everything in Open Source",
              "Managed infrastructure",
              "50,000 vector limit",
              "API access",
              "Priority support",
              "Scheduled crawls"
            ]}
            buttonText="Start Free Trial"
            buttonVariant="default"
            href="/sign-up?plan=pro"
          />

          {/* Team */}
          <PricingCard
            title="Team"
            price="$99"
            period="/month"
            description="Collaborative workspace for AI teams."
            features={[
              "Everything in Hosted",
              "Unlimited vectors",
              "Team management",
              "SSO / SAML",
              "Dedicated support",
              "Custom connectors"
            ]}
            buttonText="Contact Sales"
            buttonVariant="outline"
            href="mailto:sales@contextpilot.com"
          />
        </div>
      </div>
    </div>
  );
}

function PricingCard({
  title,
  price,
  period,
  description,
  features,
  buttonText,
  buttonVariant = "outline",
  highlighted = false,
  href
}: {
  title: string;
  price: string;
  period?: string;
  description: string;
  features: string[];
  buttonText: string;
  buttonVariant?: "default" | "outline";
  highlighted?: boolean;
  href: string;
}) {
  return (
    <div className={`rounded-2xl p-8 flex flex-col ${
      highlighted 
        ? "bg-zinc-900 border-2 border-violet-500 relative" 
        : "bg-zinc-900/50 border border-zinc-800"
    }`}>
      {highlighted && (
        <div className="absolute -top-4 left-1/2 -translate-x-1/2 bg-violet-600 text-white text-xs font-bold px-3 py-1 rounded-full uppercase tracking-wide">
          Most Popular
        </div>
      )}
      <div className="mb-8">
        <h3 className="text-lg font-medium text-zinc-400 mb-2">{title}</h3>
        <div className="flex items-baseline gap-1">
          <span className="text-4xl font-bold text-zinc-100">{price}</span>
          {period && <span className="text-zinc-500">{period}</span>}
        </div>
        <p className="text-zinc-500 mt-4 text-sm leading-relaxed">{description}</p>
      </div>
      
      <div className="flex-1 space-y-4 mb-8">
        {features.map((feature, i) => (
          <div key={i} className="flex items-start gap-3 text-sm text-zinc-300">
            <Check className="w-5 h-5 text-violet-500 shrink-0" />
            <span>{feature}</span>
          </div>
        ))}
      </div>

      <Link href={href} className="w-full">
        <Button 
          className={`w-full h-12 font-medium ${
            buttonVariant === "default" 
              ? "bg-violet-600 hover:bg-violet-700" 
              : "border-zinc-700 hover:bg-zinc-800"
          }`}
          variant={buttonVariant}
        >
          {buttonText}
        </Button>
      </Link>
    </div>
  );
}
