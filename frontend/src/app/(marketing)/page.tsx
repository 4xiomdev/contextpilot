import Link from "next/link";
import { Button } from "@/components/ui/button";
import { ArrowRight, Database, Globe, Zap, Shield, GitBranch, Terminal } from "lucide-react";

export default function LandingPage() {
  return (
    <div className="flex flex-col gap-20 pb-20">
      {/* Hero Section */}
      <section className="relative pt-32 pb-20 px-4 overflow-hidden">
        <div className="container mx-auto text-center relative z-10">
          <div className="inline-flex items-center rounded-full border border-violet-500/20 bg-violet-500/10 px-3 py-1 text-sm font-medium text-violet-300 mb-8 backdrop-blur-sm">
            <span className="flex h-2 w-2 rounded-full bg-violet-400 mr-2 animate-pulse"></span>
            ContextPilot v1.0 is now available
          </div>
          <h1 className="text-5xl md:text-7xl font-bold tracking-tight mb-8 bg-clip-text text-transparent bg-gradient-to-b from-white to-zinc-500">
            The Context Layer for <br />
            <span className="text-violet-500">Intelligent Agents</span>
          </h1>
          <p className="text-xl text-zinc-400 max-w-2xl mx-auto mb-10 leading-relaxed">
            Give your AI agents infinite long-term memory and real-time knowledge.
            Open source, self-hosted, and built for the future of LLM applications.
          </p>
          <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
            <Link href="/sign-up">
              <Button size="lg" className="h-12 px-8 text-base bg-violet-600 hover:bg-violet-700">
                Start Building Free
                <ArrowRight className="ml-2 w-4 h-4" />
              </Button>
            </Link>
            <Link href="https://github.com/contextpilot/contextpilot" target="_blank">
              <Button variant="outline" size="lg" className="h-12 px-8 text-base border-zinc-700 bg-zinc-900/50 hover:bg-zinc-900">
                <GitBranch className="mr-2 w-4 h-4" />
                View on GitHub
              </Button>
            </Link>
          </div>
        </div>
        
        {/* Background Gradients */}
        <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[1000px] h-[600px] bg-violet-600/20 rounded-full blur-[120px] -z-10 opacity-50" />
      </section>

      {/* Feature Grid */}
      <section className="container mx-auto px-4">
        <div className="grid md:grid-cols-3 gap-8">
          <FeatureCard 
            icon={Globe}
            title="Universal Crawling"
            description="Ingest documentation, websites, and APIs. Automatically keep your context fresh with scheduled re-crawls."
          />
          <FeatureCard 
            icon={Database}
            title="Vector Database"
            description="Built-in semantic search powered by high-performance vector embeddings. Find exactly what your agent needs."
          />
          <FeatureCard 
            icon={Zap}
            title="MCP Native"
            description="Fully compatible with the Model Context Protocol. Connect to Claude, Gemini, and other LLMs instantly."
          />
          <FeatureCard 
            icon={Terminal}
            title="Developer First"
            description="Simple CLI and API interfaces. Built by developers, for developers. No black boxes."
          />
          <FeatureCard 
            icon={Shield}
            title="Self-Hosted"
            description="Keep your data on your infrastructure. Docker-ready and easy to deploy anywhere."
          />
          <FeatureCard 
            icon={GitBranch}
            title="Open Source"
            description="MIT Licensed. Contribute, fork, and extend. Join our growing community of AI engineers."
          />
        </div>
      </section>

      {/* CTA Section */}
      <section className="container mx-auto px-4">
        <div className="bg-gradient-to-br from-violet-900/50 to-zinc-900 border border-violet-500/20 rounded-3xl p-12 text-center relative overflow-hidden">
          <div className="relative z-10">
            <h2 className="text-3xl md:text-4xl font-bold mb-6">Ready to upgrade your agents?</h2>
            <p className="text-zinc-400 max-w-xl mx-auto mb-8">
              Join thousands of developers building the next generation of context-aware AI applications.
            </p>
            <Link href="/sign-up">
              <Button size="lg" className="bg-white text-violet-900 hover:bg-zinc-100 font-bold h-12 px-8">
                Get Started Now
              </Button>
            </Link>
          </div>
          <div className="absolute top-0 left-0 w-full h-full bg-[url('/grid.svg')] opacity-10" />
        </div>
      </section>
    </div>
  );
}

function FeatureCard({ icon: Icon, title, description }: { icon: any, title: string, description: string }) {
  return (
    <div className="p-6 rounded-2xl bg-zinc-900/50 border border-zinc-800 hover:border-violet-500/50 transition-colors group">
      <div className="w-12 h-12 bg-zinc-800 rounded-lg flex items-center justify-center mb-4 group-hover:bg-violet-500/20 transition-colors">
        <Icon className="w-6 h-6 text-zinc-400 group-hover:text-violet-400 transition-colors" />
      </div>
      <h3 className="text-xl font-bold mb-2 text-zinc-100">{title}</h3>
      <p className="text-zinc-400 leading-relaxed">
        {description}
      </p>
    </div>
  );
}
