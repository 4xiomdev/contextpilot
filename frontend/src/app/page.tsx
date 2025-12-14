"use client";

import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Database,
  FileText,
  Globe,
  Layers,
  TrendingUp,
  CheckCircle,
  XCircle,
  Clock,
} from "lucide-react";

function StatCard({
  title,
  value,
  description,
  icon: Icon,
  trend,
}: {
  title: string;
  value: string | number;
  description?: string;
  icon: React.ElementType;
  trend?: "up" | "down" | "neutral";
}) {
  return (
    <Card className="bg-zinc-900/50 border-zinc-800 hover:border-zinc-700 transition-colors">
      <CardHeader className="flex flex-row items-center justify-between pb-2">
        <CardTitle className="text-sm font-medium text-zinc-400">
          {title}
        </CardTitle>
        <Icon className="w-4 h-4 text-zinc-500" />
      </CardHeader>
      <CardContent>
        <div className="text-3xl font-bold tracking-tight">{value}</div>
        {description && (
          <p className="text-xs text-zinc-500 mt-1 flex items-center gap-1">
            {trend === "up" && <TrendingUp className="w-3 h-3 text-emerald-500" />}
            {description}
          </p>
        )}
      </CardContent>
    </Card>
  );
}

function CrawlStatusCard({
  completed,
  failed,
  running,
}: {
  completed: number;
  failed: number;
  running: number;
}) {
  return (
    <Card className="bg-zinc-900/50 border-zinc-800 col-span-2">
      <CardHeader>
        <CardTitle className="text-sm font-medium text-zinc-400">
          Crawl Status
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-3 gap-4">
          <div className="flex items-center gap-3 p-3 rounded-lg bg-emerald-500/10 border border-emerald-500/20">
            <CheckCircle className="w-5 h-5 text-emerald-500" />
            <div>
              <p className="text-2xl font-bold">{completed}</p>
              <p className="text-xs text-zinc-500">Completed</p>
            </div>
          </div>
          <div className="flex items-center gap-3 p-3 rounded-lg bg-red-500/10 border border-red-500/20">
            <XCircle className="w-5 h-5 text-red-500" />
            <div>
              <p className="text-2xl font-bold">{failed}</p>
              <p className="text-xs text-zinc-500">Failed</p>
            </div>
          </div>
          <div className="flex items-center gap-3 p-3 rounded-lg bg-amber-500/10 border border-amber-500/20">
            <Clock className="w-5 h-5 text-amber-500" />
            <div>
              <p className="text-2xl font-bold">{running}</p>
              <p className="text-xs text-zinc-500">Running</p>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

export default function DashboardPage() {
  const { data: stats, isLoading } = useQuery({
    queryKey: ["stats"],
    queryFn: () => api.getStats(),
    refetchInterval: 10000, // Refresh every 10s
  });

  const { data: sources } = useQuery({
    queryKey: ["sources"],
    queryFn: () => api.getSources(),
  });

  if (isLoading) {
    return (
      <div className="space-y-6">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Dashboard</h1>
          <p className="text-zinc-500 mt-1">
            Overview of your context augmentation layer
          </p>
        </div>
        <div className="grid grid-cols-4 gap-4">
          {[...Array(4)].map((_, i) => (
            <Skeleton key={i} className="h-32 bg-zinc-800" />
          ))}
        </div>
      </div>
    );
  }

  const dbStats = stats?.database;
  const indexStats = stats?.index;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Dashboard</h1>
        <p className="text-zinc-500 mt-1">
          Overview of your context augmentation layer
        </p>
      </div>

      {/* Main stats */}
      <div className="grid grid-cols-4 gap-4">
        <StatCard
          title="Total Chunks"
          value={dbStats?.indexed_docs.total.toLocaleString() ?? "0"}
          description="Document chunks indexed"
          icon={Layers}
        />
        <StatCard
          title="Sources"
          value={dbStats?.indexed_docs.sources ?? 0}
          description="Unique documentation sources"
          icon={Globe}
        />
        <StatCard
          title="Normalized Docs"
          value={dbStats?.normalized_docs.total ?? 0}
          description="Synthesized documents"
          icon={FileText}
        />
        <StatCard
          title="Vectors"
          value={indexStats?.total_vectors?.toLocaleString() ?? "0"}
          description="In Pinecone index"
          icon={Database}
        />
      </div>

      {/* Crawl status */}
      <CrawlStatusCard
        completed={dbStats?.crawl_jobs.completed ?? 0}
        failed={dbStats?.crawl_jobs.failed ?? 0}
        running={dbStats?.crawl_jobs.running ?? 0}
      />

      {/* Recent sources */}
      <Card className="bg-zinc-900/50 border-zinc-800">
        <CardHeader>
          <CardTitle>Indexed Sources</CardTitle>
          <CardDescription>
            Documentation sources in your context layer
          </CardDescription>
        </CardHeader>
        <CardContent>
          {sources?.sources && sources.sources.length > 0 ? (
            <div className="space-y-2">
              {sources.sources.slice(0, 5).map((source, i) => (
                <div
                  key={i}
                  className="flex items-center justify-between p-3 rounded-lg bg-zinc-800/50 border border-zinc-700/50"
                >
                  <div className="flex items-center gap-3">
                    <Globe className="w-4 h-4 text-zinc-500" />
                    <div>
                      <p className="text-sm font-medium truncate max-w-md">
                        {source.title || source.source_url}
                      </p>
                      <p className="text-xs text-zinc-500 truncate max-w-md">
                        {source.source_url}
                      </p>
                    </div>
                  </div>
                  <div className="text-right">
                    <p className="text-sm font-medium">{source.chunks} chunks</p>
                    <p className="text-xs text-zinc-500">
                      {new Date(source.last_indexed * 1000).toLocaleDateString()}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-8 text-zinc-500">
              <Globe className="w-8 h-8 mx-auto mb-2 opacity-50" />
              <p>No sources indexed yet</p>
              <p className="text-xs mt-1">Go to Crawl Manager to add documentation</p>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
