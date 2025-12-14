"use client";

import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { api, CrawlJob } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Globe,
  Plus,
  RefreshCw,
  Loader2,
  CheckCircle,
  XCircle,
  Clock,
  Zap,
  Server,
} from "lucide-react";

function StatusBadge({ status }: { status: CrawlJob["status"] }) {
  const variants = {
    pending: { color: "bg-zinc-500/10 text-zinc-400 border-zinc-500/20", icon: Clock },
    running: { color: "bg-amber-500/10 text-amber-400 border-amber-500/20", icon: Loader2 },
    completed: { color: "bg-emerald-500/10 text-emerald-400 border-emerald-500/20", icon: CheckCircle },
    failed: { color: "bg-red-500/10 text-red-400 border-red-500/20", icon: XCircle },
  };

  const variant = variants[status];
  const Icon = variant.icon;

  return (
    <Badge className={`${variant.color} border`}>
      <Icon className={`w-3 h-3 mr-1 ${status === "running" ? "animate-spin" : ""}`} />
      {status}
    </Badge>
  );
}

function MethodBadge({ method }: { method: CrawlJob["method"] }) {
  if (!method) return <span className="text-zinc-500">-</span>;

  const variants = {
    firecrawl: { color: "bg-violet-500/10 text-violet-400 border-violet-500/20", icon: Zap },
    local: { color: "bg-blue-500/10 text-blue-400 border-blue-500/20", icon: Server },
  };

  const variant = variants[method];
  const Icon = variant.icon;

  return (
    <Badge className={`${variant.color} border`}>
      <Icon className="w-3 h-3 mr-1" />
      {method}
    </Badge>
  );
}

export default function CrawlPage() {
  const [urls, setUrls] = useState("");
  const [statusFilter, setStatusFilter] = useState<string>("all");
  const queryClient = useQueryClient();

  const { data: crawls, isLoading } = useQuery({
    queryKey: ["crawls", statusFilter],
    queryFn: () => api.getCrawls(statusFilter === "all" ? undefined : statusFilter),
    refetchInterval: 5000, // Refresh every 5s to see progress
  });

  const crawlMutation = useMutation({
    mutationFn: (url: string) => api.startCrawl(url),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["crawls"] });
      queryClient.invalidateQueries({ queryKey: ["stats"] });
    },
  });

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    const urlList = urls
      .split("\n")
      .map((u) => u.trim())
      .filter((u) => u && u.startsWith("http"));

    for (const url of urlList) {
      await crawlMutation.mutateAsync(url);
    }

    setUrls("");
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Crawl Manager</h1>
        <p className="text-zinc-500 mt-1">
          Add documentation sources to your context layer
        </p>
      </div>

      {/* URL Input */}
      <Card className="bg-zinc-900/50 border-zinc-800">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Plus className="w-5 h-5" />
            Add URLs
          </CardTitle>
          <CardDescription>
            Enter URLs to crawl (one per line). Supports documentation sites, API references, etc.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-4">
            <Textarea
              placeholder="https://docs.example.com/api&#10;https://another-docs.com/guide"
              value={urls}
              onChange={(e) => setUrls(e.target.value)}
              className="min-h-32 bg-zinc-800/50 border-zinc-700 font-mono text-sm"
            />
            <div className="flex items-center justify-between">
              <p className="text-xs text-zinc-500">
                {urls.split("\n").filter((u) => u.trim() && u.startsWith("http")).length} URL(s) ready
              </p>
              <Button
                type="submit"
                disabled={crawlMutation.isPending || !urls.trim()}
                className="bg-violet-600 hover:bg-violet-700"
              >
                {crawlMutation.isPending ? (
                  <>
                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                    Crawling...
                  </>
                ) : (
                  <>
                    <Globe className="w-4 h-4 mr-2" />
                    Start Crawl
                  </>
                )}
              </Button>
            </div>
          </form>
        </CardContent>
      </Card>

      {/* Crawl History */}
      <Card className="bg-zinc-900/50 border-zinc-800">
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle>Crawl History</CardTitle>
              <CardDescription>Recent crawl jobs and their status</CardDescription>
            </div>
            <div className="flex items-center gap-2">
              <Select value={statusFilter} onValueChange={setStatusFilter}>
                <SelectTrigger className="w-32 bg-zinc-800 border-zinc-700">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All</SelectItem>
                  <SelectItem value="running">Running</SelectItem>
                  <SelectItem value="completed">Completed</SelectItem>
                  <SelectItem value="failed">Failed</SelectItem>
                </SelectContent>
              </Select>
              <Button
                variant="outline"
                size="icon"
                onClick={() => queryClient.invalidateQueries({ queryKey: ["crawls"] })}
                className="border-zinc-700"
              >
                <RefreshCw className="w-4 h-4" />
              </Button>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <div className="text-center py-8 text-zinc-500">
              <Loader2 className="w-6 h-6 mx-auto animate-spin" />
            </div>
          ) : crawls?.jobs && crawls.jobs.length > 0 ? (
            <Table>
              <TableHeader>
                <TableRow className="border-zinc-800 hover:bg-transparent">
                  <TableHead className="text-zinc-400">URL</TableHead>
                  <TableHead className="text-zinc-400">Status</TableHead>
                  <TableHead className="text-zinc-400">Method</TableHead>
                  <TableHead className="text-zinc-400">Chunks</TableHead>
                  <TableHead className="text-zinc-400">Time</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {crawls.jobs.map((job) => (
                  <TableRow key={job.id} className="border-zinc-800">
                    <TableCell className="font-mono text-sm max-w-md truncate">
                      {job.url}
                    </TableCell>
                    <TableCell>
                      <StatusBadge status={job.status} />
                    </TableCell>
                    <TableCell>
                      <MethodBadge method={job.method} />
                    </TableCell>
                    <TableCell>
                      {job.chunks_count > 0 ? (
                        <span className="text-emerald-400">{job.chunks_count}</span>
                      ) : (
                        <span className="text-zinc-500">-</span>
                      )}
                    </TableCell>
                    <TableCell className="text-zinc-500 text-sm">
                      {new Date(job.created_at * 1000).toLocaleString()}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          ) : (
            <div className="text-center py-8 text-zinc-500">
              <Globe className="w-8 h-8 mx-auto mb-2 opacity-50" />
              <p>No crawl jobs yet</p>
              <p className="text-xs mt-1">Add URLs above to get started</p>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}


