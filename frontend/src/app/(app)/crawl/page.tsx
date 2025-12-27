"use client";

import { useState, useMemo, useEffect } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { api, CrawlJob, CreateSourceInput, CrawlPlanResult } from "@/lib/api";
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
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Progress } from "@/components/ui/progress";
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
  FolderPlus,
  Power,
  PowerOff,
  Sparkles,
  GitBranch,
  ShieldCheck,
  Check,
  X,
  HelpCircle,
  Folder,
  FileText,
  ChevronRight,
  ChevronDown,
  Search,
  AlertTriangle,
  Info,
} from "lucide-react";

// Types for URL classification
type UrlStatus = "keep" | "drop" | "maybe";
interface ClassifiedUrl {
  id: string;
  url: string;
  status: UrlStatus;
}

// ===================== URL Item Component =====================
function UrlItem({
  item,
  onChangeStatus,
}: {
  item: ClassifiedUrl;
  onChangeStatus: (id: string, status: UrlStatus) => void;
}) {
  const statusColors = {
    keep: "bg-emerald-500/5 border-emerald-500/20",
    drop: "bg-red-500/5 border-red-500/20",
    maybe: "bg-amber-500/5 border-amber-500/20",
  };

  // Extract path from URL for compact display
  const displayPath = useMemo(() => {
    try {
      const url = new URL(item.url);
      return url.pathname + url.search;
    } catch {
      return item.url;
    }
  }, [item.url]);

  return (
    <div
      className={`flex items-center gap-2 px-2 py-1.5 rounded border ${statusColors[item.status]} group hover:bg-zinc-800/50 transition-colors`}
    >
      <span
        className="font-mono text-xs text-zinc-400 truncate flex-1"
        title={item.url}
      >
        {displayPath}
      </span>
      <div className="flex items-center gap-0.5 opacity-60 group-hover:opacity-100 transition-opacity">
        <button
          onClick={() => onChangeStatus(item.id, "keep")}
          className={`p-1 rounded hover:bg-emerald-500/20 transition-colors ${
            item.status === "keep" ? "text-emerald-400" : "text-zinc-500 hover:text-emerald-400"
          }`}
          title="Keep"
        >
          <Check className="w-3 h-3" />
        </button>
        <button
          onClick={() => onChangeStatus(item.id, "drop")}
          className={`p-1 rounded hover:bg-red-500/20 transition-colors ${
            item.status === "drop" ? "text-red-400" : "text-zinc-500 hover:text-red-400"
          }`}
          title="Drop"
        >
          <X className="w-3 h-3" />
        </button>
        <button
          onClick={() => onChangeStatus(item.id, "maybe")}
          className={`p-1 rounded hover:bg-amber-500/20 transition-colors ${
            item.status === "maybe" ? "text-amber-400" : "text-zinc-500 hover:text-amber-400"
          }`}
          title="Maybe"
        >
          <HelpCircle className="w-3 h-3" />
        </button>
      </div>
    </div>
  );
}

// ===================== URL List Panel =====================
function UrlListPanel({
  items,
  status,
  onChangeStatus,
  emptyMessage,
}: {
  items: ClassifiedUrl[];
  status: UrlStatus;
  onChangeStatus: (id: string, status: UrlStatus) => void;
  emptyMessage: string;
}) {
  const [search, setSearch] = useState("");

  const filtered = useMemo(() => {
    const statusFiltered = items.filter((i) => i.status === status);
    if (!search.trim()) return statusFiltered;
    const q = search.toLowerCase();
    return statusFiltered.filter((i) => i.url.toLowerCase().includes(q));
  }, [items, status, search]);

  const count = items.filter((i) => i.status === status).length;

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2">
        <div className="relative flex-1">
          <Search className="absolute left-2 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-zinc-500" />
          <Input
            placeholder="Filter URLs..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="pl-7 h-8 text-xs bg-zinc-800/50 border-zinc-700"
          />
        </div>
        <Badge variant="outline" className="text-xs border-zinc-700">
          {count} URL{count !== 1 ? "s" : ""}
        </Badge>
      </div>
      <ScrollArea className="h-56 rounded-md border border-zinc-800 bg-zinc-900/30 p-2">
        {filtered.length > 0 ? (
          <div className="space-y-1">
            {filtered.slice(0, 200).map((item) => (
              <UrlItem key={item.id} item={item} onChangeStatus={onChangeStatus} />
            ))}
            {filtered.length > 200 && (
              <div className="text-xs text-zinc-500 text-center py-2">
                Showing 200 of {filtered.length} URLs
              </div>
            )}
          </div>
        ) : (
          <div className="text-xs text-zinc-500 text-center py-8">{emptyMessage}</div>
        )}
      </ScrollArea>
    </div>
  );
}

// ===================== Tree View Component =====================
interface TreeNode {
  path: string;
  count: number;
  examples: string[];
  children: TreeNode[];
  isExpanded?: boolean;
}

function buildTree(branches: Array<{ path_prefix: string; count: number; examples?: string[] }>): TreeNode[] {
  if (!branches || branches.length === 0) return [];

  // Sort branches by path for consistent ordering
  const sorted = [...branches].sort((a, b) => a.path_prefix.localeCompare(b.path_prefix));

  // Build a flat list of nodes
  const nodes: TreeNode[] = sorted.map((b) => ({
    path: b.path_prefix,
    count: b.count,
    examples: b.examples || [],
    children: [],
    isExpanded: true,
  }));

  return nodes;
}

function TreeNodeRow({
  node,
  depth,
  isLast,
}: {
  node: TreeNode;
  depth: number;
  isLast: boolean;
}) {
  const [expanded, setExpanded] = useState(true);
  const hasChildren = node.children.length > 0;

  return (
    <div>
      <div
        className="flex items-center gap-1.5 py-1 px-2 hover:bg-zinc-800/50 rounded transition-colors cursor-pointer group"
        style={{ paddingLeft: `${depth * 16 + 8}px` }}
        onClick={() => hasChildren && setExpanded(!expanded)}
      >
        {hasChildren ? (
          expanded ? (
            <ChevronDown className="w-3 h-3 text-zinc-500" />
          ) : (
            <ChevronRight className="w-3 h-3 text-zinc-500" />
          )
        ) : (
          <span className="w-3" />
        )}
        <Folder className="w-3.5 h-3.5 text-violet-400" />
        <span className="font-mono text-xs text-zinc-300 flex-1">{node.path}</span>
        <Badge
          variant="outline"
          className="text-[10px] px-1.5 py-0 h-4 border-zinc-700 text-zinc-400"
        >
          {node.count}
        </Badge>
      </div>
      {hasChildren && expanded && (
        <div>
          {node.children.map((child, idx) => (
            <TreeNodeRow
              key={child.path}
              node={child}
              depth={depth + 1}
              isLast={idx === node.children.length - 1}
            />
          ))}
        </div>
      )}
    </div>
  );
}

function TreeView({ report }: { report: Record<string, unknown> }) {
  const tree = useMemo(() => {
    // Try to get tree data from report
    const treeData = (report as any)?.tree || (report as any)?.summary?.tree || [];
    if (Array.isArray(treeData)) {
      return buildTree(treeData);
    }
    // If tree is already an array of branches
    if (Array.isArray((report as any)?.tree)) {
      return buildTree((report as any).tree);
    }
    return [];
  }, [report]);

  // Also show branches if tree is empty
  const branches = useMemo(() => {
    const branchData = (report as any)?.branches || [];
    if (tree.length === 0 && Array.isArray(branchData)) {
      return buildTree(branchData);
    }
    return [];
  }, [report, tree.length]);

  const displayNodes = tree.length > 0 ? tree : branches;

  if (displayNodes.length === 0) {
    // Fallback: show derived_rules if available
    const derivedRules = (report as any)?.derived_rules;
    if (derivedRules) {
      return (
        <ScrollArea className="h-64 rounded-md border border-zinc-800 bg-zinc-900/30 p-3">
          <div className="space-y-4">
            {derivedRules.allow_prefixes?.length > 0 && (
              <div>
                <div className="text-xs font-medium text-zinc-400 mb-2">Allow Prefixes</div>
                <div className="flex flex-wrap gap-1">
                  {derivedRules.allow_prefixes.map((p: string) => (
                    <Badge key={p} className="bg-emerald-500/10 text-emerald-400 border-emerald-500/20 font-mono text-xs">
                      {p}
                    </Badge>
                  ))}
                </div>
              </div>
            )}
            {derivedRules.deny_prefixes?.length > 0 && (
              <div>
                <div className="text-xs font-medium text-zinc-400 mb-2">Deny Prefixes</div>
                <div className="flex flex-wrap gap-1">
                  {derivedRules.deny_prefixes.map((p: string) => (
                    <Badge key={p} className="bg-red-500/10 text-red-400 border-red-500/20 font-mono text-xs">
                      {p}
                    </Badge>
                  ))}
                </div>
              </div>
            )}
          </div>
        </ScrollArea>
      );
    }
    return (
      <div className="h-64 rounded-md border border-zinc-800 bg-zinc-900/30 p-3 flex items-center justify-center text-zinc-500 text-sm">
        No tree structure available
      </div>
    );
  }

  return (
    <ScrollArea className="h-64 rounded-md border border-zinc-800 bg-zinc-900/30">
      <div className="py-2">
        {displayNodes.map((node, idx) => (
          <TreeNodeRow key={node.path} node={node} depth={0} isLast={idx === displayNodes.length - 1} />
        ))}
      </div>
    </ScrollArea>
  );
}

// ===================== Rules Panel =====================
function RulesPanel({ rules }: { rules: Record<string, unknown> }) {
  const allowPrefixes = (rules as any)?.allow_prefixes || [];
  const denyPrefixes = (rules as any)?.deny_prefixes || [];
  const maybePrefixes = (rules as any)?.maybe_prefixes || [];
  const queryPolicy = (rules as any)?.query_param_policy || {};

  return (
    <ScrollArea className="h-64 rounded-md border border-zinc-800 bg-zinc-900/30 p-4">
      <div className="space-y-4">
        {/* Allow Prefixes */}
        <div>
          <div className="flex items-center gap-2 mb-2">
            <Check className="w-3.5 h-3.5 text-emerald-400" />
            <span className="text-xs font-medium text-zinc-300">Allow Prefixes</span>
            <Badge variant="outline" className="text-[10px] h-4 border-zinc-700">
              {allowPrefixes.length}
            </Badge>
          </div>
          <div className="flex flex-wrap gap-1.5">
            {allowPrefixes.length > 0 ? (
              allowPrefixes.map((p: string) => (
                <Badge
                  key={p}
                  className="bg-emerald-500/10 text-emerald-400 border-emerald-500/20 font-mono text-xs"
                >
                  {p}
                </Badge>
              ))
            ) : (
              <span className="text-xs text-zinc-500">None specified</span>
            )}
          </div>
        </div>

        {/* Deny Prefixes */}
        <div>
          <div className="flex items-center gap-2 mb-2">
            <X className="w-3.5 h-3.5 text-red-400" />
            <span className="text-xs font-medium text-zinc-300">Deny Prefixes</span>
            <Badge variant="outline" className="text-[10px] h-4 border-zinc-700">
              {denyPrefixes.length}
            </Badge>
          </div>
          <div className="flex flex-wrap gap-1.5">
            {denyPrefixes.length > 0 ? (
              denyPrefixes.map((p: string) => (
                <Badge
                  key={p}
                  className="bg-red-500/10 text-red-400 border-red-500/20 font-mono text-xs"
                >
                  {p}
                </Badge>
              ))
            ) : (
              <span className="text-xs text-zinc-500">None specified</span>
            )}
          </div>
        </div>

        {/* Maybe Prefixes */}
        {maybePrefixes.length > 0 && (
          <div>
            <div className="flex items-center gap-2 mb-2">
              <HelpCircle className="w-3.5 h-3.5 text-amber-400" />
              <span className="text-xs font-medium text-zinc-300">Maybe Prefixes</span>
              <Badge variant="outline" className="text-[10px] h-4 border-zinc-700">
                {maybePrefixes.length}
              </Badge>
            </div>
            <div className="flex flex-wrap gap-1.5">
              {maybePrefixes.map((p: string) => (
                <Badge
                  key={p}
                  className="bg-amber-500/10 text-amber-400 border-amber-500/20 font-mono text-xs"
                >
                  {p}
                </Badge>
              ))}
            </div>
          </div>
        )}

        {/* Query Param Policy */}
        <div className="border-t border-zinc-800 pt-3">
          <div className="text-xs font-medium text-zinc-300 mb-2">Query Parameter Policy</div>
          <div className="grid grid-cols-2 gap-3 text-xs">
            <div>
              <div className="text-zinc-500 mb-1">Keep Keys</div>
              <div className="flex flex-wrap gap-1">
                {(queryPolicy.keep_keys || []).length > 0 ? (
                  (queryPolicy.keep_keys || []).map((k: string) => (
                    <Badge key={k} variant="outline" className="font-mono text-[10px] border-zinc-700">
                      {k}
                    </Badge>
                  ))
                ) : (
                  <span className="text-zinc-500">None</span>
                )}
              </div>
            </div>
            <div>
              <div className="text-zinc-500 mb-1">Drop Keys</div>
              <div className="flex flex-wrap gap-1">
                {(queryPolicy.drop_keys || []).slice(0, 5).map((k: string) => (
                  <Badge key={k} variant="outline" className="font-mono text-[10px] border-zinc-700">
                    {k}
                  </Badge>
                ))}
                {(queryPolicy.drop_keys || []).length > 5 && (
                  <Badge variant="outline" className="text-[10px] border-zinc-700">
                    +{queryPolicy.drop_keys.length - 5}
                  </Badge>
                )}
              </div>
            </div>
          </div>
          <div className="mt-2 text-xs text-zinc-500">
            Default: <span className="text-zinc-400">{queryPolicy.default || "drop_unknown_keys"}</span>
          </div>
        </div>
      </div>
    </ScrollArea>
  );
}

// ===================== Report Panel =====================
function ReportPanel({ report }: { report: Record<string, unknown> }) {
  const summary = (report as any)?.summary || {};
  const decisions = (report as any)?.decisions || {};
  const crawlPhases = (report as any)?.crawl_phases || [];
  const audit = (report as any)?.audit || {};

  return (
    <ScrollArea className="h-64 rounded-md border border-zinc-800 bg-zinc-900/30 p-4">
      <div className="space-y-4">
        {/* Summary */}
        {Object.keys(summary).length > 0 && (
          <div>
            <div className="flex items-center gap-2 mb-2">
              <Info className="w-3.5 h-3.5 text-blue-400" />
              <span className="text-xs font-medium text-zinc-300">Summary</span>
            </div>
            <div className="bg-zinc-900/50 rounded p-2.5 space-y-1">
              {summary.total_urls_seen && (
                <div className="text-xs">
                  <span className="text-zinc-500">Total URLs:</span>{" "}
                  <span className="text-zinc-300">{summary.total_urls_seen}</span>
                </div>
              )}
              {summary.recommended_strategy && (
                <div className="text-xs">
                  <span className="text-zinc-500">Strategy:</span>{" "}
                  <span className="text-zinc-300">{summary.recommended_strategy}</span>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Decisions */}
        {Object.keys(decisions).length > 0 && (
          <div>
            <div className="flex items-center gap-2 mb-2">
              <GitBranch className="w-3.5 h-3.5 text-violet-400" />
              <span className="text-xs font-medium text-zinc-300">Decisions</span>
            </div>
            <div className="space-y-2">
              {["keep", "drop", "maybe"].map((key) => {
                const decision = decisions[key];
                if (!decision || (Array.isArray(decision) && decision.length === 0)) return null;
                const items = Array.isArray(decision) ? decision : [decision];
                return (
                  <div key={key} className="bg-zinc-900/50 rounded p-2.5">
                    <div className="flex items-center gap-2 mb-1">
                      {key === "keep" && <Check className="w-3 h-3 text-emerald-400" />}
                      {key === "drop" && <X className="w-3 h-3 text-red-400" />}
                      {key === "maybe" && <HelpCircle className="w-3 h-3 text-amber-400" />}
                      <span className="text-xs font-medium text-zinc-300 capitalize">{key}</span>
                    </div>
                    {items.slice(0, 3).map((item: any, idx: number) => (
                      <div key={idx} className="text-xs text-zinc-400 ml-5">
                        {typeof item === "string" ? item : item.why || item.target || JSON.stringify(item)}
                      </div>
                    ))}
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Crawl Phases */}
        {crawlPhases.length > 0 && (
          <div>
            <div className="flex items-center gap-2 mb-2">
              <Sparkles className="w-3.5 h-3.5 text-amber-400" />
              <span className="text-xs font-medium text-zinc-300">Crawl Phases</span>
            </div>
            <div className="space-y-1.5">
              {crawlPhases.map((phase: any, idx: number) => (
                <div key={idx} className="flex items-start gap-2 text-xs">
                  <Badge variant="outline" className="text-[10px] h-4 px-1.5 border-zinc-700">
                    {idx + 1}
                  </Badge>
                  <span className="text-zinc-400">
                    {typeof phase === "string" ? phase : phase.description || phase.name || JSON.stringify(phase)}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Audit */}
        {Object.keys(audit).length > 0 && (
          <div>
            <div className="flex items-center gap-2 mb-2">
              <AlertTriangle className="w-3.5 h-3.5 text-amber-400" />
              <span className="text-xs font-medium text-zinc-300">Audit</span>
            </div>
            <div className="bg-zinc-900/50 rounded p-2.5 space-y-1">
              {audit.risks && (
                <div className="text-xs">
                  <span className="text-zinc-500">Risks:</span>{" "}
                  <span className="text-amber-400">
                    {Array.isArray(audit.risks) ? audit.risks.join(", ") : audit.risks}
                  </span>
                </div>
              )}
              {audit.spot_checks && (
                <div className="text-xs">
                  <span className="text-zinc-500">Spot Checks:</span>{" "}
                  <span className="text-zinc-400">
                    {Array.isArray(audit.spot_checks) ? audit.spot_checks.length : "Available"}
                  </span>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Fallback if nothing else is shown */}
        {Object.keys(summary).length === 0 &&
          Object.keys(decisions).length === 0 &&
          crawlPhases.length === 0 &&
          Object.keys(audit).length === 0 && (
            <div className="text-xs text-zinc-500 text-center py-4">
              Report data is available but in an unrecognized format.
              <pre className="mt-2 text-left text-[10px] text-zinc-600 overflow-auto max-h-32">
                {JSON.stringify(report, null, 2).slice(0, 500)}...
              </pre>
            </div>
          )}
      </div>
    </ScrollArea>
  );
}


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
  const [addSourceOpen, setAddSourceOpen] = useState(false);
  const [planResult, setPlanResult] = useState<CrawlPlanResult | null>(null);
  const [planSourceId, setPlanSourceId] = useState<string | null>(null);
  const [planOpen, setPlanOpen] = useState(false);
  const [planStep, setPlanStep] = useState(0);
  const [planProgress, setPlanProgress] = useState(0);
  const [planApproved, setPlanApproved] = useState(false);
  const [urlClassifications, setUrlClassifications] = useState<ClassifiedUrl[]>([]);
  const [newSource, setNewSource] = useState<CreateSourceInput>({
    name: "",
    base_url: "",
    sitemap_url: "",
    crawl_frequency: "weekly",
    max_pages: 500,
    description: "",
  });
  const queryClient = useQueryClient();

  const { data: crawls, isLoading } = useQuery({
    queryKey: ["crawls", statusFilter],
    queryFn: () => api.getCrawls(statusFilter === "all" ? undefined : statusFilter),
    refetchInterval: 5000, // Refresh every 5s to see progress
  });

  const { data: registrySources, isLoading: registryLoading } = useQuery({
    queryKey: ["registry-sources"],
    queryFn: () => api.getRegistrySources(),
    refetchInterval: 15000,
  });
  const activeSource = registrySources?.sources?.find((s) => s.id === planSourceId) ?? null;

  const crawlMutation = useMutation({
    mutationFn: (url: string) => api.startCrawl(url),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["crawls"] });
      queryClient.invalidateQueries({ queryKey: ["stats"] });
    },
  });

  const registryCrawlMutation = useMutation({
    mutationFn: (sourceId: string) => api.triggerRegistryCrawl(sourceId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["registry-sources"] });
      queryClient.invalidateQueries({ queryKey: ["crawls"] });
      queryClient.invalidateQueries({ queryKey: ["stats"] });
    },
  });

  const registryDeleteMutation = useMutation({
    mutationFn: (sourceId: string) => api.deleteRegistrySource(sourceId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["registry-sources"] });
      queryClient.invalidateQueries({ queryKey: ["stats"] });
    },
  });

  const planMutation = useMutation({
    mutationFn: (sourceId: string) => api.generateCrawlPlan(sourceId, 5000),
    onSuccess: (data) => {
      setPlanResult(data);
      setPlanApproved(false);
      setPlanStep(4);
      setPlanProgress(100);

      // Initialize URL classifications from samples
      const classifications: ClassifiedUrl[] = [];
      if (data.samples) {
        (data.samples.keep || []).forEach((s) => {
          classifications.push({ id: s.id, url: s.url, status: "keep" });
        });
        (data.samples.drop || []).forEach((s) => {
          classifications.push({ id: s.id, url: s.url, status: "drop" });
        });
        (data.samples.maybe || []).forEach((s) => {
          classifications.push({ id: s.id, url: s.url, status: "maybe" });
        });
      }
      setUrlClassifications(classifications);
    },
  });

  const crawlFromPlanMutation = useMutation({
    mutationFn: (sourceId: string) => api.crawlFromPlan(sourceId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["crawls"] });
      queryClient.invalidateQueries({ queryKey: ["stats"] });
    },
  });

  const approvePlanMutation = useMutation({
    mutationFn: ({ sourceId, planId }: { sourceId: string; planId: string | number }) =>
      api.approveCrawlPlan(sourceId, planId),
    onSuccess: () => {
      setPlanApproved(true);
    },
  });

  const startPlan = async (sourceId: string, forceRegenerate = false) => {
    setPlanSourceId(sourceId);
    setPlanResult(null);
    setPlanApproved(false);
    setPlanOpen(true);
    setPlanStep(0);
    setPlanProgress(5);

    // First, check if there's an existing plan we can show
    if (!forceRegenerate) {
      try {
        const { plan } = await api.getCrawlPlan(sourceId);
        if (plan && plan.id) {
          // We have an existing plan - show it directly
          console.log("[CrawlPlan] Found existing plan:", plan);
          const existingResult: CrawlPlanResult = {
            plan_id: plan.id as number | string,
            source_id: sourceId,
            counts: (plan.counts as CrawlPlanResult["counts"]) || {
              total_urls_seen: 0,
              kept_urls: 0,
              dropped_urls: 0,
              maybe_urls: 0,
            },
            rules: (plan.rules as Record<string, unknown>) || {},
            report: (plan.report as Record<string, unknown>) || {},
            samples: plan.samples as CrawlPlanResult["samples"],
          };
          setPlanResult(existingResult);
          setPlanApproved(plan.status === "approved");
          setPlanStep(4);
          setPlanProgress(100);
          return;
        }
      } catch (err) {
        console.log("[CrawlPlan] No existing plan found, generating new one");
      }
    }

    // No existing plan or force regenerate - generate a new one
    // Fake stepper progression while backend works
    const startedAt = Date.now();
    const tick = () => {
      const elapsed = Date.now() - startedAt;
      if (!planMutation.isPending) return;

      if (elapsed > 600) setPlanStep(1);
      if (elapsed > 1600) setPlanStep(2);
      if (elapsed > 2600) setPlanStep(3);

      // Ease towards 90% while pending
      setPlanProgress((p) => Math.min(90, p + 3));
      requestAnimationFrame(tick);
    };

    planMutation.mutate(sourceId);
    requestAnimationFrame(tick);
  };

  const createSourceMutation = useMutation({
    mutationFn: (source: CreateSourceInput) => api.createRegistrySource(source),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["registry-sources"] });
      setAddSourceOpen(false);
      setNewSource({
        name: "",
        base_url: "",
        sitemap_url: "",
        crawl_frequency: "weekly",
        max_pages: 500,
        description: "",
      });
    },
  });

  const toggleSourceMutation = useMutation({
    mutationFn: ({ sourceId, enabled }: { sourceId: string; enabled: boolean }) =>
      api.toggleRegistrySource(sourceId, enabled),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["registry-sources"] });
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
              placeholder="https://docs.example.com/api; https://another-docs.com/guide"
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

      {/* Source Registry */}
      <Card className="bg-zinc-900/50 border-zinc-800">
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle>Source Registry</CardTitle>
              <CardDescription>Managed documentation sources and crawl status</CardDescription>
            </div>
            <div className="flex items-center gap-2">
              <Dialog open={addSourceOpen} onOpenChange={setAddSourceOpen}>
                <DialogTrigger asChild>
                  <Button className="bg-violet-600 hover:bg-violet-700">
                    <FolderPlus className="w-4 h-4 mr-2" />
                    Add Source
                  </Button>
                </DialogTrigger>
                <DialogContent className="bg-zinc-900 border-zinc-800">
                  <DialogHeader>
                    <DialogTitle>Add Documentation Source</DialogTitle>
                    <DialogDescription>
                      Add a new documentation source to crawl and index.
                    </DialogDescription>
                  </DialogHeader>
                  <div className="space-y-4 py-4">
                    <div className="space-y-2">
                      <label className="text-sm font-medium text-zinc-300">Name *</label>
                      <Input
                        placeholder="e.g., Gemini API Docs"
                        value={newSource.name}
                        onChange={(e) => setNewSource({ ...newSource, name: e.target.value })}
                        className="bg-zinc-800 border-zinc-700"
                      />
                    </div>
                    <div className="space-y-2">
                      <label className="text-sm font-medium text-zinc-300">Base URL *</label>
                      <Input
                        placeholder="https://ai.google.dev/gemini-api/docs"
                        value={newSource.base_url}
                        onChange={(e) => setNewSource({ ...newSource, base_url: e.target.value })}
                        className="bg-zinc-800 border-zinc-700"
                      />
                    </div>
                    <div className="space-y-2">
                      <label className="text-sm font-medium text-zinc-300">Sitemap URL (optional)</label>
                      <Input
                        placeholder="https://ai.google.dev/sitemap.xml"
                        value={newSource.sitemap_url || ""}
                        onChange={(e) => setNewSource({ ...newSource, sitemap_url: e.target.value })}
                        className="bg-zinc-800 border-zinc-700"
                      />
                      <p className="text-xs text-zinc-500">Leave empty to auto-discover from robots.txt</p>
                    </div>
                    <div className="grid grid-cols-2 gap-4">
                      <div className="space-y-2">
                        <label className="text-sm font-medium text-zinc-300">Crawl Frequency</label>
                        <Select
                          value={newSource.crawl_frequency}
                          onValueChange={(v) => setNewSource({ ...newSource, crawl_frequency: v })}
                        >
                          <SelectTrigger className="bg-zinc-800 border-zinc-700">
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="daily">Daily</SelectItem>
                            <SelectItem value="weekly">Weekly</SelectItem>
                            <SelectItem value="monthly">Monthly</SelectItem>
                            <SelectItem value="manual">Manual only</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>
                      <div className="space-y-2">
                        <label className="text-sm font-medium text-zinc-300">Max Pages</label>
                        <Input
                          type="number"
                          value={newSource.max_pages}
                          onChange={(e) => setNewSource({ ...newSource, max_pages: parseInt(e.target.value) || 500 })}
                          className="bg-zinc-800 border-zinc-700"
                        />
                      </div>
                    </div>
                    <div className="space-y-2">
                      <label className="text-sm font-medium text-zinc-300">Description (optional)</label>
                      <Textarea
                        placeholder="Brief description of this documentation source..."
                        value={newSource.description || ""}
                        onChange={(e) => setNewSource({ ...newSource, description: e.target.value })}
                        className="bg-zinc-800 border-zinc-700 min-h-20"
                      />
                    </div>
                  </div>
                  <DialogFooter>
                    <Button
                      variant="outline"
                      onClick={() => setAddSourceOpen(false)}
                      className="border-zinc-700"
                    >
                      Cancel
                    </Button>
                    <Button
                      onClick={() => createSourceMutation.mutate(newSource)}
                      disabled={!newSource.name || !newSource.base_url || createSourceMutation.isPending}
                      className="bg-violet-600 hover:bg-violet-700"
                    >
                      {createSourceMutation.isPending ? (
                        <>
                          <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                          Adding...
                        </>
                      ) : (
                        "Add Source"
                      )}
                    </Button>
                  </DialogFooter>
                </DialogContent>
              </Dialog>
              <Button
                variant="outline"
                size="icon"
                onClick={() => queryClient.invalidateQueries({ queryKey: ["registry-sources"] })}
                className="border-zinc-700"
              >
                <RefreshCw className="w-4 h-4" />
              </Button>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          {registryLoading ? (
            <div className="text-center py-8 text-zinc-500">
              <Loader2 className="w-6 h-6 mx-auto animate-spin" />
            </div>
          ) : registrySources?.sources && registrySources.sources.length > 0 ? (
            <Table>
              <TableHeader>
                <TableRow className="border-zinc-800 hover:bg-transparent">
                  <TableHead className="text-zinc-400">Source</TableHead>
                  <TableHead className="text-zinc-400">Status</TableHead>
                  <TableHead className="text-zinc-400">Frequency</TableHead>
                  <TableHead className="text-zinc-400">Chunks</TableHead>
                  <TableHead className="text-zinc-400">Actions</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {registrySources.sources.map((source) => (
                  <TableRow key={source.id} className="border-zinc-800">
                    <TableCell className="max-w-md">
                      <div className="flex items-center gap-2">
                        {source.is_enabled ? (
                          <Power className="w-3 h-3 text-emerald-400" />
                        ) : (
                          <PowerOff className="w-3 h-3 text-zinc-500" />
                        )}
                        <div>
                          <div className="font-medium">{source.name || source.base_url}</div>
                          <div className="text-xs text-zinc-500 truncate max-w-xs">{source.base_url}</div>
                        </div>
                      </div>
                    </TableCell>
                    <TableCell>
                      <Badge
                        className={
                          source.health_status === "healthy"
                            ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/20"
                            : source.health_status === "error"
                            ? "bg-red-500/10 text-red-400 border-red-500/20"
                            : source.health_status === "stale"
                            ? "bg-amber-500/10 text-amber-400 border-amber-500/20"
                            : "bg-zinc-500/10 text-zinc-400 border-zinc-500/20"
                        }
                      >
                        {source.health_status}
                      </Badge>
                    </TableCell>
                    <TableCell className="text-sm text-zinc-400">
                      {source.crawl_frequency}
                    </TableCell>
                    <TableCell className="text-sm text-zinc-300">
                      {source.chunks_count ?? 0}
                    </TableCell>
                    <TableCell>
                      <div className="flex items-center gap-2">
                        <Button
                          size="sm"
                          className="bg-violet-600 hover:bg-violet-700"
                          disabled={planMutation.isPending || !source.is_enabled}
                          onClick={() => startPlan(source.id)}
                        >
                          {planMutation.isPending ? (
                            <Loader2 className="w-4 h-4 animate-spin" />
                          ) : (
                            "Plan & Crawl"
                          )}
                        </Button>
                        <Button
                          size="sm"
                          variant="outline"
                          className="border-emerald-700 text-emerald-400 hover:bg-emerald-900/30"
                          disabled={!source.is_enabled || crawlFromPlanMutation.isPending}
                          onClick={() => {
                            crawlFromPlanMutation.mutate(source.id, {
                              onError: (err) => {
                                alert(`Cannot refresh: ${err instanceof Error ? err.message : String(err)}`);
                              },
                            });
                          }}
                        >
                          {crawlFromPlanMutation.isPending ? (
                            <Loader2 className="w-4 h-4 animate-spin" />
                          ) : (
                            <>
                              <RefreshCw className="w-3 h-3 mr-1" />
                              Refresh
                            </>
                          )}
                        </Button>
                        <Button
                          size="sm"
                          variant="outline"
                          className="border-zinc-700"
                          disabled={toggleSourceMutation.isPending}
                          onClick={() => toggleSourceMutation.mutate({ sourceId: source.id, enabled: !source.is_enabled })}
                        >
                          {source.is_enabled ? "Disable" : "Enable"}
                        </Button>
                        <Button
                          size="sm"
                          variant="destructive"
                          disabled={registryDeleteMutation.isPending}
                          onClick={() => {
                            if (window.confirm(`Delete ${source.name || source.base_url}?`)) {
                              registryDeleteMutation.mutate(source.id);
                            }
                          }}
                        >
                          Delete
                        </Button>
                      </div>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          ) : (
            <div className="text-center py-8 text-zinc-500">
              <FolderPlus className="w-8 h-8 mx-auto mb-2 opacity-50" />
              <p>No sources in the registry yet</p>
              <p className="text-xs mt-1">Click &quot;Add Source&quot; to add your first documentation source</p>
            </div>
          )}
        </CardContent>
      </Card>

      <Dialog open={planOpen} onOpenChange={setPlanOpen}>
        <DialogContent className="max-w-4xl bg-zinc-950 border-zinc-800">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <GitBranch className="w-5 h-5" />
              Crawl Plan Review
            </DialogTitle>
            <DialogDescription>
              {activeSource ? (
                <span>
                  {activeSource.name || "Source"} ·{" "}
                  <span className="font-mono text-xs">{activeSource.base_url}</span>
                </span>
              ) : (
                "Generate a plan, review what will be crawled, then approve."
              )}
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-4">
            <div className="flex items-center justify-between gap-4">
              <div className="flex items-center gap-3 text-sm text-zinc-300">
                <Sparkles className="w-4 h-4 text-violet-400" />
                <div className="space-y-1">
                  <div className="font-medium">Planning steps</div>
                  <div className="text-xs text-zinc-500">
                    {["Discover URL universe", "Build tree summary", "Gemini curation", "Finalize report", "Ready"][planStep]}
                  </div>
                </div>
              </div>
              <div className="w-48">
                <Progress value={planProgress} />
              </div>
            </div>

            {planMutation.isPending ? (
              <div className="rounded-md border border-zinc-800 bg-zinc-900/30 p-4">
                <div className="flex items-center gap-3">
                  <Loader2 className="w-5 h-5 animate-spin text-violet-400" />
                  <div className="text-sm">
                    <div className="font-medium text-zinc-200">Generating crawl plan…</div>
                    <div className="text-xs text-zinc-500">
                      Building a site tree and classifying keep/drop/maybe.
                    </div>
                  </div>
                </div>
              </div>
            ) : planResult ? (
              <Tabs defaultValue="overview" className="w-full">
                <TabsList className="bg-zinc-900 border border-zinc-800">
                  <TabsTrigger value="overview">Overview</TabsTrigger>
                  <TabsTrigger value="tree">Tree</TabsTrigger>
                  <TabsTrigger value="keep" className="gap-1.5">
                    Keep
                    <span className="text-[10px] text-emerald-400">
                      {urlClassifications.filter((u) => u.status === "keep").length}
                    </span>
                  </TabsTrigger>
                  <TabsTrigger value="drop" className="gap-1.5">
                    Drop
                    <span className="text-[10px] text-red-400">
                      {urlClassifications.filter((u) => u.status === "drop").length}
                    </span>
                  </TabsTrigger>
                  <TabsTrigger value="maybe" className="gap-1.5">
                    Maybe
                    <span className="text-[10px] text-amber-400">
                      {urlClassifications.filter((u) => u.status === "maybe").length}
                    </span>
                  </TabsTrigger>
                  <TabsTrigger value="rules">Rules</TabsTrigger>
                  <TabsTrigger value="report">Report</TabsTrigger>
                </TabsList>

                <TabsContent value="overview" className="mt-4 space-y-4">
                  <div className="grid grid-cols-4 gap-3">
                    <Card className="bg-zinc-900/50 border-zinc-800">
                      <CardHeader className="py-3">
                        <CardTitle className="text-sm">Total URLs</CardTitle>
                      </CardHeader>
                      <CardContent className="pt-0 text-2xl font-semibold">
                        {planResult.counts.total_urls_seen}
                      </CardContent>
                    </Card>
                    <Card className="bg-zinc-900/50 border-zinc-800">
                      <CardHeader className="py-3">
                        <CardTitle className="text-sm">Keep</CardTitle>
                      </CardHeader>
                      <CardContent className="pt-0 text-2xl font-semibold text-emerald-400">
                        {planResult.counts.kept_urls}
                      </CardContent>
                    </Card>
                    <Card className="bg-zinc-900/50 border-zinc-800">
                      <CardHeader className="py-3">
                        <CardTitle className="text-sm">Drop</CardTitle>
                      </CardHeader>
                      <CardContent className="pt-0 text-2xl font-semibold text-red-400">
                        {planResult.counts.dropped_urls}
                      </CardContent>
                    </Card>
                    <Card className="bg-zinc-900/50 border-zinc-800">
                      <CardHeader className="py-3">
                        <CardTitle className="text-sm">Maybe</CardTitle>
                      </CardHeader>
                      <CardContent className="pt-0 text-2xl font-semibold text-amber-400">
                        {planResult.counts.maybe_urls}
                      </CardContent>
                    </Card>
                  </div>

                  {/* Status badge and regenerate option */}
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      {planApproved ? (
                        <Badge className="bg-emerald-500/20 text-emerald-400 border-emerald-500/30">
                          <Check className="w-3 h-3 mr-1" />
                          Approved
                        </Badge>
                      ) : (
                        <Badge className="bg-amber-500/20 text-amber-400 border-amber-500/30">
                          <Clock className="w-3 h-3 mr-1" />
                          Pending Approval
                        </Badge>
                      )}
                    </div>
                    <Button
                      size="sm"
                      variant="ghost"
                      className="text-xs text-zinc-400 hover:text-zinc-200"
                      disabled={planMutation.isPending}
                      onClick={() => {
                        if (planSourceId) {
                          startPlan(planSourceId, true);
                        }
                      }}
                    >
                      <RefreshCw className="w-3 h-3 mr-1" />
                      Regenerate Plan
                    </Button>
                  </div>

                  <div className="rounded-md border border-emerald-500/20 bg-emerald-500/5 p-4 space-y-3">
                    <div className="flex items-center gap-2 text-sm font-medium text-zinc-200">
                      <ShieldCheck className="w-4 h-4 text-emerald-400" />
                      Ready to crawl {planResult.counts.kept_urls} URLs
                    </div>
                    <p className="text-xs text-zinc-400">
                      Review the Keep/Drop/Maybe tabs above. When ready, click the button below to start crawling.
                    </p>
                    <Button
                      className="w-full bg-emerald-600 hover:bg-emerald-700"
                      disabled={
                        !planSourceId ||
                        !planResult ||
                        crawlFromPlanMutation.isPending ||
                        approvePlanMutation.isPending
                      }
                      onClick={async () => {
                        if (!planSourceId || !planResult) return;
                        try {
                          console.log("[CrawlPlan] Approving plan:", planResult.plan_id, "for source:", planSourceId);
                          const approveResult = await api.approveCrawlPlan(planSourceId, planResult.plan_id);
                          console.log("[CrawlPlan] Approve result:", approveResult);

                          console.log("[CrawlPlan] Starting crawl from plan...");
                          crawlFromPlanMutation.mutate(planSourceId, {
                            onSuccess: (data) => {
                              console.log("[CrawlPlan] Crawl started successfully:", data);
                            },
                            onError: (err) => {
                              console.error("[CrawlPlan] Crawl failed:", err);
                            },
                          });
                          setPlanOpen(false);
                          // Refresh crawl history to show new jobs
                          queryClient.invalidateQueries({ queryKey: ["crawls"] });
                        } catch (err) {
                          console.error("[CrawlPlan] Failed to approve/crawl:", err);
                          alert(`Failed to start crawl: ${err instanceof Error ? err.message : String(err)}`);
                        }
                      }}
                    >
                      {crawlFromPlanMutation.isPending ? (
                        <>
                          <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                          Starting crawl…
                        </>
                      ) : (
                        <>
                          <CheckCircle className="w-4 h-4 mr-2" />
                          Start Crawling {planResult.counts.kept_urls} URLs
                        </>
                      )}
                    </Button>
                  </div>
                </TabsContent>

                <TabsContent value="tree" className="mt-4">
                  <TreeView report={planResult.report} />
                </TabsContent>

                <TabsContent value="keep" className="mt-4">
                  <UrlListPanel
                    items={urlClassifications}
                    status="keep"
                    onChangeStatus={(id, status) => {
                      setUrlClassifications((prev) =>
                        prev.map((item) =>
                          item.id === id ? { ...item, status } : item
                        )
                      );
                    }}
                    emptyMessage="No URLs marked to keep"
                  />
                </TabsContent>

                <TabsContent value="drop" className="mt-4">
                  <UrlListPanel
                    items={urlClassifications}
                    status="drop"
                    onChangeStatus={(id, status) => {
                      setUrlClassifications((prev) =>
                        prev.map((item) =>
                          item.id === id ? { ...item, status } : item
                        )
                      );
                    }}
                    emptyMessage="No URLs marked to drop"
                  />
                </TabsContent>

                <TabsContent value="maybe" className="mt-4">
                  <UrlListPanel
                    items={urlClassifications}
                    status="maybe"
                    onChangeStatus={(id, status) => {
                      setUrlClassifications((prev) =>
                        prev.map((item) =>
                          item.id === id ? { ...item, status } : item
                        )
                      );
                    }}
                    emptyMessage="No URLs marked as maybe"
                  />
                </TabsContent>

                <TabsContent value="rules" className="mt-4">
                  <RulesPanel rules={planResult.rules} />
                </TabsContent>

                <TabsContent value="report" className="mt-4">
                  <ReportPanel report={planResult.report} />
                </TabsContent>
              </Tabs>
            ) : (
              <div className="rounded-md border border-zinc-800 bg-zinc-900/30 p-4 text-sm text-zinc-400">
                Click “Plan” on a source to generate a crawl plan.
              </div>
            )}
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}
