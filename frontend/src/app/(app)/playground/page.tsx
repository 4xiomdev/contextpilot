"use client";

import { useState } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { motion, AnimatePresence } from "framer-motion";
import {
  FlaskConical,
  Play,
  Copy,
  Check,
  ChevronDown,
  Loader2,
  Terminal,
  Sparkles,
  AlertCircle,
} from "lucide-react";
import { api, McpTool, McpExecutionResult } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { cn } from "@/lib/utils";

interface ExecutionHistory {
  id: string;
  tool: string;
  parameters: Record<string, unknown>;
  result: McpExecutionResult;
  timestamp: Date;
}

export default function PlaygroundPage() {
  const [selectedTool, setSelectedTool] = useState<McpTool | null>(null);
  const [parameters, setParameters] = useState<Record<string, string>>({});
  const [history, setHistory] = useState<ExecutionHistory[]>([]);
  const [copied, setCopied] = useState(false);

  // Fetch available tools
  const { data: toolsData, isLoading: toolsLoading } = useQuery({
    queryKey: ["mcp-tools"],
    queryFn: () => api.getMcpTools(),
  });

  // Execute tool mutation
  const executeMutation = useMutation({
    mutationFn: ({ tool, params }: { tool: string; params: Record<string, unknown> }) =>
      api.executeMcpTool(tool, params),
    onSuccess: (result, variables) => {
      const newEntry: ExecutionHistory = {
        id: Date.now().toString(),
        tool: variables.tool,
        parameters: variables.params,
        result,
        timestamp: new Date(),
      };
      setHistory((prev) => [newEntry, ...prev].slice(0, 20));
    },
  });

  const handleToolSelect = (tool: McpTool) => {
    setSelectedTool(tool);
    // Initialize parameters with defaults
    const defaultParams: Record<string, string> = {};
    Object.entries(tool.parameters).forEach(([key, param]) => {
      if (param.default !== undefined) {
        defaultParams[key] = String(param.default);
      } else {
        defaultParams[key] = "";
      }
    });
    setParameters(defaultParams);
  };

  const handleExecute = () => {
    if (!selectedTool) return;

    // Convert string parameters to appropriate types
    const typedParams: Record<string, unknown> = {};
    Object.entries(parameters).forEach(([key, value]) => {
      const paramDef = selectedTool.parameters[key];
      if (!value && !paramDef?.required) {
        return; // Skip empty optional params
      }
      if (paramDef?.type === "integer") {
        typedParams[key] = parseInt(value, 10) || 0;
      } else {
        typedParams[key] = value;
      }
    });

    executeMutation.mutate({ tool: selectedTool.name, params: typedParams });
  };

  const handleCopy = (text: string) => {
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const latestResult = history[0]?.result;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight flex items-center gap-3">
            <FlaskConical className="w-7 h-7 text-violet-400" />
            MCP Playground
          </h1>
          <p className="text-zinc-500 mt-1">
            Test MCP tools exactly as Claude or Cursor would use them
          </p>
        </div>
        <Badge variant="outline" className="bg-violet-500/10 text-violet-400 border-violet-500/20">
          <Terminal className="w-3 h-3 mr-1" />
          {toolsData?.tools.length || 0} Tools Available
        </Badge>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Tool Selection */}
        <Card className="bg-zinc-900/50 border-zinc-800">
          <CardHeader>
            <CardTitle className="text-sm">Select Tool</CardTitle>
            <CardDescription>Choose an MCP tool to execute</CardDescription>
          </CardHeader>
          <CardContent className="space-y-2">
            {toolsLoading ? (
              <div className="flex items-center justify-center py-8">
                <Loader2 className="w-6 h-6 animate-spin text-zinc-500" />
              </div>
            ) : (
              toolsData?.tools.map((tool) => (
                <motion.button
                  key={tool.name}
                  whileHover={{ scale: 1.01 }}
                  whileTap={{ scale: 0.99 }}
                  onClick={() => handleToolSelect(tool)}
                  className={cn(
                    "w-full text-left p-3 rounded-lg border transition-all duration-200",
                    selectedTool?.name === tool.name
                      ? "bg-violet-500/10 border-violet-500/30 text-violet-400"
                      : "bg-zinc-800/50 border-zinc-700/50 hover:border-zinc-600"
                  )}
                >
                  <div className="font-medium text-sm">{tool.name}</div>
                  <div className="text-xs text-zinc-500 mt-1 line-clamp-2">
                    {tool.description}
                  </div>
                </motion.button>
              ))
            )}
          </CardContent>
        </Card>

        {/* Parameters & Execute */}
        <Card className="bg-zinc-900/50 border-zinc-800">
          <CardHeader>
            <CardTitle className="text-sm">Parameters</CardTitle>
            <CardDescription>
              {selectedTool
                ? `Configure ${selectedTool.name} parameters`
                : "Select a tool to configure"}
            </CardDescription>
          </CardHeader>
          <CardContent>
            <AnimatePresence mode="wait">
              {selectedTool ? (
                <motion.div
                  key={selectedTool.name}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                  className="space-y-4"
                >
                  {Object.entries(selectedTool.parameters).map(([key, param]) => (
                    <div key={key} className="space-y-2">
                      <Label className="text-xs flex items-center gap-2">
                        {key}
                        {param.required && (
                          <span className="text-rose-400">*</span>
                        )}
                        {param.enum && (
                          <Badge variant="outline" className="text-[10px] py-0">
                            enum
                          </Badge>
                        )}
                      </Label>
                      {param.enum ? (
                        <select
                          value={parameters[key] || ""}
                          onChange={(e) =>
                            setParameters((p) => ({ ...p, [key]: e.target.value }))
                          }
                          className="w-full h-9 px-3 rounded-md bg-zinc-800 border border-zinc-700 text-sm focus:outline-none focus:ring-2 focus:ring-violet-500"
                        >
                          {param.enum.map((opt) => (
                            <option key={opt} value={opt}>
                              {opt}
                            </option>
                          ))}
                        </select>
                      ) : (
                        <Input
                          type={param.type === "integer" ? "number" : "text"}
                          value={parameters[key] || ""}
                          onChange={(e) =>
                            setParameters((p) => ({ ...p, [key]: e.target.value }))
                          }
                          placeholder={param.description}
                          className="bg-zinc-800 border-zinc-700"
                        />
                      )}
                      <p className="text-[10px] text-zinc-500">{param.description}</p>
                    </div>
                  ))}

                  <Button
                    onClick={handleExecute}
                    disabled={executeMutation.isPending}
                    className="w-full bg-gradient-to-r from-violet-600 to-fuchsia-600 hover:from-violet-500 hover:to-fuchsia-500"
                  >
                    {executeMutation.isPending ? (
                      <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                    ) : (
                      <Play className="w-4 h-4 mr-2" />
                    )}
                    Execute Tool
                  </Button>
                </motion.div>
              ) : (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="text-center py-8 text-zinc-500"
                >
                  <Sparkles className="w-8 h-8 mx-auto mb-2 opacity-50" />
                  <p className="text-sm">Select a tool to get started</p>
                </motion.div>
              )}
            </AnimatePresence>
          </CardContent>
        </Card>

        {/* Result */}
        <Card className="bg-zinc-900/50 border-zinc-800">
          <CardHeader className="flex flex-row items-center justify-between">
            <div>
              <CardTitle className="text-sm">Response</CardTitle>
              <CardDescription>Tool execution result</CardDescription>
            </div>
            {latestResult?.raw && (
              <Button
                variant="ghost"
                size="sm"
                onClick={() => handleCopy(latestResult.raw || "")}
                className="h-8"
              >
                {copied ? (
                  <Check className="w-4 h-4 text-emerald-400" />
                ) : (
                  <Copy className="w-4 h-4" />
                )}
              </Button>
            )}
          </CardHeader>
          <CardContent>
            <ScrollArea className="h-[400px]">
              <AnimatePresence mode="wait">
                {executeMutation.isPending ? (
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="flex items-center justify-center py-16"
                  >
                    <div className="text-center">
                      <Loader2 className="w-8 h-8 animate-spin text-violet-400 mx-auto" />
                      <p className="text-sm text-zinc-500 mt-2">Executing...</p>
                    </div>
                  </motion.div>
                ) : latestResult ? (
                  <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="space-y-3"
                  >
                    {latestResult.error ? (
                      <div className="p-3 rounded-lg bg-rose-500/10 border border-rose-500/20">
                        <div className="flex items-center gap-2 text-rose-400 text-sm font-medium">
                          <AlertCircle className="w-4 h-4" />
                          Error
                        </div>
                        <p className="text-xs text-rose-300 mt-1">{latestResult.error}</p>
                      </div>
                    ) : (
                      <div className="p-3 rounded-lg bg-emerald-500/10 border border-emerald-500/20">
                        <div className="flex items-center gap-2 text-emerald-400 text-sm font-medium">
                          <Check className="w-4 h-4" />
                          Success
                        </div>
                      </div>
                    )}
                    <pre className="p-4 rounded-lg bg-zinc-800/80 border border-zinc-700/50 overflow-x-auto text-xs font-mono text-zinc-300">
                      {JSON.stringify(latestResult.result || latestResult.error, null, 2)}
                    </pre>
                  </motion.div>
                ) : (
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="text-center py-16 text-zinc-500"
                  >
                    <Terminal className="w-8 h-8 mx-auto mb-2 opacity-50" />
                    <p className="text-sm">Execute a tool to see results</p>
                  </motion.div>
                )}
              </AnimatePresence>
            </ScrollArea>
          </CardContent>
        </Card>
      </div>

      {/* Execution History */}
      {history.length > 0 && (
        <Card className="bg-zinc-900/50 border-zinc-800">
          <CardHeader>
            <CardTitle className="text-sm">Execution History</CardTitle>
            <CardDescription>Recent tool executions</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {history.map((entry, idx) => (
                <motion.div
                  key={entry.id}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: idx * 0.05 }}
                  className="flex items-center justify-between p-3 rounded-lg bg-zinc-800/50 border border-zinc-700/50"
                >
                  <div className="flex items-center gap-3">
                    <Badge
                      variant="outline"
                      className={
                        entry.result.error
                          ? "bg-rose-500/10 text-rose-400 border-rose-500/20"
                          : "bg-emerald-500/10 text-emerald-400 border-emerald-500/20"
                      }
                    >
                      {entry.result.error ? "Error" : "Success"}
                    </Badge>
                    <span className="font-mono text-sm">{entry.tool}</span>
                  </div>
                  <span className="text-xs text-zinc-500">
                    {entry.timestamp.toLocaleTimeString()}
                  </span>
                </motion.div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
