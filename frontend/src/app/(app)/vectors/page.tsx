"use client";

import { useState, useMemo, useCallback } from "react";
import { useQuery } from "@tanstack/react-query";
import { motion, AnimatePresence } from "framer-motion";
import {
  Boxes,
  Loader2,
  Search,
  Layers,
  Maximize2,
  RotateCcw,
  Info,
  Database,
} from "lucide-react";
import { api, VectorItem } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { cn } from "@/lib/utils";

// UMAP for dimension reduction - we'll implement a simple version
function computeUMAP(vectors: number[][], targetDim: 2 | 3 = 2): number[][] {
  if (vectors.length === 0) return [];

  // Simple PCA-like projection for demo
  // In production, you'd use a proper UMAP library
  const result: number[][] = [];

  for (const vec of vectors) {
    if (targetDim === 2) {
      // Project to 2D using first two principal components (simplified)
      const x = vec.slice(0, 256).reduce((a, b) => a + b, 0) / 256;
      const y = vec.slice(256, 512).reduce((a, b) => a + b, 0) / 256;
      result.push([x * 100, y * 100]);
    } else {
      const x = vec.slice(0, 256).reduce((a, b) => a + b, 0) / 256;
      const y = vec.slice(256, 512).reduce((a, b) => a + b, 0) / 256;
      const z = vec.slice(512, 768).reduce((a, b) => a + b, 0) / 256;
      result.push([x * 100, y * 100, z * 100]);
    }
  }

  return result;
}

// Generate color from URL for consistent coloring
function urlToColor(url: string): string {
  let hash = 0;
  for (let i = 0; i < url.length; i++) {
    hash = ((hash << 5) - hash) + url.charCodeAt(i);
    hash = hash & hash;
  }

  const hue = Math.abs(hash) % 360;
  return `hsl(${hue}, 70%, 60%)`;
}

interface VectorPoint {
  original: VectorItem;
  projected: number[];
  color: string;
}

export default function VectorsPage() {
  const [selectedVector, setSelectedVector] = useState<VectorItem | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [viewMode, setViewMode] = useState<"2d" | "3d">("2d");
  const [hoveredPoint, setHoveredPoint] = useState<VectorPoint | null>(null);

  // Fetch vector stats
  const { data: statsData, isLoading: statsLoading } = useQuery({
    queryKey: ["vector-stats"],
    queryFn: () => api.getVectorStats(),
  });

  // Fetch vector samples
  const { data: samplesData, isLoading: samplesLoading, refetch } = useQuery({
    queryKey: ["vector-samples"],
    queryFn: () => api.getVectorSamples(200),
  });

  // Search vectors
  const { data: searchData, isLoading: searchLoading } = useQuery({
    queryKey: ["vector-search", searchQuery],
    queryFn: () => api.queryVectors(searchQuery, 20, true),
    enabled: searchQuery.length > 2,
  });

  // Process vectors for visualization
  const processedPoints = useMemo<VectorPoint[]>(() => {
    const vectors = searchQuery.length > 2 ? searchData?.results : samplesData?.vectors;
    if (!vectors || vectors.length === 0) return [];

    const rawVectors = vectors
      .filter((v) => v.values && v.values.length > 0)
      .map((v) => v.values!);

    if (rawVectors.length === 0) return [];

    const projected = computeUMAP(rawVectors, viewMode === "3d" ? 3 : 2);

    return vectors
      .filter((v) => v.values && v.values.length > 0)
      .map((v, i) => ({
        original: v,
        projected: projected[i] || [0, 0],
        color: urlToColor(v.url),
      }));
  }, [samplesData, searchData, searchQuery, viewMode]);

  // Group by source URL for legend
  const sourceGroups = useMemo(() => {
    const groups: Record<string, { color: string; count: number }> = {};
    for (const point of processedPoints) {
      const domain = new URL(point.original.url).hostname;
      if (!groups[domain]) {
        groups[domain] = { color: point.color, count: 0 };
      }
      groups[domain].count++;
    }
    return Object.entries(groups).sort((a, b) => b[1].count - a[1].count);
  }, [processedPoints]);

  const isLoading = statsLoading || samplesLoading;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight flex items-center gap-3">
            <Boxes className="w-7 h-7 text-violet-400" />
            Vector Explorer
          </h1>
          <p className="text-zinc-500 mt-1">
            Visualize your embedding space in 2D and 3D
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => refetch()}
            disabled={isLoading}
          >
            <RotateCcw className={cn("w-4 h-4 mr-2", isLoading && "animate-spin")} />
            Refresh
          </Button>
        </div>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card className="bg-zinc-900/50 border-zinc-800">
          <CardContent className="pt-6">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-lg bg-violet-500/10 flex items-center justify-center">
                <Database className="w-5 h-5 text-violet-400" />
              </div>
              <div>
                <p className="text-2xl font-bold">
                  {statsData?.index.total_vectors?.toLocaleString() || "—"}
                </p>
                <p className="text-xs text-zinc-500">Total Vectors</p>
              </div>
            </div>
          </CardContent>
        </Card>
        <Card className="bg-zinc-900/50 border-zinc-800">
          <CardContent className="pt-6">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-lg bg-fuchsia-500/10 flex items-center justify-center">
                <Layers className="w-5 h-5 text-fuchsia-400" />
              </div>
              <div>
                <p className="text-2xl font-bold">
                  {statsData?.index.dimension || "768"}
                </p>
                <p className="text-xs text-zinc-500">Dimensions</p>
              </div>
            </div>
          </CardContent>
        </Card>
        <Card className="bg-zinc-900/50 border-zinc-800">
          <CardContent className="pt-6">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-lg bg-emerald-500/10 flex items-center justify-center">
                <Boxes className="w-5 h-5 text-emerald-400" />
              </div>
              <div>
                <p className="text-2xl font-bold">
                  {Object.keys(statsData?.index.namespaces || {}).length || "—"}
                </p>
                <p className="text-xs text-zinc-500">Namespaces</p>
              </div>
            </div>
          </CardContent>
        </Card>
        <Card className="bg-zinc-900/50 border-zinc-800">
          <CardContent className="pt-6">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-lg bg-amber-500/10 flex items-center justify-center">
                <Search className="w-5 h-5 text-amber-400" />
              </div>
              <div>
                <p className="text-2xl font-bold">{processedPoints.length}</p>
                <p className="text-xs text-zinc-500">Visible Points</p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Visualization */}
        <div className="lg:col-span-3">
          <Card className="bg-zinc-900/50 border-zinc-800 h-[600px]">
            <CardHeader className="flex flex-row items-center justify-between pb-2">
              <div>
                <CardTitle className="text-sm">Embedding Space</CardTitle>
                <CardDescription>
                  {searchQuery.length > 2
                    ? `Search results for "${searchQuery}"`
                    : "Random sample of vectors"}
                </CardDescription>
              </div>
              <Tabs value={viewMode} onValueChange={(v) => setViewMode(v as "2d" | "3d")}>
                <TabsList className="bg-zinc-800/50">
                  <TabsTrigger value="2d" className="text-xs">2D</TabsTrigger>
                  <TabsTrigger value="3d" className="text-xs">3D</TabsTrigger>
                </TabsList>
              </Tabs>
            </CardHeader>
            <CardContent className="h-[calc(100%-80px)]">
              {/* Search bar */}
              <div className="mb-4">
                <div className="relative">
                  <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-zinc-500" />
                  <Input
                    placeholder="Search to highlight similar vectors..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    className="pl-9 bg-zinc-800/50 border-zinc-700"
                  />
                  {searchLoading && (
                    <Loader2 className="absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 animate-spin text-violet-400" />
                  )}
                </div>
              </div>

              {/* Canvas */}
              <div className="relative h-[calc(100%-60px)] bg-zinc-950 rounded-lg border border-zinc-800 overflow-hidden">
                {samplesLoading ? (
                  <div className="absolute inset-0 flex items-center justify-center">
                    <Loader2 className="w-8 h-8 animate-spin text-violet-400" />
                  </div>
                ) : processedPoints.length === 0 ? (
                  <div className="absolute inset-0 flex items-center justify-center text-zinc-500">
                    <div className="text-center">
                      <Boxes className="w-12 h-12 mx-auto mb-2 opacity-50" />
                      <p>No vectors to display</p>
                      <p className="text-xs mt-1">Index some documentation first</p>
                    </div>
                  </div>
                ) : viewMode === "2d" ? (
                  <svg
                    className="w-full h-full"
                    viewBox="-150 -150 300 300"
                    preserveAspectRatio="xMidYMid meet"
                  >
                    {/* Grid */}
                    <defs>
                      <pattern id="grid" width="20" height="20" patternUnits="userSpaceOnUse">
                        <path d="M 20 0 L 0 0 0 20" fill="none" stroke="rgba(255,255,255,0.05)" strokeWidth="0.5" />
                      </pattern>
                    </defs>
                    <rect x="-150" y="-150" width="300" height="300" fill="url(#grid)" />

                    {/* Points */}
                    {processedPoints.map((point, i) => (
                      <motion.circle
                        key={point.original.id}
                        initial={{ opacity: 0, scale: 0 }}
                        animate={{ opacity: 0.8, scale: 1 }}
                        transition={{ delay: i * 0.002 }}
                        cx={point.projected[0]}
                        cy={point.projected[1]}
                        r={hoveredPoint?.original.id === point.original.id ? 6 : 4}
                        fill={point.color}
                        stroke={selectedVector?.id === point.original.id ? "white" : "none"}
                        strokeWidth={2}
                        className="cursor-pointer transition-all duration-200"
                        onMouseEnter={() => setHoveredPoint(point)}
                        onMouseLeave={() => setHoveredPoint(null)}
                        onClick={() => setSelectedVector(point.original)}
                      />
                    ))}

                    {/* Query point */}
                    {searchData?.query_vector && (
                      <circle
                        cx={0}
                        cy={0}
                        r={8}
                        fill="white"
                        stroke="rgba(139, 92, 246, 0.8)"
                        strokeWidth={3}
                      />
                    )}
                  </svg>
                ) : (
                  <div className="w-full h-full flex items-center justify-center text-zinc-500">
                    <div className="text-center">
                      <Maximize2 className="w-12 h-12 mx-auto mb-2 opacity-50" />
                      <p>3D visualization requires Three.js</p>
                      <p className="text-xs mt-1">Coming soon...</p>
                    </div>
                  </div>
                )}

                {/* Hover tooltip */}
                <AnimatePresence>
                  {hoveredPoint && (
                    <motion.div
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: 10 }}
                      className="absolute bottom-4 left-4 right-4 p-3 rounded-lg bg-zinc-900/95 border border-zinc-700/50 backdrop-blur-sm"
                    >
                      <p className="font-medium text-sm truncate">{hoveredPoint.original.title}</p>
                      <p className="text-xs text-zinc-500 truncate mt-1">{hoveredPoint.original.url}</p>
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Sidebar */}
        <div className="space-y-4">
          {/* Legend */}
          <Card className="bg-zinc-900/50 border-zinc-800">
            <CardHeader>
              <CardTitle className="text-sm">Sources</CardTitle>
              <CardDescription>Color-coded by domain</CardDescription>
            </CardHeader>
            <CardContent>
              <ScrollArea className="h-[200px]">
                <div className="space-y-2">
                  {sourceGroups.map(([domain, { color, count }]) => (
                    <div key={domain} className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <div
                          className="w-3 h-3 rounded-full"
                          style={{ backgroundColor: color }}
                        />
                        <span className="text-xs text-zinc-400 truncate max-w-[120px]">
                          {domain}
                        </span>
                      </div>
                      <Badge variant="outline" className="text-[10px]">
                        {count}
                      </Badge>
                    </div>
                  ))}
                </div>
              </ScrollArea>
            </CardContent>
          </Card>

          {/* Selected Vector Details */}
          <Card className="bg-zinc-900/50 border-zinc-800">
            <CardHeader>
              <CardTitle className="text-sm">Selected Vector</CardTitle>
              <CardDescription>Click a point to see details</CardDescription>
            </CardHeader>
            <CardContent>
              {selectedVector ? (
                <div className="space-y-3">
                  <div>
                    <p className="text-xs text-zinc-500">Title</p>
                    <p className="text-sm font-medium">{selectedVector.title}</p>
                  </div>
                  <div>
                    <p className="text-xs text-zinc-500">URL</p>
                    <a
                      href={selectedVector.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-xs text-violet-400 hover:underline break-all"
                    >
                      {selectedVector.url}
                    </a>
                  </div>
                  <div>
                    <p className="text-xs text-zinc-500">Score</p>
                    <Badge variant="outline">{(selectedVector.score * 100).toFixed(1)}%</Badge>
                  </div>
                  {selectedVector.content_preview && (
                    <div>
                      <p className="text-xs text-zinc-500">Preview</p>
                      <p className="text-xs text-zinc-400 line-clamp-4">
                        {selectedVector.content_preview}
                      </p>
                    </div>
                  )}
                </div>
              ) : (
                <div className="text-center py-4 text-zinc-500">
                  <Info className="w-6 h-6 mx-auto mb-2 opacity-50" />
                  <p className="text-xs">Select a vector point</p>
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
