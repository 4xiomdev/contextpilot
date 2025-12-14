"use client";

import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { api, SearchResult } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Search,
  Loader2,
  FileText,
  ExternalLink,
  Sparkles,
  Layers,
} from "lucide-react";

function ResultCard({ result, index }: { result: SearchResult; index: number }) {
  const [expanded, setExpanded] = useState(false);

  return (
    <Card
      className={`bg-zinc-900/50 border-zinc-800 hover:border-zinc-700 transition-all cursor-pointer ${
        expanded ? "ring-1 ring-violet-500/50" : ""
      }`}
      onClick={() => setExpanded(!expanded)}
    >
      <CardHeader className="pb-2">
        <div className="flex items-start justify-between gap-4">
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 mb-1">
              <span className="text-xs font-mono text-zinc-500">#{index + 1}</span>
              <Badge
                className={
                  result.type === "normalized"
                    ? "bg-violet-500/10 text-violet-400 border-violet-500/20"
                    : "bg-zinc-500/10 text-zinc-400 border-zinc-500/20"
                }
              >
                {result.type === "normalized" ? (
                  <Sparkles className="w-3 h-3 mr-1" />
                ) : (
                  <Layers className="w-3 h-3 mr-1" />
                )}
                {result.type}
              </Badge>
              <span className="text-xs text-zinc-500">
                Score: {(result.score * 100).toFixed(1)}%
              </span>
            </div>
            <CardTitle className="text-base truncate">{result.title || "Untitled"}</CardTitle>
            <CardDescription className="truncate font-mono text-xs">
              {result.url}
            </CardDescription>
          </div>
          <a
            href={result.url}
            target="_blank"
            rel="noopener noreferrer"
            onClick={(e) => e.stopPropagation()}
            className="text-zinc-500 hover:text-zinc-300 transition-colors"
          >
            <ExternalLink className="w-4 h-4" />
          </a>
        </div>
      </CardHeader>
      <CardContent>
        <div
          className={`text-sm text-zinc-400 whitespace-pre-wrap font-mono ${
            expanded ? "" : "line-clamp-3"
          }`}
        >
          {result.content}
        </div>
        {!expanded && result.content.length > 300 && (
          <p className="text-xs text-violet-400 mt-2">Click to expand...</p>
        )}
      </CardContent>
    </Card>
  );
}

export default function SearchPage() {
  const [query, setQuery] = useState("");
  const [urlFilter, setUrlFilter] = useState("");
  const [results, setResults] = useState<SearchResult[]>([]);
  const [source, setSource] = useState<string>("");

  const searchMutation = useMutation({
    mutationFn: () => api.search(query, 20, urlFilter),
    onSuccess: (data) => {
      setResults(data.results);
      setSource(data.source);
    },
  });

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    if (query.trim()) {
      searchMutation.mutate();
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Search</h1>
        <p className="text-zinc-500 mt-1">
          Semantic search across your indexed documentation
        </p>
      </div>

      {/* Search Form */}
      <Card className="bg-zinc-900/50 border-zinc-800">
        <CardContent className="pt-6">
          <form onSubmit={handleSearch} className="space-y-4">
            <div className="flex gap-2">
              <div className="flex-1 relative">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-zinc-500" />
                <Input
                  placeholder="Search documentation... (e.g., 'How to use Gemini API')"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  className="pl-10 bg-zinc-800/50 border-zinc-700 text-lg h-12"
                />
              </div>
              <Button
                type="submit"
                disabled={searchMutation.isPending || !query.trim()}
                className="bg-violet-600 hover:bg-violet-700 h-12 px-6"
              >
                {searchMutation.isPending ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  "Search"
                )}
              </Button>
            </div>
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2">
                <label className="text-sm text-zinc-500">URL Filter:</label>
                <Input
                  placeholder="https://docs.example.com"
                  value={urlFilter}
                  onChange={(e) => setUrlFilter(e.target.value)}
                  className="w-64 bg-zinc-800/50 border-zinc-700 text-sm"
                />
              </div>
            </div>
          </form>
        </CardContent>
      </Card>

      {/* Results */}
      {results.length > 0 && (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-semibold">
              {results.length} Results
              <span className="text-sm font-normal text-zinc-500 ml-2">
                from {source} index
              </span>
            </h2>
          </div>
          <div className="grid gap-4">
            {results.map((result, i) => (
              <ResultCard key={i} result={result} index={i} />
            ))}
          </div>
        </div>
      )}

      {searchMutation.isSuccess && results.length === 0 && (
        <Card className="bg-zinc-900/50 border-zinc-800">
          <CardContent className="py-12 text-center">
            <FileText className="w-12 h-12 mx-auto mb-4 text-zinc-600" />
            <p className="text-zinc-400">No results found</p>
            <p className="text-sm text-zinc-500 mt-1">
              Try different keywords or remove the URL filter
            </p>
          </CardContent>
        </Card>
      )}

      {!searchMutation.isSuccess && results.length === 0 && (
        <Card className="bg-zinc-900/50 border-zinc-800">
          <CardContent className="py-12 text-center">
            <Search className="w-12 h-12 mx-auto mb-4 text-zinc-600" />
            <p className="text-zinc-400">Enter a search query to get started</p>
            <p className="text-sm text-zinc-500 mt-1">
              Search uses semantic similarity to find relevant documentation
            </p>
          </CardContent>
        </Card>
      )}
    </div>
  );
}


