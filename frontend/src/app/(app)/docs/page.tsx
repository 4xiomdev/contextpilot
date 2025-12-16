"use client";

import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "@/lib/api";
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
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  FileText,
  Plus,
  Loader2,
  Sparkles,
  Layers,
  RefreshCw,
} from "lucide-react";

export default function DocsPage() {
  const [urlPrefix, setUrlPrefix] = useState("");
  const [title, setTitle] = useState("");
  const [dialogOpen, setDialogOpen] = useState(false);
  const queryClient = useQueryClient();

  const { data: docs, isLoading } = useQuery({
    queryKey: ["normalized"],
    queryFn: () => api.getNormalizedDocs(),
  });

  const normalizeMutation = useMutation({
    mutationFn: () => api.normalize(urlPrefix, title),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["normalized"] });
      queryClient.invalidateQueries({ queryKey: ["stats"] });
      setDialogOpen(false);
      setUrlPrefix("");
      setTitle("");
    },
  });

  const handleNormalize = (e: React.FormEvent) => {
    e.preventDefault();
    if (urlPrefix.trim() && title.trim()) {
      normalizeMutation.mutate();
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Normalized Docs</h1>
          <p className="text-zinc-500 mt-1">
            Synthesized documentation optimized for AI retrieval
          </p>
        </div>
        <Dialog open={dialogOpen} onOpenChange={setDialogOpen}>
          <DialogTrigger asChild>
            <Button className="bg-violet-600 hover:bg-violet-700">
              <Plus className="w-4 h-4 mr-2" />
              Build New Doc
            </Button>
          </DialogTrigger>
          <DialogContent className="bg-zinc-900 border-zinc-800">
            <DialogHeader>
              <DialogTitle>Build Normalized Document</DialogTitle>
              <DialogDescription>
                Synthesize indexed chunks into a clean, structured document using AI.
              </DialogDescription>
            </DialogHeader>
            <form onSubmit={handleNormalize} className="space-y-4 mt-4">
              <div>
                <label className="text-sm text-zinc-400 mb-1 block">URL Prefix</label>
                <Input
                  placeholder="https://docs.example.com/api"
                  value={urlPrefix}
                  onChange={(e) => setUrlPrefix(e.target.value)}
                  className="bg-zinc-800 border-zinc-700"
                />
                <p className="text-xs text-zinc-500 mt-1">
                  All indexed chunks matching this prefix will be included
                </p>
              </div>
              <div>
                <label className="text-sm text-zinc-400 mb-1 block">Document Title</label>
                <Input
                  placeholder="Gemini API Documentation"
                  value={title}
                  onChange={(e) => setTitle(e.target.value)}
                  className="bg-zinc-800 border-zinc-700"
                />
              </div>
              <div className="flex justify-end gap-2">
                <Button
                  type="button"
                  variant="outline"
                  onClick={() => setDialogOpen(false)}
                  className="border-zinc-700"
                >
                  Cancel
                </Button>
                <Button
                  type="submit"
                  disabled={normalizeMutation.isPending || !urlPrefix.trim() || !title.trim()}
                  className="bg-violet-600 hover:bg-violet-700"
                >
                  {normalizeMutation.isPending ? (
                    <>
                      <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                      Building...
                    </>
                  ) : (
                    <>
                      <Sparkles className="w-4 h-4 mr-2" />
                      Build Document
                    </>
                  )}
                </Button>
              </div>
              {normalizeMutation.isError && (
                <p className="text-sm text-red-400">
                  Error: {(normalizeMutation.error as Error).message}
                </p>
              )}
            </form>
          </DialogContent>
        </Dialog>
      </div>

      {/* Info Card */}
      <Card className="bg-gradient-to-r from-violet-500/10 to-fuchsia-500/10 border-violet-500/20">
        <CardContent className="py-4">
          <div className="flex items-center gap-4">
            <div className="w-10 h-10 rounded-lg bg-violet-500/20 flex items-center justify-center">
              <Sparkles className="w-5 h-5 text-violet-400" />
            </div>
            <div>
              <p className="font-medium">What are Normalized Docs?</p>
              <p className="text-sm text-zinc-400">
                Normalized docs are AI-synthesized documents that combine multiple raw chunks into
                clean, structured references optimized for embedding and retrieval.
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Docs List */}
      <Card className="bg-zinc-900/50 border-zinc-800">
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle>Your Documents</CardTitle>
              <CardDescription>Normalized documentation in your context layer</CardDescription>
            </div>
            <Button
              variant="outline"
              size="icon"
              onClick={() => queryClient.invalidateQueries({ queryKey: ["normalized"] })}
              className="border-zinc-700"
            >
              <RefreshCw className="w-4 h-4" />
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <div className="text-center py-8 text-zinc-500">
              <Loader2 className="w-6 h-6 mx-auto animate-spin" />
            </div>
          ) : docs?.documents && docs.documents.length > 0 ? (
            <Table>
              <TableHeader>
                <TableRow className="border-zinc-800 hover:bg-transparent">
                  <TableHead className="text-zinc-400">Title</TableHead>
                  <TableHead className="text-zinc-400">URL Prefix</TableHead>
                  <TableHead className="text-zinc-400">Source Chunks</TableHead>
                  <TableHead className="text-zinc-400">Created</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {docs.documents.map((doc) => (
                  <TableRow key={doc.id} className="border-zinc-800">
                    <TableCell>
                      <div className="flex items-center gap-2">
                        <FileText className="w-4 h-4 text-violet-400" />
                        <span className="font-medium">{doc.title}</span>
                      </div>
                    </TableCell>
                    <TableCell className="font-mono text-sm text-zinc-400 max-w-xs truncate">
                      {doc.url_prefix}
                    </TableCell>
                    <TableCell>
                      <Badge className="bg-zinc-800 text-zinc-300 border-zinc-700">
                        <Layers className="w-3 h-3 mr-1" />
                        {doc.raw_chunk_count} chunks
                      </Badge>
                    </TableCell>
                    <TableCell className="text-zinc-500 text-sm">
                      {new Date(doc.created_at * 1000).toLocaleDateString()}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          ) : (
            <div className="text-center py-12 text-zinc-500">
              <FileText className="w-12 h-12 mx-auto mb-4 opacity-50" />
              <p className="text-lg">No normalized documents yet</p>
              <p className="text-sm mt-1">
                Click &quot;Build New Doc&quot; to create your first normalized document
              </p>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
