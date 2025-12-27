"use client";

import { useState, useRef, useEffect } from "react";
import { useMutation } from "@tanstack/react-query";
import { motion, AnimatePresence } from "framer-motion";
import {
  MessageSquare,
  Send,
  Loader2,
  Sparkles,
  ExternalLink,
  Zap,
  Brain,
  ChevronRight,
} from "lucide-react";
import { api, ChatResponse } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { cn } from "@/lib/utils";

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: ChatResponse["sources"];
  followUp?: string[];
  mode?: "quick" | "deep";
  timestamp: Date;
}

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [mode, setMode] = useState<"quick" | "deep">("quick");
  const scrollRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // Auto-scroll to bottom
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  // Chat mutation
  const chatMutation = useMutation({
    mutationFn: (message: string) => api.chat(message, mode),
    onSuccess: (response) => {
      const assistantMessage: Message = {
        id: Date.now().toString(),
        role: "assistant",
        content: response.answer || "Here are the relevant sources I found:",
        sources: response.sources,
        followUp: response.follow_up_questions,
        mode: response.mode,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, assistantMessage]);
    },
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || chatMutation.isPending) return;

    // Add user message
    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content: input.trim(),
      timestamp: new Date(),
    };
    setMessages((prev) => [...prev, userMessage]);
    setInput("");

    // Send to API
    chatMutation.mutate(input.trim());
  };

  const handleFollowUp = (question: string) => {
    setInput(question);
    inputRef.current?.focus();
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  return (
    <div className="flex flex-col h-[calc(100vh-4rem)]">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div>
          <h1 className="text-2xl font-bold tracking-tight flex items-center gap-3">
            <MessageSquare className="w-7 h-7 text-violet-400" />
            Ask ContextPilot
          </h1>
          <p className="text-zinc-500 mt-1">
            Query your indexed documentation using natural language
          </p>
        </div>

        {/* Mode Toggle */}
        <div className="flex items-center gap-2 p-1 rounded-lg bg-zinc-800/50 border border-zinc-700/50">
          <button
            onClick={() => setMode("quick")}
            className={cn(
              "flex items-center gap-2 px-3 py-1.5 rounded-md text-sm font-medium transition-all",
              mode === "quick"
                ? "bg-violet-500/20 text-violet-400"
                : "text-zinc-400 hover:text-zinc-200"
            )}
          >
            <Zap className="w-4 h-4" />
            Quick
          </button>
          <button
            onClick={() => setMode("deep")}
            className={cn(
              "flex items-center gap-2 px-3 py-1.5 rounded-md text-sm font-medium transition-all",
              mode === "deep"
                ? "bg-fuchsia-500/20 text-fuchsia-400"
                : "text-zinc-400 hover:text-zinc-200"
            )}
          >
            <Brain className="w-4 h-4" />
            Deep
          </button>
        </div>
      </div>

      {/* Chat Messages */}
      <Card className="flex-1 bg-zinc-900/50 border-zinc-800 overflow-hidden">
        <ScrollArea className="h-full p-4" ref={scrollRef}>
          <div className="space-y-4">
            <AnimatePresence>
              {messages.length === 0 ? (
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="flex flex-col items-center justify-center h-[400px] text-center"
                >
                  <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-violet-500/20 to-fuchsia-500/20 flex items-center justify-center mb-4">
                    <Sparkles className="w-8 h-8 text-violet-400" />
                  </div>
                  <h2 className="text-lg font-semibold mb-2">Start a conversation</h2>
                  <p className="text-zinc-500 text-sm max-w-md">
                    Ask questions about your indexed documentation. I&apos;ll search through your
                    docs and provide relevant answers with sources.
                  </p>
                  <div className="flex flex-wrap gap-2 mt-6 justify-center">
                    {[
                      "How do I use streaming with the Gemini API?",
                      "What authentication methods does Firebase support?",
                      "How do I set up React Query?",
                    ].map((suggestion) => (
                      <button
                        key={suggestion}
                        onClick={() => handleFollowUp(suggestion)}
                        className="px-3 py-1.5 rounded-full bg-zinc-800/50 border border-zinc-700/50 text-xs text-zinc-400 hover:text-zinc-200 hover:border-zinc-600 transition-all"
                      >
                        {suggestion}
                      </button>
                    ))}
                  </div>
                </motion.div>
              ) : (
                messages.map((message, idx) => (
                  <motion.div
                    key={message.id}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: idx * 0.05 }}
                    className={cn(
                      "flex",
                      message.role === "user" ? "justify-end" : "justify-start"
                    )}
                  >
                    <div
                      className={cn(
                        "max-w-[80%] rounded-2xl px-4 py-3",
                        message.role === "user"
                          ? "bg-violet-500/20 text-violet-100"
                          : "bg-zinc-800/50 border border-zinc-700/50"
                      )}
                    >
                      <p className="text-sm whitespace-pre-wrap">{message.content}</p>

                      {/* Sources */}
                      {message.sources && message.sources.length > 0 && (
                        <div className="mt-3 pt-3 border-t border-zinc-700/50">
                          <p className="text-xs text-zinc-500 mb-2">Sources:</p>
                          <div className="space-y-1">
                            {message.sources.map((source, sidx) => (
                              <a
                                key={sidx}
                                href={source.url}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="flex items-center gap-2 p-2 rounded-lg bg-zinc-900/50 hover:bg-zinc-800/50 transition-colors group"
                              >
                                <div className="flex-1 min-w-0">
                                  <p className="text-xs font-medium truncate group-hover:text-violet-400 transition-colors">
                                    {source.title || source.url}
                                  </p>
                                  {source.content_preview && (
                                    <p className="text-[10px] text-zinc-500 truncate mt-0.5">
                                      {source.content_preview}
                                    </p>
                                  )}
                                </div>
                                <Badge variant="outline" className="text-[10px] shrink-0">
                                  {Math.round(source.score * 100)}%
                                </Badge>
                                <ExternalLink className="w-3 h-3 text-zinc-500 shrink-0" />
                              </a>
                            ))}
                          </div>
                        </div>
                      )}

                      {/* Follow-up Questions */}
                      {message.followUp && message.followUp.length > 0 && (
                        <div className="mt-3 pt-3 border-t border-zinc-700/50">
                          <p className="text-xs text-zinc-500 mb-2">Follow-up questions:</p>
                          <div className="space-y-1">
                            {message.followUp.map((q, qidx) => (
                              <button
                                key={qidx}
                                onClick={() => handleFollowUp(q)}
                                className="flex items-center gap-2 w-full text-left p-2 rounded-lg bg-zinc-900/50 hover:bg-zinc-800/50 transition-colors text-xs text-zinc-400 hover:text-zinc-200"
                              >
                                <ChevronRight className="w-3 h-3" />
                                {q}
                              </button>
                            ))}
                          </div>
                        </div>
                      )}

                      {/* Mode badge */}
                      {message.mode && (
                        <div className="mt-2 flex justify-end">
                          <Badge
                            variant="outline"
                            className={cn(
                              "text-[10px]",
                              message.mode === "deep"
                                ? "bg-fuchsia-500/10 text-fuchsia-400 border-fuchsia-500/20"
                                : "bg-violet-500/10 text-violet-400 border-violet-500/20"
                            )}
                          >
                            {message.mode === "deep" ? "Deep Search" : "Quick Search"}
                          </Badge>
                        </div>
                      )}
                    </div>
                  </motion.div>
                ))
              )}
            </AnimatePresence>

            {/* Loading indicator */}
            {chatMutation.isPending && (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="flex justify-start"
              >
                <div className="bg-zinc-800/50 border border-zinc-700/50 rounded-2xl px-4 py-3">
                  <div className="flex items-center gap-2 text-zinc-400">
                    <Loader2 className="w-4 h-4 animate-spin" />
                    <span className="text-sm">
                      {mode === "deep" ? "Synthesizing answer..." : "Searching documentation..."}
                    </span>
                  </div>
                </div>
              </motion.div>
            )}
          </div>
        </ScrollArea>
      </Card>

      {/* Input */}
      <form onSubmit={handleSubmit} className="mt-4">
        <div className="relative">
          <textarea
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask about your documentation..."
            rows={1}
            className="w-full px-4 py-3 pr-12 rounded-xl bg-zinc-800/50 border border-zinc-700/50 focus:border-violet-500/50 focus:ring-1 focus:ring-violet-500/50 resize-none text-sm placeholder:text-zinc-500 focus:outline-none"
          />
          <Button
            type="submit"
            size="sm"
            disabled={!input.trim() || chatMutation.isPending}
            className="absolute right-2 top-1/2 -translate-y-1/2 h-8 w-8 p-0 bg-violet-500 hover:bg-violet-400"
          >
            {chatMutation.isPending ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <Send className="w-4 h-4" />
            )}
          </Button>
        </div>
        <p className="text-xs text-zinc-500 mt-2 text-center">
          {mode === "deep"
            ? "Deep mode synthesizes an answer from multiple sources"
            : "Quick mode returns relevant documentation snippets"}
        </p>
      </form>
    </div>
  );
}
