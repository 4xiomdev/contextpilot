"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Copy, Check, Terminal } from "lucide-react";

export default function SetupPage() {
  const [copied, setCopied] = useState(false);
  const [keyCopied, setKeyCopied] = useState(false);

  const mcpConfig = `{
  "mcpServers": {
    "context-pilot": {
      "command": "docker",
      "args": [
        "run",
        "-i",
        "--rm",
        "-e", "API_KEY=YOUR_API_KEY",
        "contextpilot/mcp-server"
      ]
    }
  }
}`;

  const apiKey = "cp_sk_live_59283401928340192830";

  const copyConfig = () => {
    navigator.clipboard.writeText(mcpConfig);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const copyKey = () => {
    navigator.clipboard.writeText(apiKey);
    setKeyCopied(true);
    setTimeout(() => setKeyCopied(false), 2000);
  };

  return (
    <div className="space-y-6 max-w-4xl">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Connect via MCP</h1>
        <p className="text-zinc-500 mt-1">
          Configure your LLM client to use ContextPilot as a tool server.
        </p>
      </div>

      <Card className="bg-zinc-900/50 border-zinc-800">
        <CardHeader>
          <CardTitle>1. Get your API Key</CardTitle>
          <CardDescription>
            Use this key to authenticate your MCP client.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex gap-2">
            <Input 
              readOnly 
              value={apiKey} 
              className="bg-zinc-950 font-mono text-zinc-400 border-zinc-700"
            />
            <Button variant="outline" onClick={copyKey} className="border-zinc-700 hover:bg-zinc-800">
              {keyCopied ? <Check className="w-4 h-4 text-emerald-500" /> : <Copy className="w-4 h-4" />}
            </Button>
          </div>
        </CardContent>
      </Card>

      <Card className="bg-zinc-900/50 border-zinc-800">
        <CardHeader>
          <CardTitle>2. Configure Claude Desktop</CardTitle>
          <CardDescription>
            Add this configuration to your <code className="bg-zinc-800 px-1 rounded">claude_desktop_config.json</code>
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="relative group">
            <pre className="bg-zinc-950 p-4 rounded-lg overflow-x-auto border border-zinc-800">
              <code className="text-sm font-mono text-zinc-300">{mcpConfig}</code>
            </pre>
            <Button 
              size="sm" 
              variant="secondary" 
              className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity"
              onClick={copyConfig}
            >
              {copied ? (
                <>
                  <Check className="w-3 h-3 mr-2" />
                  Copied
                </>
              ) : (
                <>
                  <Copy className="w-3 h-3 mr-2" />
                  Copy Config
                </>
              )}
            </Button>
          </div>
        </CardContent>
      </Card>
      
      <div className="flex items-center gap-2 p-4 rounded-lg bg-blue-500/10 text-blue-400 border border-blue-500/20">
        <Terminal className="w-5 h-5" />
        <p className="text-sm">
          Restart your Claude Desktop app after saving the configuration file.
        </p>
      </div>
    </div>
  );
}
